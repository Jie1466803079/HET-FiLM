"""
IntentConditionedAttention (M1) — candidate-conditional attention layer.

Probe-stage minimum implementation per the v1 IBF spec, option B
(REPLACEMENT M1: fusion-into-encoder is bypassed when this layer is active;
text only enters the model via g_f_eff at the head and via mu inside this
attention).

For each scored edge (fund f, candidate stock s):
    mu(f, c)    = (W_Q_mu  g_f_eff)^T (W_K_mu  h_c) / sqrt(d)
    a(f, c)     = (W_Q_alf h_f      )^T (W_K_alf h_c) / sqrt(d) + lambda * mu(f, c)
    alpha(f, c) = softmax_{c in C(f, s)}( a(f, c) )
    z(f, s)     = sum_c alpha(f, c) * (W_V h_c)

C(f, s) = holdings(f) ∪ siblings_via_mgmt_company(f) ∪ {s}.
Safety cap applied at attention time via top-mu selection (-inf mask on rest).

Null cells (delta_t = -1): g_f_eff uses a learned null token instead of zeros.
This is a deliberate departure from ProspectusTextFusion's hard-zero staleness
gate; under M1, null cells flow through the attention with the learned constant,
and stratified test A's "null-only ~ no-text" expectation will reflect what
the null token learned rather than holding by construction.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


class IntentConditionedAttention(nn.Module):
    """M1 attention layer (probe stage).

    Inputs (per scored edge batch of size E):
        text_features: dict with
            - abs_emb (B_f, 1024): per-FUND prospectus embedding (B_f = N_funds in snapshot)
            - delta_t (B_f,) long: staleness in quarters; -1 = no filing
        h_fund      (B_f, D_enc): encoder output for ALL funds in snapshot
        h_stock     (B_s, D_enc): encoder output for ALL stocks in snapshot
        edge_fund_idx   (E,) long: per-edge fund indices
        edge_stock_idx  (E,) long: per-edge candidate stock indices
        context_indices (B_f, C_max) long: padded per-fund context stock ids
        context_mask    (B_f, C_max) bool: True where valid, False where padding
        (context built once per snapshot by ContextBuilder)

    Output:
        z (E, D_out): candidate-conditional context aggregate
        g_f_eff (E, D_text_proj): per-edge text+staleness representation
                                  (cached for downstream head concat)
    """

    INPUT_TEXT_DIM = 1024
    TEXT_PROJ_DIM  = 128       # mirrors ProspectusTextFusion.ABS_PROJ_DIM
    STALENESS_EMB_DIM = 16
    HEAD_DIM = 64              # internal attention dim
    OUTPUT_DIM = 128

    def __init__(
        self,
        encoder_dim: int,
        lambda_mu: float = 1.0,
        safety_cap: int = 512,           # DO-FIRST #1 (option A): non-binding for ~62%
        holdings_cap: int = 512,         # per-fund holdings cap in build_context_per_fund
        siblings_cap: int = 128,         # per-fund siblings cap (top-N by family freq)
        staleness_scale: float = 24.0,   # carry-H5 default
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.lambda_mu = float(lambda_mu)
        self.safety_cap = int(safety_cap)
        self.holdings_cap = int(holdings_cap)
        self.siblings_cap = int(siblings_cap)
        self.staleness_scale = float(staleness_scale)

        # Learned null token for delta_t == -1 cells (1024-d, broadcasts over fund batch).
        # User decision: learned null instead of hard-zero. Note this means M1's null cells
        # are NOT identical to no-text by construction — stratified test A reads token effect.
        self.null_token = nn.Parameter(torch.zeros(self.INPUT_TEXT_DIM))
        nn.init.normal_(self.null_token, mean=0.0, std=0.02)

        # Staleness embedding phi(delta_t): small MLP from a scalar (delta_t / staleness_scale)
        # to an embedding vector, concatenated with abs_proj to form g_f_eff.
        # For delta_t==-1, the input scalar is 0.0 (the null path) so phi outputs a fixed value.
        self.staleness_mlp = nn.Sequential(
            nn.Linear(1, self.STALENESS_EMB_DIM),
            nn.GELU(),
            nn.Linear(self.STALENESS_EMB_DIM, self.STALENESS_EMB_DIM),
        )

        # Project raw 1024 -> 128.
        self.abs_proj = nn.Sequential(
            nn.Linear(self.INPUT_TEXT_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.TEXT_PROJ_DIM),
            nn.LayerNorm(self.TEXT_PROJ_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # g_f_eff = [abs_proj || phi(delta_t)] -> TEXT_PROJ_DIM + STALENESS_EMB_DIM
        self.g_feff_dim = self.TEXT_PROJ_DIM + self.STALENESS_EMB_DIM

        # mu: eligibility prior. Project g_f_eff and h_c into HEAD_DIM, dot/sqrt(d).
        self.W_Q_mu = nn.Linear(self.g_feff_dim, self.HEAD_DIM)
        self.W_K_mu = nn.Linear(self.encoder_dim, self.HEAD_DIM)

        # alpha: candidate-conditional attention. Q from h_f, K/V from h_c.
        self.W_Q_alf = nn.Linear(self.encoder_dim, self.HEAD_DIM)
        self.W_K_alf = nn.Linear(self.encoder_dim, self.HEAD_DIM)
        self.W_V_alf = nn.Linear(self.encoder_dim, self.OUTPUT_DIM)

        self._last_alpha_entropy: float = 0.0   # mean H(alpha) for debugging
        self._last_mu_mean: float = 0.0
        self._last_null_frac: float = 0.0
        self._last_sibling_dup_count: int = 0   # # of context slots masked due to s also being a sibling

    # ------------------------------------------------------------------
    # g_f_eff: per-fund text + staleness representation
    # ------------------------------------------------------------------
    def compute_g_f_eff(
        self,
        text_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return (N_funds, g_feff_dim) per-fund text+staleness embedding.

        For delta_t == -1 (no-filing), abs_emb is replaced by the learned null token
        and the staleness scalar is set to 0.0 (so phi outputs a fixed value).
        """
        abs_emb = text_features['abs_emb']       # (N, 1024)
        delta_t = text_features['delta_t']       # (N,) long
        device = abs_emb.device

        null_mask = (delta_t == -1)              # (N,) bool
        # Replace null rows with the learned token (broadcast across batch)
        if null_mask.any():
            null_token = self.null_token.to(device).unsqueeze(0)  # (1, 1024)
            abs_emb = torch.where(null_mask.unsqueeze(-1), null_token.expand_as(abs_emb), abs_emb)
            self._last_null_frac = float(null_mask.float().mean().item())
        else:
            self._last_null_frac = 0.0

        # Staleness scalar input to phi: delta_t / staleness_scale, with -1 mapped to 0
        dt_scalar = torch.where(null_mask, torch.zeros_like(delta_t, dtype=torch.float32),
                                delta_t.float() / self.staleness_scale).unsqueeze(-1)  # (N, 1)
        phi = self.staleness_mlp(dt_scalar)       # (N, STALENESS_EMB_DIM)

        abs_p = self.abs_proj(abs_emb)            # (N, TEXT_PROJ_DIM)
        g_f_eff = torch.cat([abs_p, phi], dim=-1) # (N, g_feff_dim)
        return g_f_eff

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        g_f_eff_all: torch.Tensor,        # (N_funds, g_feff_dim)
        h_fund_all: torch.Tensor,         # (N_funds, encoder_dim)
        h_stock_all: torch.Tensor,        # (N_stocks, encoder_dim)
        edge_fund_idx: torch.Tensor,      # (E,)
        edge_stock_idx: torch.Tensor,     # (E,)
        context_indices: torch.Tensor,    # (N_funds, C_max)
        context_mask: torch.Tensor,       # (N_funds, C_max) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute z(f, s) and return per-edge g_f_eff for the head."""
        device = h_fund_all.device
        E = edge_fund_idx.shape[0]

        # Per-edge slices
        h_f_edge   = h_fund_all[edge_fund_idx]      # (E, D_enc)
        h_s_edge   = h_stock_all[edge_stock_idx]    # (E, D_enc)
        g_feff_edge = g_f_eff_all[edge_fund_idx]    # (E, g_feff_dim)

        # Per-edge context = fund's precomputed context ∪ {s}.
        # Pull (C_max,) per-edge context from the per-fund table.
        ctx_idx_edge  = context_indices[edge_fund_idx]   # (E, C_max)
        ctx_mask_edge = context_mask[edge_fund_idx]      # (E, C_max) bool

        # COLD-START INVARIANT: in C(f, s), the candidate s must appear EXACTLY
        # ONCE — as the appended candidate column. The holdings channel cannot
        # contain s by the upstream cold-start filter on edge_label_index. The
        # siblings channel CAN legitimately contain s (a family-mate fund may
        # hold s). Mask out any context entry equal to the candidate before the
        # append so s never appears twice in the attention slots.
        dup_mask = (ctx_idx_edge == edge_stock_idx.unsqueeze(-1))   # (E, C_max) bool
        ctx_mask_edge = ctx_mask_edge & ~dup_mask
        self._last_sibling_dup_count = int(dup_mask.sum().item())  # for diagnostics

        # Append s as one extra always-valid column.
        ctx_idx_edge  = torch.cat([ctx_idx_edge, edge_stock_idx.unsqueeze(-1)], dim=-1)   # (E, C_max+1)
        ctx_mask_edge = torch.cat([ctx_mask_edge,
                                   torch.ones(E, 1, dtype=torch.bool, device=device)], dim=-1)

        # Gather h_c for the per-edge context.
        # Clamp out-of-range pad indices (we mask them anyway) to keep gather safe.
        N_stocks = h_stock_all.shape[0]
        ctx_idx_safe = ctx_idx_edge.clamp(min=0, max=N_stocks - 1)
        h_c = h_stock_all[ctx_idx_safe]            # (E, C_max+1, D_enc)

        # mu(f, c)  — eligibility prior, candidate-conditional via g_f_eff(f).
        Q_mu = self.W_Q_mu(g_feff_edge).unsqueeze(1)               # (E, 1, head)
        K_mu = self.W_K_mu(h_c)                                    # (E, C, head)
        d = float(self.HEAD_DIM)
        mu = (Q_mu * K_mu).sum(-1) / (d ** 0.5)                    # (E, C)

        # Safety cap: keep top-cap by mu (within the valid mask). If C <= cap, no-op.
        C = ctx_mask_edge.shape[-1]
        if C > self.safety_cap:
            # For invalid entries, set mu to -inf so they are never selected.
            mu_masked = mu.masked_fill(~ctx_mask_edge, float('-inf'))
            top_vals, top_idx = mu_masked.topk(self.safety_cap, dim=-1)      # (E, cap)
            mu = mu.gather(1, top_idx)
            h_c = h_c.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, h_c.shape[-1]))
            ctx_mask_edge = ctx_mask_edge.gather(1, top_idx)

        # Attention logits.
        Q_alf = self.W_Q_alf(h_f_edge).unsqueeze(1)                # (E, 1, head)
        K_alf = self.W_K_alf(h_c)                                  # (E, C', head)
        attn_score = (Q_alf * K_alf).sum(-1) / (d ** 0.5)          # (E, C')
        logits = attn_score + self.lambda_mu * mu                  # (E, C')

        # Mask out padding (set to -inf so softmax assigns ~0 weight).
        logits = logits.masked_fill(~ctx_mask_edge, float('-inf'))

        # Edge cases: all-False mask row (no holdings, no siblings, candidate
        # somehow filtered) — softmax over all -inf is NaN. Force at least the
        # candidate position (last column originally, may have been topk'd away).
        # Detect rows with no valid entries:
        no_valid = (~ctx_mask_edge).all(dim=-1)
        if no_valid.any():
            # Re-include the candidate column (we always passed s in last); but
            # after topk it may not be in position -1 anymore. Safest fallback:
            # zero out z for those rows.
            pass  # handled below by masked_fill on alpha

        # Softmax → alpha.
        alpha = torch.softmax(logits, dim=-1)                      # (E, C')
        # Replace NaN (from all-(-inf)) with zero.
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # Weighted sum of values.
        V = self.W_V_alf(h_c)                                      # (E, C', OUTPUT_DIM)
        z = (alpha.unsqueeze(-1) * V).sum(dim=1)                   # (E, OUTPUT_DIM)

        # Monitoring (detached scalars).
        with torch.no_grad():
            p = alpha.clamp_min(1e-12)
            H = -(p * p.log()).sum(dim=-1)
            self._last_alpha_entropy = float(H.mean().item())
            valid = ctx_mask_edge.float()
            mu_valid = (mu * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
            self._last_mu_mean = float(mu_valid.mean().item())

        return z, g_feff_edge


# ----------------------------------------------------------------------
# Context builder
# ----------------------------------------------------------------------
def build_context_per_fund(
    snapshot_graph,
    holdings_cap: int = 512,    # DO-FIRST #1 (option A): covers p98 of |holdings|
    siblings_cap: int = 128,    # DO-FIRST #1 (option A): top-128 by family-coholding
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-fund context = holdings ∪ siblings_via_mgmt_company.

    Returns:
        context_indices: (N_funds, C_max) long — padded stock ids
        context_mask:    (N_funds, C_max) bool — True where valid

    Holdings are taken from the ('fund','holds_stock','stock') edge_index.
    Siblings are stocks held by other funds in the same management company,
    accumulated via ('fund','by_company','mgmt_company') + reverse + holds_stock.

    Caps:
      - holdings: keep ALL when |h| <= holdings_cap, else keep all (we don't sort
        without an objective here — the attention's mu-based top-K cap will trim
        downstream). This per-fund cap is only a hard memory bound.
      - siblings: when sibling-stock count exceeds siblings_cap, keep the most
        frequent (most often held across family funds).
    """
    # TEMPORAL CONTRACT: this function takes ONE snapshot graph (time T). The
    # returned per-fund context = holdings(f, T) ∪ siblings_via_mgmt_co(f, T).
    # No union across past quarters — that's the caller's responsibility (and it
    # currently passes support_graph[-1] = time T only).
    fwd_holds = ('fund', 'holds_stock', 'stock')
    fund_to_mc = ('fund', 'by_company', 'mgmt_company')
    mc_to_fund = ('mgmt_company', 'rev_by_company', 'fund')

    device = 'cpu'  # context is a CPU tensor; moved to GPU per use
    N_funds = snapshot_graph['fund'].x.shape[0] if 'fund' in snapshot_graph.node_types else 0

    if N_funds == 0 or fwd_holds not in snapshot_graph.edge_types:
        empty = torch.zeros((N_funds, 1), dtype=torch.long, device=device)
        emask = torch.zeros((N_funds, 1), dtype=torch.bool, device=device)
        return empty, emask

    ei = snapshot_graph[fwd_holds].edge_index.cpu()  # (2, E)

    # holdings[f] = set of stocks fund f holds
    holdings_per_fund = [[] for _ in range(N_funds)]
    for k in range(ei.shape[1]):
        f = int(ei[0, k].item())
        s = int(ei[1, k].item())
        if 0 <= f < N_funds:
            holdings_per_fund[f].append(s)

    # Siblings via mgmt_company hub. Only the FORWARD edge is required — both
    # the (fund -> mgmt_company) and (mgmt_company -> sibling funds) maps are
    # derived from it by inverting. (The casmln_neg loader synthesizes the
    # reverse edge at runtime, but we don't depend on its presence here.)
    siblings_per_fund: list = [[] for _ in range(N_funds)]
    has_company = (fund_to_mc in snapshot_graph.edge_types
                   and 'mgmt_company' in snapshot_graph.node_types)
    if has_company:
        f2mc = snapshot_graph[fund_to_mc].edge_index.cpu()       # (2, F)

        # fund -> mgmt_company id (assume each fund has one company; take first)
        # mgmt_company -> [sibling funds] (derived from the same forward edge)
        fund_company: dict = {}
        company_funds: dict = {}
        for k in range(f2mc.shape[1]):
            f_i = int(f2mc[0, k].item())
            m_i = int(f2mc[1, k].item())
            fund_company.setdefault(f_i, m_i)
            company_funds.setdefault(m_i, []).append(f_i)

        for f in range(N_funds):
            m = fund_company.get(f)
            if m is None:
                continue
            siblings_funds = [other_f for other_f in company_funds.get(m, []) if other_f != f]
            # Tally sibling-stock frequencies across family
            tally: dict = {}
            for sf in siblings_funds:
                for s in holdings_per_fund[sf]:
                    tally[s] = tally.get(s, 0) + 1
            if not tally:
                continue
            ranked = sorted(tally.items(), key=lambda kv: -kv[1])
            sibling_stocks = [s for s, _ in ranked[:siblings_cap]]
            # Exclude any stock already in holdings(f) to avoid double-counting
            own = set(holdings_per_fund[f])
            siblings_per_fund[f] = [s for s in sibling_stocks if s not in own]

    # Merge holdings + siblings per fund, cap to (holdings_cap + siblings_cap)
    merged_per_fund: list = []
    max_len = 1
    for f in range(N_funds):
        h = holdings_per_fund[f][:holdings_cap]
        s_sib = siblings_per_fund[f][:siblings_cap]
        merged = h + s_sib
        if not merged:
            merged = [0]  # placeholder; mask will be all-False so it won't be used
        merged_per_fund.append(merged)
        if len(merged) > max_len:
            max_len = len(merged)

    # Build padded tensor + mask
    ctx_idx = torch.zeros((N_funds, max_len), dtype=torch.long)
    ctx_mask = torch.zeros((N_funds, max_len), dtype=torch.bool)
    for f, lst in enumerate(merged_per_fund):
        L = len(lst)
        if L == 0:
            continue
        ctx_idx[f, :L] = torch.tensor(lst, dtype=torch.long)
        # If the only entry was the [0] placeholder for an empty fund, mark False.
        is_real = (len(holdings_per_fund[f]) + len(siblings_per_fund[f])) > 0
        if is_real:
            ctx_mask[f, :L] = True

    return ctx_idx, ctx_mask
