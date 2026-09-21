"""Prospectus text fusion modules for fund node features (OpenAI 1024-d).

Simplified version per 2026-04-07 spec:
- ProspectusTextFusion: 1024-d abs_emb + delta_t + 16-d numerical → 128-d
- PhaseScheduler: three-phase training orchestrator (Phase C lr=5e-4)
- ContrastiveLoss: optional InfoNCE on abs_proj (off by default; opt-in via --use_contrastive)
- FundStockContrastiveLoss: optional InfoNCE aligning fund text to held stocks (Approach 4)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ProspectusTextFusion
# ---------------------------------------------------------------------------
class ProspectusTextFusion(nn.Module):  # PROSPECTUS_INTEGRATION
    """Gated fusion of OpenAI 1024-d prospectus embeddings with numerical fund features.

    Forward inputs:
        text_features: dict with keys
            - abs_emb: (B, 1024) weighted strategy/risk combo from the loader
            - delta_t: (B,) int — quarters since last real text; -1 = cold-start
        numerical: (B, numerical_dim) fund numerical features.

    Returns:
        h_fund:   (B, 128) fused fund representation, replaces graph['fund'].x.
        abs_proj: (B, 128) static text projection (cached for text_stock_prior /
                  contrastive auxiliary heads).
    """

    INPUT_DIM = 1024
    ABS_PROJ_DIM = 128
    OUTPUT_DIM = 128

    def __init__(
        self,
        numerical_dim: int = 16,
        dropout: float = 0.2,
        no_staleness: bool = False,
        staleness_scale: float = 6.0,
        # Fund-aware gate (additive, off by default):
        # When True, gate_linear takes a learned per-fund "text reliability prior"
        # as additional input. Embedding is zero-initialized so behavior at start
        # is identical to the standard gate.
        use_fund_aware_gate: bool = False,
        n_funds: int = 25000,
        fund_prior_dim: int = 8,
        # Jacobian-ratio gate input (additive, off by default):
        # When True, gate_linear takes log(|ΔE'|/|ΔE|) per fund per quarter as an
        # extra scalar input. High value = MLP amplifying noise → gate can learn
        # to suppress text. Bidirectional: negative = MLP filtering, neutral.
        use_jacobian_ratio_gate: bool = False,
        # When True, fund node feature ignores text: h_fund := num_proj(numerical).
        # abs_proj_out is still returned so TBA/SpAB/SpatialAlign/ToA aux losses
        # keep working. Default off — existing behavior preserved.
        no_text_fund_feature: bool = False,
        # Fusion mode: "convex" (default, existing behavior) or "residual"
        # (zero-init additive add-on). See forward() for the exact formulas.
        fusion_mode: str = "convex",
    ):
        super().__init__()
        self.numerical_dim = numerical_dim
        self.no_staleness = no_staleness
        self.staleness_scale = float(staleness_scale)
        self.use_fund_aware_gate = use_fund_aware_gate
        self.fund_prior_dim = fund_prior_dim if use_fund_aware_gate else 0
        self.use_jacobian_ratio_gate = use_jacobian_ratio_gate
        self.jacobian_ratio_dim = 1 if use_jacobian_ratio_gate else 0
        self.no_text_fund_feature = bool(no_text_fund_feature)
        if fusion_mode not in ("convex", "residual", "film"):
            raise ValueError(f"fusion_mode must be 'convex', 'residual', or 'film', got {fusion_mode!r}")
        self.fusion_mode = fusion_mode

        # 1024 → 256 → 128
        self.abs_proj = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.ABS_PROJ_DIM),
            nn.LayerNorm(self.ABS_PROJ_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learned staleness scalars: sigmoid(w * delta_t/staleness_scale + b); init w=-2, b=1.
        # staleness_scale defaults to 6.0 (calibrated for max_lag=6 bounded H5).
        # For unbounded carry-forward H5 (max_lag=65), set higher (e.g. 24)
        # so the sigmoid spans the actual in-distribution Δ range.
        self.staleness_w = nn.Parameter(torch.tensor(-2.0))
        self.staleness_b = nn.Parameter(torch.tensor(1.0))

        # Gating fusion. When fund-aware gate is on, gate_linear has expanded
        # input dim including the per-fund prior. Zero-init Embedding means the
        # prior contributes 0 at construction — gate output is identical to the
        # standard variant on the first forward, then deviates as it learns.
        self.text_norm = nn.LayerNorm(self.ABS_PROJ_DIM)
        gate_in_dim = (numerical_dim + self.ABS_PROJ_DIM
                       + self.fund_prior_dim + self.jacobian_ratio_dim)
        self.gate_linear = nn.Linear(gate_in_dim, self.ABS_PROJ_DIM)
        if fusion_mode == "residual":
            # Zero-init gate for additive-residual mode: sigmoid(-5) ≈ 0.0067,
            # so at init text branch contributes ~0 and h_fund ≈ num_proj(numerical).
            # Gate opens only where it helps as training proceeds.
            nn.init.constant_(self.gate_linear.bias, -5.0)
            print(f"[PROSPECTUS] fusion_mode=residual: gate zero-init "
                  f"(bias=-5, sigmoid≈0.007). h_fund = num_proj(numerical) + "
                  f"gate * text_norm(text_input); text is a strictly additive add-on.")
        else:
            nn.init.constant_(self.gate_linear.bias, 1.0)
        if use_fund_aware_gate:
            self.fund_prior = nn.Embedding(n_funds, fund_prior_dim)
            nn.init.zeros_(self.fund_prior.weight)
            print(f"[PROSPECTUS] Fund-aware gate enabled "
                  f"(n_funds={n_funds}, fund_prior_dim={fund_prior_dim}, zero-init)")
        if use_jacobian_ratio_gate:
            print(f"[PROSPECTUS] Jacobian-ratio gate input enabled "
                  f"(log_ratio scalar added to gate_linear input, +1 dim)")
        self.num_proj = nn.Linear(numerical_dim, self.OUTPUT_DIM)

        if fusion_mode == "film":
            self.film_gamma = nn.Linear(self.ABS_PROJ_DIM, self.OUTPUT_DIM)
            self.film_beta = nn.Linear(self.ABS_PROJ_DIM, self.OUTPUT_DIM)
            nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight); nn.init.zeros_(self.film_beta.bias)
            print(f"[PROSPECTUS] fusion_mode=film: h_fund = num_proj(num) * "
                  f"(1 + γ(text_norm(text_input))) + β(...); γ/β zero-init so "
                  f"h_fund ≈ num_proj(num) at start.")

        # Monitoring for Jacobian-ratio gate
        self._last_log_ratio_mean: float = 0.0

        # Monitoring
        self._last_gate_mean: float = 0.0
        self._last_staleness_mean: float = 0.0

    def _staleness_weight(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Per-fund staleness scalar.

        delta_t == -1 → 0.0 (cold-start, hard zero — collapses gate to numerical-only).
        Otherwise → sigmoid(w * delta_t/self.staleness_scale + b), or 1.0 if no_staleness=True.
        """
        dt = delta_t.float()
        if self.no_staleness:
            w = torch.ones_like(dt)
        else:
            w = torch.sigmoid(self.staleness_w * (dt / self.staleness_scale) + self.staleness_b)
        # Hard zero for cold-start (regardless of no_staleness)
        return torch.where(delta_t == -1, torch.zeros_like(w), w)

    def forward(
        self,
        text_features: Dict[str, torch.Tensor],
        numerical: torch.Tensor,
        behavior_gate: torch.Tensor = None,
        log_ratio: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        behavior_gate: optional (B,) per-fund scalar in (0, 1) from
            TextBehaviorAlignment. When provided, multiplied into the text
            branch before the numerical/text mixing gate.
        log_ratio: optional (B,) per-fund scalar = log(|ΔE'|) − log(|ΔE|).
            When use_jacobian_ratio_gate=True, concatenated into gate_linear's
            input. High value ⇒ MLP locally amplifying small text changes
            ⇒ noise suspect ⇒ gate can learn to suppress. Default None preserves
            the historical forward pass.
        """
        abs_emb = text_features["abs_emb"]      # (B, 1024)
        delta_t = text_features["delta_t"]      # (B,)

        abs_proj_out = self.abs_proj(abs_emb)   # (B, 128)
        staleness = self._staleness_weight(delta_t).unsqueeze(-1)  # (B, 1)
        text_input = staleness * abs_proj_out   # (B, 128) — zeroed for cold-start

        if behavior_gate is not None:
            text_input = behavior_gate.unsqueeze(-1) * text_input

        # Build gate input. Order: [numerical, text_input, fund_prior?, log_ratio?]
        gate_parts = [numerical, text_input]
        if self.use_fund_aware_gate and 'fund_ids' in text_features:
            prior = self.fund_prior(text_features['fund_ids'])  # (B, fund_prior_dim)
            gate_parts.append(prior)
        if self.use_jacobian_ratio_gate:
            if log_ratio is not None:
                # Rescale to roughly [-1, 1] range for clean sigmoid input
                lr = (log_ratio * 0.5).unsqueeze(-1)             # (B, 1)
            else:
                # Default: zero (neutral) when log_ratio not provided
                lr = torch.zeros(numerical.shape[0], 1, device=numerical.device,
                                 dtype=numerical.dtype)
            gate_parts.append(lr)
            with torch.no_grad():
                if log_ratio is not None:
                    self._last_log_ratio_mean = float(log_ratio.mean().item())

        gate_in = torch.cat(gate_parts, dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_in))             # (B, 128)
        if self.fusion_mode == "film":
            # FiLM at input: numerical is content, text is condition.
            # text_input already has staleness applied → cold-start (text_input=0)
            # → text_normed=0 → γ=β=0 → h_fund = num_proj(numerical). Same
            # cold-start semantics as convex/residual modes.
            # gate_linear output above is unused in film mode (harmless).
            text_normed = self.text_norm(text_input)
            γ = self.film_gamma(text_normed)                        # (B, 128)
            β = self.film_beta(text_normed)                         # (B, 128)
            h_fund = self.num_proj(numerical) * (1.0 + γ) + β
        elif self.fusion_mode == "residual":
            # Additive residual: text is a strictly additive add-on. At init
            # gate ≈ 0 (bias=-5) so h_fund ≈ num_proj(numerical) — the numerical
            # pathway is never diluted. For zero-fund-feature datasets (Canadian
            # brokenfeats), num_proj(0)=num_proj.bias is a shared constant that
            # HGT's input projection handles just as it does in the text-free
            # baseline. Cold-start funds (text_input=0) never lose signal.
            h_fund = self.num_proj(numerical) + gate * self.text_norm(text_input)
        else:
            h_fund = gate * self.text_norm(text_input) + (1.0 - gate) * self.num_proj(numerical)

        self._last_gate_mean = float(gate.detach().mean().item())
        self._last_staleness_mean = float(staleness.detach().mean().item())
        return h_fund, abs_proj_out


# ---------------------------------------------------------------------------
# PhaseScheduler (Phase C lr=5e-4 per 2026-04-07 spec)
# ---------------------------------------------------------------------------
@dataclass
class PhaseConfig:  # PROSPECTUS_INTEGRATION
    phase_name: str
    freeze_backbone: bool
    lr_backbone: float
    lr_text_fusion: float
    weight_decay: float
    modality_dropout_p: float


class PhaseScheduler:  # PROSPECTUS_INTEGRATION
    """Three-phase training orchestrator.

    Phase A (epochs 0..A-1):  Freeze backbone, train fusion only, lr=1e-3.
    Phase B (epochs A..A+B-1): Unfreeze, modality dropout 0.80→0.20 linear,
                               backbone lr=5e-4, fusion lr=1e-3.
    Phase C (A+B..):           All params, dropout 0.20, lr=5e-4 (both groups).
    """

    def __init__(
        self,
        phase_a_epochs: int = 10,
        phase_b_epochs: int = 30,
        no_modality_dropout: bool = False,
    ):
        self.phase_a_end = phase_a_epochs
        self.phase_b_end = phase_a_epochs + phase_b_epochs
        self.no_modality_dropout = no_modality_dropout

    def get_phase(self, epoch: int) -> PhaseConfig:
        if epoch < self.phase_a_end:
            return PhaseConfig(
                phase_name="A (freeze backbone)",
                freeze_backbone=True,
                lr_backbone=0.0,
                lr_text_fusion=1e-3,
                weight_decay=0.0,
                modality_dropout_p=0.0,
            )
        if epoch < self.phase_b_end:
            denom = max(1, self.phase_b_end - self.phase_a_end - 1)
            progress = (epoch - self.phase_a_end) / denom
            dropout_p = 0.0 if self.no_modality_dropout else (0.80 - 0.60 * progress)
            return PhaseConfig(
                phase_name="B (gradual unfreeze)",
                freeze_backbone=False,
                lr_backbone=5e-4,
                lr_text_fusion=1e-3,
                weight_decay=0.0,
                modality_dropout_p=dropout_p,
            )
        return PhaseConfig(
            phase_name="C (full training)",
            freeze_backbone=False,
            lr_backbone=5e-4,
            lr_text_fusion=5e-4,
            weight_decay=0.0,
            modality_dropout_p=0.0 if self.no_modality_dropout else 0.20,
        )

    def apply_phase(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> PhaseConfig:
        """Freeze/unfreeze backbone params and update LR groups in place.

        Assumes optimizer has two param groups:
            group 0: backbone params
            group 1: prospectus_fusion params
        """
        cfg = self.get_phase(epoch)
        for name, p in model.named_parameters():
            if "prospectus_fusion" in name:
                p.requires_grad = True
            else:
                p.requires_grad = not cfg.freeze_backbone
        if len(optimizer.param_groups) >= 2:
            optimizer.param_groups[0]["lr"] = cfg.lr_backbone
            optimizer.param_groups[0]["weight_decay"] = cfg.weight_decay
            optimizer.param_groups[1]["lr"] = cfg.lr_text_fusion
            optimizer.param_groups[1]["weight_decay"] = cfg.weight_decay
        return cfg


# ---------------------------------------------------------------------------
# ContrastiveLoss (optional, opt-in via --use_contrastive)
# ---------------------------------------------------------------------------
class ContrastiveLoss(nn.Module):  # PROSPECTUS_INTEGRATION
    """InfoNCE contrastive loss on abs_proj embeddings.

    Off by default. Opt-in via ``--use_contrastive`` if you want to add a
    self-supervised regulariser on the text projection.

    Positive pairs: funds whose novel-acquisition Jaccard overlap >0.5.
    Negatives: all other funds in the same batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        abs_proj: torch.Tensor,
        novel_stock_sets: List[Set[int]],
    ) -> torch.Tensor:
        B = abs_proj.shape[0]
        if B < 2:
            return abs_proj.new_tensor(0.0)

        pos_mask = torch.zeros(B, B, dtype=torch.bool, device=abs_proj.device)
        for i in range(B):
            if not novel_stock_sets[i]:
                continue
            for j in range(i + 1, B):
                if not novel_stock_sets[j]:
                    continue
                overlap = len(novel_stock_sets[i] & novel_stock_sets[j])
                min_size = min(len(novel_stock_sets[i]), len(novel_stock_sets[j]))
                if min_size > 0 and overlap / min_size > 0.5:
                    pos_mask[i, j] = True
                    pos_mask[j, i] = True

        if not pos_mask.any():
            return abs_proj.new_tensor(0.0)

        emb = F.normalize(abs_proj, dim=-1)
        sim = emb @ emb.t() / self.temperature
        sim = sim.masked_fill(torch.eye(B, dtype=torch.bool, device=abs_proj.device), -1e9)

        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return abs_proj.new_tensor(0.0)

        log_softmax = F.log_softmax(sim, dim=1)
        pos_log_probs = (log_softmax * pos_mask.float()).sum(dim=1)
        pos_counts = pos_mask.float().sum(dim=1).clamp(min=1)
        return -(pos_log_probs[has_pos] / pos_counts[has_pos]).mean()


# ---------------------------------------------------------------------------
# FundStockContrastiveLoss (Approach 4)
# ---------------------------------------------------------------------------
class FundStockContrastiveLoss(nn.Module):  # PROSPECTUS_INTEGRATION
    """InfoNCE loss aligning fund abs_proj to raw features of stocks the fund holds.

    For each (fund, stock) positive edge in the query batch, treat
    abs_proj[fund] as the anchor and the stock's raw feature row (projected
    via a learned LazyLinear) as the positive. All other stocks in the batch
    act as in-batch negatives via cross-entropy on similarity logits.

    Inputs to forward():
        abs_proj:        (N_fund, 128) — pre-GNN text projections for all funds in the snapshot
        stock_raw:       (N_stock, stock_raw_dim) — raw stock features (pre-GNN)
        pos_edge_index:  (2, E_pos) — fund→stock indices of positive edges in the query

    Returns: scalar loss, or 0 if there are no positives.
    """

    def __init__(self, abs_proj_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.abs_proj_dim = abs_proj_dim
        self.temperature = temperature
        # LazyLinear infers stock_raw_dim on first forward
        self.stock_proj = nn.Sequential(
            nn.LazyLinear(abs_proj_dim),
            nn.LayerNorm(abs_proj_dim),
            nn.GELU(),
        )

    def forward(
        self,
        abs_proj: torch.Tensor,        # (N_fund, 128)
        stock_raw: torch.Tensor,       # (N_stock, raw_dim)
        pos_edge_index: torch.Tensor,  # (2, E_pos)
    ) -> torch.Tensor:
        if pos_edge_index is None or pos_edge_index.numel() == 0 or stock_raw is None:
            return abs_proj.new_tensor(0.0)

        # Project all stocks once and L2-normalize for cosine-similarity logits
        stock_emb = self.stock_proj(stock_raw)                       # (N_stock, 128)
        abs_norm   = F.normalize(abs_proj, dim=-1)                   # (N_fund, 128)
        stock_norm = F.normalize(stock_emb, dim=-1)                  # (N_stock, 128)

        # Each anchor is the fund of a positive edge; positive class is the held stock
        fund_idx  = pos_edge_index[0]                                # (E_pos,)
        stock_idx = pos_edge_index[1]                                # (E_pos,)
        f = abs_norm[fund_idx]                                        # (E_pos, 128)

        logits = (f @ stock_norm.t()) / self.temperature              # (E_pos, N_stock)
        loss = F.cross_entropy(logits, stock_idx)
        return loss
