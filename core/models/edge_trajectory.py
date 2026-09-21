"""Edge weight trajectory encoder (GRU over per-persistent-edge weight history).

Used by HGT+ message-passing FiLM to inject [w(t), Δw(t), present_t] dynamics
into the value path. See docs/plans/2026-06-12-edge-trajectory-film.md.
"""
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn


class EdgeTrajectoryEncoder(nn.Module):
    """GRU over per-persistent-edge weight series.

    Registry is built from union of support-window edges and (optional) query
    edges at scoring time. Output is the full sequence of GRU hidden states
    H[T, E_reg, traj_dim] — causal: h(t) only depends on w(1..t).
    """

    def __init__(self, traj_dim: int = 32, log1p_input: bool = False,
                 no_present: bool = False, static: bool = False,
                 no_delta_w: bool = False,
                 use_text_init: bool = False,
                 text_init_dim: int = 128,
                 use_text_step_input: bool = False,
                 text_step_input_dim: int = 128):
        super().__init__()
        self.traj_dim = traj_dim
        # When True, ew is mapped through log1p before computing w_t / Δw_t.
        # prev_w then tracks log-space values, so Δw_t = log1p(w_t) - log1p(w_{t-1}).
        self.log1p_input = bool(log1p_input)
        # When True, drop the present_t channel: GRU sees [w(t), Δw(t)] only.
        self.no_present = bool(no_present)
        # When True (with static), drop the Δw(t) channel: static encoder emits
        # 1-channel [w(t)] only. No-op when static=False (GRU still gets dw_t).
        self.no_delta_w = bool(no_delta_w)
        # When True, the encoder has ZERO learnable parameters: h(t) is just
        # the raw 2-channel [w(t), Δw(t)] for each registry edge. FiLM heads
        # then consume this 2-d input directly (constructor must be called with
        # traj_dim=2). prev_w buffer is still maintained to compute Δw(t) as a
        # 1-step lookback, but no parameterized layer touches the trajectory.
        # This is the strongest "no recurrence" claim: if the GRU still wins,
        # it's specifically the multi-step aggregation that matters.
        # --edge_trajectory_no_present is ignored for the static path (always
        # 2-channel). --edge_trajectory_log1p still affects what w_t looks like.
        # Text-step-input flag (Option 2b) — declared before static/non-static
        # branch so both paths expose the attribute.
        self.use_text_step_input = bool(use_text_step_input)
        self.text_step_input_dim = int(text_step_input_dim) if self.use_text_step_input else 0
        if self.use_text_step_input and bool(static):
            raise ValueError("use_text_step_input is incompatible with static encoder "
                             "(no GRU to receive per-step input).")
        self.static = bool(static)
        if self.static:
            expected_dim = 1 if self.no_delta_w else 2
            assert traj_dim == expected_dim, (
                f"Static encoder is parameterless and emits raw "
                f"{'[w]' if self.no_delta_w else '[w, Δw]'}; "
                f"traj_dim must be {expected_dim} (got {traj_dim}). The trainer should "
                f"override dim to {expected_dim} when configuring static (no_delta_w={self.no_delta_w})."
            )
            self.gru_cell = None
            self.feature_mlp = None
        else:
            # Text-per-step input (Option 2b): concatenate text_per_reg_t to the
            # scalar (w, dw, present) inputs at every GRU step. Bloats GRU input
            # size by text_step_input_dim (typ. 128). Applied inside encode()'s
            # step loop below when self.use_text_step_input is True.
            base_input_size = 2 if self.no_present else 3
            input_size = base_input_size + self.text_step_input_dim
            self.feature_mlp = None
            self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=traj_dim)
            if self.use_text_step_input:
                print(f"[TCETF text-step-input] GRU input = [{base_input_size}-scalars, "
                      f"{self.text_step_input_dim}-text] = {input_size}-d; text concatenated "
                      f"per step. GRU param count: 3 * ({input_size}*{traj_dim} + {traj_dim}²) "
                      f"= {3 * (input_size*traj_dim + traj_dim*traj_dim + traj_dim)}.")

        # Text-conditions-trajectory: when enabled, GRU's initial hidden state h_0
        # is a text-derived projection (per registry edge) instead of zeros.
        # Zero-init W_init → h_0 ≈ 0 at start = current behavior. Learns to
        # encode text as a "prior" over the fund's trajectory representation.
        # Requires a `text_per_registry` tensor to be supplied to encode().
        self.use_text_init = bool(use_text_init)
        self.text_init_dim = int(text_init_dim)
        if self.use_text_init:
            self.text_init_proj = nn.Linear(text_init_dim, traj_dim)
            nn.init.zeros_(self.text_init_proj.weight)
            nn.init.zeros_(self.text_init_proj.bias)
            print(f"[TCETF text-init] GRU h_0 = Linear({text_init_dim}->{traj_dim})(text_per_edge); "
                  f"zero-init so h_0 == 0 at start (matches current behavior).")

    def encode(
        self,
        snapshots: List[Dict[str, torch.Tensor]],
        query_edge_index: Optional[torch.Tensor] = None,
        text_per_fund: Optional[torch.Tensor] = None,
        text_per_snapshot: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, int]:
        """Build registry and run the GRU.

        Args:
            snapshots: list of T dicts, each with 'edge_index' (2, E_t) long and
                'edge_weight' (E_t,) float. Same shape as EdgeMemoryModule's
                process_support_window input.
            query_edge_index: optional (2, Q) long tensor of additional edges
                (e.g., window-8 positives ∪ negatives) to include in the
                registry. These edges contribute to the registry but never
                appear in any snapshot's edge_index — their GRU input is
                all-zeros at every t, producing a learned cold-start vector.

        Returns:
            H: (T, E_reg, traj_dim) float — H[t, e] is the GRU hidden state at
                snapshot t for registry edge e.
            inverse_per_snapshot: list of T long tensors (E_t,) — index into
                E_reg for each edge present in snapshot t. Use for gather:
                `H[t, inverse_per_snapshot[t]]` returns (E_t, traj_dim).
            unique_keys: (E_reg,) long — fund*MAX_STOCK + stock for each
                registry edge. For external lookup if needed.
            max_stock: int — the divisor used for key encoding.
        """
        # When the encoder is parameterless (--edge_trajectory_static), there
        # are no parameters to read a device from. Fall back to the first
        # snapshot's edge_index device, or CPU if even that is unavailable.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            if snapshots and 'edge_index' in snapshots[0]:
                device = snapshots[0]['edge_index'].device
            elif query_edge_index is not None:
                device = query_edge_index.device
            else:
                device = torch.device('cpu')

        # 1. Build registry from union of snapshot edges + query edges.
        all_fids: List[torch.Tensor] = []
        all_sids: List[torch.Tensor] = []
        for snap in snapshots:
            ei = snap["edge_index"].to(device).long()
            all_fids.append(ei[0])
            all_sids.append(ei[1])
        if query_edge_index is not None:
            qe = query_edge_index.to(device).long()
            all_fids.append(qe[0])
            all_sids.append(qe[1])

        cat_fids = torch.cat(all_fids) if all_fids else torch.zeros(0, dtype=torch.long, device=device)
        cat_sids = torch.cat(all_sids) if all_sids else torch.zeros(0, dtype=torch.long, device=device)
        if cat_fids.numel() == 0:
            empty_H = torch.zeros(len(snapshots), 0, self.traj_dim, device=device)
            empty_inv = [torch.zeros(0, dtype=torch.long, device=device) for _ in snapshots]
            empty_keys = torch.zeros(0, dtype=torch.long, device=device)
            return empty_H, empty_inv, empty_keys, 1

        max_stock = int(cat_sids.max().item()) + 1
        max_stock = max(max_stock, 1)
        cat_keys = cat_fids * max_stock + cat_sids
        unique_keys = torch.unique(cat_keys)
        n_edges = unique_keys.shape[0]
        sorted_keys = unique_keys  # torch.unique returns sorted output

        # 2. Per-snapshot gather indices via searchsorted on sorted_keys.
        inverse_per_snapshot: List[torch.Tensor] = []
        for snap in snapshots:
            ei = snap["edge_index"].to(device).long()
            keys_t = ei[0] * max_stock + ei[1]
            idx_t = torch.searchsorted(sorted_keys, keys_t)
            idx_t = idx_t.clamp(max=n_edges - 1)
            inverse_per_snapshot.append(idx_t)

        # 3. Run GRU per snapshot. Inputs default to zeros (cold-start / absent).
        H = torch.zeros(len(snapshots), n_edges, self.traj_dim, device=device)
        # Initial hidden state: zeros by default, or text-derived when
        # use_text_init and text_per_fund is provided. text_per_fund is
        # (N_funds, text_init_dim); we gather per-registry-edge text by mapping
        # each key back to its fund_id: fund_id = key // max_stock.
        if (self.use_text_init and text_per_fund is not None
                and getattr(self, 'text_init_proj', None) is not None):
            fund_ids = (sorted_keys // max_stock).long().to(text_per_fund.device)
            # Guard against fund_ids that exceed text_per_fund rows (shouldn't
            # happen with correctly-built text_per_fund but be defensive).
            fund_ids = fund_ids.clamp(max=text_per_fund.shape[0] - 1)
            text_per_reg = text_per_fund[fund_ids].to(device)
            h = self.text_init_proj(text_per_reg)                # (E_reg, traj_dim)
        else:
            h = torch.zeros(n_edges, self.traj_dim, device=device)
        prev_w = torch.zeros(n_edges, device=device)

        for t, snap in enumerate(snapshots):
            mem_idx = inverse_per_snapshot[t]
            ew = snap["edge_weight"].float().to(device)
            if self.log1p_input:
                # Apply log1p in-place (clamp guards against tiny negatives from
                # float roundoff). prev_w then tracks log-space values, so
                # dw_t = log1p(w(t)) - log1p(w(t-1)) is the multiplicative-change
                # signal in the log domain.
                ew = torch.log1p(ew.clamp(min=0.0))

            w_t = torch.zeros(n_edges, device=device)
            dw_t = torch.zeros(n_edges, device=device)
            present_t = torch.zeros(n_edges, device=device)

            if mem_idx.numel() > 0:
                # If a registry edge appears multiple times in one snapshot
                # (shouldn't normally happen but be defensive), index_copy_
                # keeps the last write — acceptable for this signal.
                w_t.index_copy_(0, mem_idx, ew)
                dw_t.index_copy_(0, mem_idx, ew - prev_w.index_select(0, mem_idx))
                present_t.index_copy_(0, mem_idx,
                                      torch.ones_like(ew))

            if self.static:
                # No learnable layer in the encoder: h(t) is the raw 2-channel
                # feature. FiLM heads consume it directly. The hidden h carried
                # from the previous iteration is ignored — no recurrence.
                # With no_delta_w, drop dw_t and emit 1-channel [w_t] instead.
                if self.no_delta_w:
                    H[t] = w_t.unsqueeze(-1)              # (n_edges, 1)
                else:
                    H[t] = torch.stack([w_t, dw_t], dim=-1)  # (n_edges, 2)
            else:
                if self.no_present:
                    step_input = torch.stack([w_t, dw_t], dim=-1)
                else:
                    step_input = torch.stack([w_t, dw_t, present_t], dim=-1)
                # Text-per-step input (Option 2b): concat text_per_reg_t to step
                # scalars. Uses text_per_snapshot[t] if available; falls back to
                # zeros for absent snapshots. text_per_reg_t is looked up per
                # registry edge via fund_id = sorted_keys[e] // max_stock.
                if self.use_text_step_input:
                    if (text_per_snapshot is not None and t < len(text_per_snapshot)
                            and text_per_snapshot[t] is not None):
                        text_t = text_per_snapshot[t]
                        fund_ids = (sorted_keys // max_stock).long().to(text_t.device)
                        fund_ids = fund_ids.clamp(max=text_t.shape[0] - 1)
                        text_per_reg_t = text_t[fund_ids].to(device)  # (E_reg, text_step_input_dim)
                    else:
                        text_per_reg_t = torch.zeros(n_edges, self.text_step_input_dim, device=device)
                    step_input = torch.cat([step_input, text_per_reg_t], dim=-1)
                h = self.gru_cell(step_input, h)
                H[t] = h
            if mem_idx.numel() > 0:
                prev_w.index_copy_(0, mem_idx, ew)

        return H, inverse_per_snapshot, sorted_keys, max_stock

    def lookup_query(
        self,
        H: torch.Tensor,
        sorted_keys: torch.Tensor,
        max_stock: int,
        query_edge_index: torch.Tensor,
        t: int = -1,
    ) -> torch.Tensor:
        """Gather h(t) for a set of query edges from the encoded H.

        Returns (Q, traj_dim). `t=-1` (default) means the last snapshot.

        IMPORTANT: every (fund, stock) pair in `query_edge_index` must have
        been present in the registry passed to `encode()` (i.e. it appeared
        in some snapshot or in encode()'s `query_edge_index`). Keys not in
        the registry silently snap to the nearest neighbor via `searchsorted +
        clamp`, returning a wrong embedding rather than an error. Callers
        are responsible for not querying out-of-registry edges.
        """
        if H.shape[1] == 0 or query_edge_index.numel() == 0:
            return torch.zeros(query_edge_index.shape[1] if query_edge_index.dim() == 2 else 0,
                               self.traj_dim, device=H.device)
        qe = query_edge_index.to(H.device).long()
        keys = qe[0] * max_stock + qe[1]
        idx = torch.searchsorted(sorted_keys, keys)
        idx = idx.clamp(max=H.shape[1] - 1)
        return H[t, idx]


class EdgeTrajectoryFiLM(nn.Module):
    """Per-edge-type FiLM heads: edge_attr (E, traj_dim) → (γ, β) of shape
    (E, heads, head_dim) each.

    Zero-initialized so γ ≡ 0 and β ≡ 0 at start ⇒ V_j unchanged ⇒ baseline-
    identical at the first forward.

    TCETF extension (use_tcetf=True): additive Δγ(text_f), Δβ(text_f) zero-init
    offsets per edge_type. Per-edge text is supplied by the trainer via the
    optional `text_per_edge` arg (the fund-endpoint's abs_proj). With Δ-networks
    zero-initialized, TCETF behaves identically to plain edge-trajectory FiLM
    at construction time and learns text-conditional offsets during training.
    """

    def __init__(self, edge_types, traj_dim: int, hidden_dim: int, heads: int,
                 use_tcetf: bool = False, tcetf_text_dim: int = 128,
                 tcetf_mode: str = "additive"):
        super().__init__()
        assert hidden_dim % heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        if tcetf_mode not in ("additive", "gated"):
            raise ValueError(f"tcetf_mode must be 'additive' or 'gated', got {tcetf_mode!r}")
        self.traj_dim = traj_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.gamma = nn.ModuleDict()
        self.beta = nn.ModuleDict()
        for etype in edge_types:
            key = "__".join(etype)
            g = nn.Linear(traj_dim, hidden_dim)
            b = nn.Linear(traj_dim, hidden_dim)
            nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)
            nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)
            self.gamma[key] = g
            self.beta[key] = b

        self.use_tcetf = bool(use_tcetf)
        self.tcetf_text_dim = int(tcetf_text_dim)
        self.tcetf_mode = str(tcetf_mode)
        if self.use_tcetf:
            self.gamma_text = nn.ModuleDict()
            self.beta_text = nn.ModuleDict()
            for etype in edge_types:
                key = "__".join(etype)
                g_t = nn.Linear(self.tcetf_text_dim, hidden_dim)
                b_t = nn.Linear(self.tcetf_text_dim, hidden_dim)
                nn.init.zeros_(g_t.weight); nn.init.zeros_(g_t.bias)
                nn.init.zeros_(b_t.weight); nn.init.zeros_(b_t.bias)
                self.gamma_text[key] = g_t
                self.beta_text[key] = b_t
            # Gated-mode multiplicative modulation of γ_base, β_base by text.
            # Gates are init to bias=5 so sigmoid(5)≈0.993, i.e. behavior at init
            # ≈ γ_base + Δγ_text (matches additive mode). During training the gate
            # can suppress γ_base (approaching 0) when text says the trajectory
            # signal is unreliable for this fund/edge.
            if self.tcetf_mode == "gated":
                self.gate_gamma = nn.ModuleDict()
                self.gate_beta = nn.ModuleDict()
                for etype in edge_types:
                    key = "__".join(etype)
                    gg = nn.Linear(self.tcetf_text_dim, hidden_dim)
                    gb = nn.Linear(self.tcetf_text_dim, hidden_dim)
                    nn.init.zeros_(gg.weight); nn.init.constant_(gg.bias, 5.0)
                    nn.init.zeros_(gb.weight); nn.init.constant_(gb.bias, 5.0)
                    self.gate_gamma[key] = gg
                    self.gate_beta[key] = gb
                print(f"[TCETF gated] γ = γ_base·σ(W_gate·text) + Δγ_text (σ≈0.993 at init "
                      f"so γ≈γ_base+Δγ_text initially).")
            print(f"[TCETF mode={self.tcetf_mode}] Text-conditioned edge trajectory FiLM enabled "
                  f"(text_dim={self.tcetf_text_dim}, hidden_dim={hidden_dim}, "
                  f"edge_types={[' '.join(e) for e in edge_types]}); "
                  f"init: identical to plain etraj.")

    def forward(self, edge_attr: torch.Tensor, etype,
                text_per_edge: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """edge_attr: (E, traj_dim) — h(t) for the edges of edge_type `etype`.

        text_per_edge: optional (E, tcetf_text_dim) — fund-endpoint's text. Used
            only when self.use_tcetf is True. Otherwise ignored.

        Returns (γ, β) each shaped (E, heads, head_dim). Both are zero at init.
        """
        key = "__".join(etype) if isinstance(etype, (list, tuple)) else etype
        if key not in self.gamma:
            raise KeyError(
                f"EdgeTrajectoryFiLM has no head for etype={etype!r}; "
                f"registered keys: {list(self.gamma.keys())}"
            )
        γ = self.gamma[key](edge_attr).view(-1, self.heads, self.head_dim)
        β = self.beta[key](edge_attr).view(-1, self.heads, self.head_dim)
        if self.use_tcetf and text_per_edge is not None and key in self.gamma_text:
            if self.tcetf_mode == "gated" and key in self.gate_gamma:
                gate_g = torch.sigmoid(self.gate_gamma[key](text_per_edge)).view(-1, self.heads, self.head_dim)
                gate_b = torch.sigmoid(self.gate_beta[key](text_per_edge)).view(-1, self.heads, self.head_dim)
                γ = γ * gate_g
                β = β * gate_b
            Δγ_text = self.gamma_text[key](text_per_edge).view(-1, self.heads, self.head_dim)
            Δβ_text = self.beta_text[key](text_per_edge).view(-1, self.heads, self.head_dim)
            γ = γ + Δγ_text
            β = β + Δβ_text
        return γ, β
