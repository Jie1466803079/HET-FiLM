"""Spatial text-stock alignment (Option 2 — spatiotemporal alignment, additive).

Sister module to TextBehaviorAlignment (temporal). Both produce per-edge "text
reliability" scores; combined via a learned posterior gate in MultiTaskEdgePredictor.

Components:
- SpatialAlignment: per-edge cosine(W_T · abs_proj[fund], W_S · stock_feat[stock]).
  Trained via InfoNCE on positive fund→stock edges.
- beta_prior_penalty: weak prior on the alignment posterior distribution
  (encourages alpha values to stay near 0.5 absent strong evidence).

Gated entirely behind `--use_spatial_alignment`. Risk-section text is excluded
by construction (this module consumes abs_proj which is computed from
strategy-only embedding when --risk_weight 0).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAlignment(nn.Module):
    """Per-edge text-stock spatial alignment.

    Provides three pieces of functionality:

    1. ``cosine_score(abs_proj, stock_features, edge_index)`` →
       per-edge cos(text, stock) score, shape (E,), in [-1, 1].

    2. ``infonce_loss(abs_proj, stock_features, pos_edge_index)`` →
       contrastive supervision: each fund's actually-held stocks should
       have higher alignment than other stocks.

    3. ``combined_gate(temporal_score, spatial_score, fund_idx)`` →
       sigmoid posterior α = σ(γ_T·temporal[f] + γ_S·spatial[e] + δ).
       Combines TBA's temporal score (per fund) with spatial (per edge).

    Inputs:
        abs_proj_dim: 128 (matches ProspectusTextFusion.ABS_PROJ_DIM)
        stock_feat_dim: 15 (raw pre-GNN stock feature dim on 2005Q3 dataset)
        align_dim: 64 (projection space for alignment)
        temperature: 0.1 (InfoNCE temperature)
    """

    DEFAULT_STOCK_FEAT_DIM = 15  # 2005Q3 stock_features dim

    def __init__(
        self,
        abs_proj_dim: int = 128,
        stock_feat_dim: int = DEFAULT_STOCK_FEAT_DIM,
        align_dim: int = 64,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.abs_proj_dim = abs_proj_dim
        self.stock_feat_dim = stock_feat_dim
        self.align_dim = align_dim
        self.temperature = temperature

        # Light projections; eager (not LazyLinear) so all params are in optimizer at construction.
        self.text_align_proj = nn.Sequential(
            nn.Linear(abs_proj_dim, align_dim),
            nn.LayerNorm(align_dim),
        )
        self.stock_align_proj = nn.Sequential(
            nn.Linear(stock_feat_dim, align_dim),
            nn.LayerNorm(align_dim),
        )

        # Combined-gate learnable scalars. γ_T, γ_S init positive; δ init 0
        # so α at init ≈ σ(0.5·tem + 0.5·spt) — neither dominates.
        self.gamma_temporal = nn.Parameter(torch.tensor(0.5))
        self.gamma_spatial = nn.Parameter(torch.tensor(0.5))
        self.gate_delta = nn.Parameter(torch.tensor(0.0))

        # Attention-bias scale (used by HGT+ attention modulation).
        # Init 0 → no effect at start; model learns to amplify if useful.
        self.attention_beta = nn.Parameter(torch.tensor(0.0))

        # Per-fund InfoNCE universe mask, stashed externally by MultiTaskEdgePredictor
        # in _apply_prospectus_fusion when --spa_negative_scope=per_fund. None (default)
        # → infonce_loss uses the global stock pool (baseline behaviour). Lets the
        # trainer call infonce_loss(...) with no extra args while still picking up
        # the per-fund mask transparently.
        self._universe_mask_for_batch: Optional[torch.Tensor] = None

        # Monitoring (read by run_model.py for results JSON)
        self._last_spatial_mean: float = 0.0
        self._last_spatial_pos_mean: float = 0.0
        self._last_spatial_neg_mean: float = 0.0
        self._last_combined_alpha_mean: float = 0.5
        self._last_n_pos: int = 0
        self._last_used_per_fund_negatives: bool = False

    def _project(self, abs_proj: torch.Tensor, stock_features: torch.Tensor):
        """Return (text_unit, stock_unit) — L2-normalized projections."""
        t = F.normalize(self.text_align_proj(abs_proj), dim=-1)
        s = F.normalize(self.stock_align_proj(stock_features), dim=-1)
        return t, s

    def cosine_score(
        self,
        abs_proj: torch.Tensor,
        stock_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Per-edge cos(text, stock) in [-1, 1]. Shape: (E,)."""
        t, s = self._project(abs_proj, stock_features)
        fund_idx, stock_idx = edge_index[0], edge_index[1]
        scores = (t[fund_idx] * s[stock_idx]).sum(dim=-1)
        with torch.no_grad():
            self._last_spatial_mean = float(scores.mean().item())
        return scores

    def infonce_loss(
        self,
        abs_proj: torch.Tensor,
        stock_features: torch.Tensor,
        pos_edge_index: torch.Tensor,
        fund_universe_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """InfoNCE: each fund's positive-edge stock vs. negatives.

        For each (f, s+) positive edge, the anchor is text[f] and the target is
        stock_idx s+. Cross-entropy over candidate stocks supervises the projections.

        Negative pool:
        - ``fund_universe_mask=None`` (default): every stock in ``stock_features``
          is a candidate ("global" pool — original behaviour).
        - ``fund_universe_mask: (N_fund, N_stock) bool``: per-fund candidate set.
          For anchor f, only stocks where ``fund_universe_mask[f]`` is True act
          as negatives; the positive ``s+`` is always added so cross-entropy has
          a valid target slot. Used by ``--spa_negative_scope per_fund``.
        """
        if pos_edge_index is None or pos_edge_index.numel() == 0:
            return abs_proj.new_tensor(0.0)
        if stock_features.shape[0] < 2:
            return abs_proj.new_tensor(0.0)

        t, s = self._project(abs_proj, stock_features)
        fund_idx = pos_edge_index[0]
        stock_idx = pos_edge_index[1]
        anchor = t[fund_idx]                                    # (E_pos, align_dim)

        # (E_pos, N_stock): each row is anchor's similarity to every stock
        logits = (anchor @ s.t()) / self.temperature

        # Pick up the universe mask either from the explicit kwarg or, if absent,
        # from self._universe_mask_for_batch (set externally by MultiTaskEdgePredictor
        # in --spa_negative_scope=per_fund mode). None on both → global pool.
        effective_mask = (
            fund_universe_mask
            if fund_universe_mask is not None
            else self._universe_mask_for_batch
        )
        if effective_mask is not None:
            # Per-fund InfoNCE: restrict negatives to each anchor fund's universe.
            # Force-include the positive index so cross-entropy always has a valid
            # target slot (relevant for NEW_EDGE_STRICT positives that are not in
            # the fund's prior support-window holdings).
            edge_mask = effective_mask[fund_idx]                # (E_pos, N_stock)
            if edge_mask.dtype != torch.bool:
                edge_mask = edge_mask.bool()
            edge_mask = edge_mask.clone()
            e_arange = torch.arange(edge_mask.shape[0], device=edge_mask.device)
            edge_mask[e_arange, stock_idx] = True
            logits = logits.masked_fill(~edge_mask, float('-inf'))
            self._last_used_per_fund_negatives = True
        else:
            self._last_used_per_fund_negatives = False

        loss = F.cross_entropy(logits, stock_idx)

        with torch.no_grad():
            # Monitoring: avg similarity for pos pairs vs random pairs
            pos_sims = (anchor * s[stock_idx]).sum(dim=-1)
            self._last_spatial_pos_mean = float(pos_sims.mean().item())
            self._last_n_pos = int(pos_edge_index.shape[1])
            # Approximate negative similarity: sample random stocks
            n_sample = min(64, stock_features.shape[0])
            rand_idx = torch.randint(0, stock_features.shape[0], (anchor.shape[0], n_sample), device=anchor.device)
            neg_sims = (anchor.unsqueeze(1) * s[rand_idx]).sum(dim=-1)   # (E_pos, n_sample)
            self._last_spatial_neg_mean = float(neg_sims.mean().item())

        return loss

    def attention_bias(
        self,
        abs_proj: torch.Tensor,
        stock_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """β · cos(text, stock) per edge — for HGT+ attention modulation.

        Same projections as cosine_score, with the learnable β multiplier.
        Returns (E,) tensor to be added to pre-softmax attention logits.
        """
        scores = self.cosine_score(abs_proj, stock_features, edge_index)  # (E,)
        return self.attention_beta * scores                                # (E,)

    def combined_gate(
        self,
        temporal_align_per_fund: torch.Tensor,        # (N_fund,) — TBA's align score (cos, in [-1,1])
        spatial_align_per_edge: torch.Tensor,         # (E,) — spatial score (cos, in [-1,1])
        fund_idx_per_edge: torch.Tensor,              # (E,) — fund index for each edge
    ) -> torch.Tensor:
        """Per-edge alignment posterior α = σ(γ_T·temp_f + γ_S·spat_{f,s} + δ).

        Returns (E,) gate values in (0, 1).
        """
        # Broadcast per-fund temporal to per-edge
        temp_per_edge = temporal_align_per_fund[fund_idx_per_edge]      # (E,)
        gate_logits = (
            self.gamma_temporal * temp_per_edge
            + self.gamma_spatial * spatial_align_per_edge
            + self.gate_delta
        )
        alpha = torch.sigmoid(gate_logits)
        with torch.no_grad():
            self._last_combined_alpha_mean = float(alpha.mean().item())
        return alpha


def beta_prior_penalty(alpha: torch.Tensor, a0: float = 2.0, b0: float = 2.0) -> torch.Tensor:
    """Negative log-density of α under Beta(a0, b0). Acts as a soft prior.

    With (a0, b0) = (2, 2), the prior is centered at 0.5 with moderate variance —
    discourages α collapse to 0 or 1 without strong evidence.

    Args:
        alpha: tensor of values in (0, 1) — typically the combined-gate output.
        a0, b0: Beta distribution shape parameters. Default Beta(2,2) is a weak prior.

    Returns:
        scalar mean -log p_Beta(α). Lower = closer to prior.
    """
    eps = 1e-6
    a = alpha.clamp(eps, 1.0 - eps)
    # -log Beta(α; a0, b0) up to a normalizing constant (which doesn't affect gradients):
    #   = -((a0-1)log α + (b0-1)log(1-α))
    log_a = torch.log(a)
    log_1ma = torch.log(1.0 - a)
    return -((a0 - 1.0) * log_a + (b0 - 1.0) * log_1ma).mean()
