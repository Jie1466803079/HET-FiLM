"""Text-behavior trajectory alignment (Option 1 — additive).

Computes a per-fund alignment between text trajectory (Δstrategy_emb) and
portfolio trajectory (Δh built from holdings + raw stock features, GNN-free).

Gated entirely behind `--use_text_behavior_alignment`. Risk-section text is
excluded by construction (this module consumes the strategy embedding only,
via ProspectusEmbeddingLoader.get_strategy_delta — risk_weight irrelevant).

Provides:
- portfolio_summary_per_fund(): per-fund weighted mean/var/HHI/n_eff from raw stock features.
- churn_per_fund(): per-fund 1−Jaccard between two snapshots' holdings sets.
- TextBehaviorAlignment: learns aligned projections, returns gate + InfoNCE loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def portfolio_summary_per_fund(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    stock_features: torch.Tensor,
    N_funds: int,
) -> torch.Tensor:
    """Per-fund [weighted_mean | weighted_var | HHI | n_eff].

    Args:
        edge_index:     (2, E) fund→stock edges for one snapshot.
        edge_weight:    (E,)   raw weight per edge.
        stock_features: (N_stock, F) raw pre-GNN stock features.
        N_funds:        scalar.

    Returns:
        (N_funds, 2*F + 2). Rows for funds with no edges in this snapshot
        return zeros (caller masks those out via valid_mask).
    """
    device = edge_weight.device
    F_ = stock_features.shape[-1]
    eps = 1e-8

    ei0, ei1 = edge_index[0], edge_index[1]
    w = edge_weight
    sx = stock_features[ei1]                                                  # (E, F)

    total_w = torch.zeros(N_funds, device=device).index_add_(0, ei0, w)
    total_w_safe = total_w.clamp_min(eps).unsqueeze(-1)                       # (N, 1)

    weighted_sum = torch.zeros(N_funds, F_, device=device)
    weighted_sum.index_add_(0, ei0, w.unsqueeze(-1) * sx)
    h_mean = weighted_sum / total_w_safe                                      # (N, F)

    weighted_sq = torch.zeros(N_funds, F_, device=device)
    weighted_sq.index_add_(0, ei0, w.unsqueeze(-1) * sx * sx)
    h_var = (weighted_sq / total_w_safe) - h_mean * h_mean
    h_var = h_var.clamp_min(0.0)                                              # guard fp negatives

    w_norm = w / total_w[ei0].clamp_min(eps)
    hhi = torch.zeros(N_funds, device=device).index_add_(0, ei0, w_norm * w_norm)
    n_eff = 1.0 / hhi.clamp_min(eps)

    return torch.cat([h_mean, h_var, hhi.unsqueeze(-1), n_eff.unsqueeze(-1)], dim=-1)


def churn_per_fund(
    edge_index_t: torch.Tensor,
    edge_index_tm1: torch.Tensor,
    N_funds: int,
    N_stocks: int,
) -> torch.Tensor:
    """Per-fund churn = 1 − Jaccard(holdings_t, holdings_{t−1}). Returns (N_funds,)."""
    device = edge_index_t.device
    fund_t, stock_t = edge_index_t[0], edge_index_t[1]
    fund_tm1, stock_tm1 = edge_index_tm1[0], edge_index_tm1[1]

    # Pack (fund, stock) into scalar hash for set membership.
    hash_t = fund_t * N_stocks + stock_t
    hash_tm1 = fund_tm1 * N_stocks + stock_tm1
    in_both = torch.isin(hash_t, hash_tm1)

    ones_t = torch.ones_like(fund_t, dtype=torch.float)
    ones_tm1 = torch.ones_like(fund_tm1, dtype=torch.float)

    n_t = torch.zeros(N_funds, device=device).index_add_(0, fund_t, ones_t)
    n_tm1 = torch.zeros(N_funds, device=device).index_add_(0, fund_tm1, ones_tm1)
    n_intersect = torch.zeros(N_funds, device=device).index_add_(
        0, fund_t[in_both], ones_t[in_both]
    )
    n_union = (n_t + n_tm1 - n_intersect).clamp_min(1.0)
    return 1.0 - n_intersect / n_union


class TextBehaviorAlignment(nn.Module):
    """Per-fund alignment between text Δ and portfolio Δ.

    Forward:
        delta_E:    (N, text_dim)   strategy embedding delta (strategy section only).
        delta_h:    (N, behav_dim)  portfolio summary delta.
        valid_mask: (N,) bool       False for cold-start or missing-text funds.

    Returns:
        align: (N,)   cosine ∈ [−1, 1], 0 for invalid funds.
        gate:  (N,)   σ(γ·align + δ), 0.5 for invalid funds (neutral).
        L_align: scalar  symmetric InfoNCE over valid funds (0 if <2 valid).
    """

    PROJ_DIM = 128
    # Default behavior-summary dim: 2*stock_feat_dim + 2 (HHI, n_eff) + 1 (churn).
    # For the 2005Q3 dataset stock_feat_dim=15 → 2*15+2+1 = 33.
    DEFAULT_BEHAV_DIM = 33

    def __init__(self, text_dim: int = 1024, behav_dim: int = DEFAULT_BEHAV_DIM,
                 temperature: float = 0.1):
        super().__init__()
        self.text_dim = text_dim
        self.behav_dim = behav_dim
        self.temperature = temperature

        self.text_delta_proj = nn.Sequential(
            nn.Linear(text_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, self.PROJ_DIM), nn.LayerNorm(self.PROJ_DIM),
        )
        # FIX (bug #2): eager Linear so params are in the optimizer at construction
        # time. The prior LazyLinear materialized after optimizer creation, leaving
        # behav_delta_proj permanently frozen at random init.
        self.behav_delta_proj = nn.Sequential(
            nn.Linear(behav_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, self.PROJ_DIM), nn.LayerNorm(self.PROJ_DIM),
        )
        self.gate_gamma = nn.Parameter(torch.tensor(2.0))
        self.gate_delta = nn.Parameter(torch.tensor(0.0))

        # Monitoring (read by run_model.py for the results JSON)
        self._last_align_mean: float = 0.0
        self._last_gate_mean: float = 0.5
        self._last_valid_frac: float = 0.0

    def forward(self, delta_E, delta_h, valid_mask=None):
        if valid_mask is None:
            valid_mask = torch.ones(delta_E.shape[0], dtype=torch.bool, device=delta_E.device)
        u = F.normalize(self.text_delta_proj(delta_E), dim=-1)
        v = F.normalize(self.behav_delta_proj(delta_h), dim=-1)
        align = (u * v).sum(-1)
        gate = torch.sigmoid(self.gate_gamma * align + self.gate_delta)

        align_out = torch.where(valid_mask, align, torch.zeros_like(align))
        gate_out = torch.where(valid_mask, gate, torch.full_like(gate, 0.5))

        L_align = self._infonce(u, v, valid_mask)

        with torch.no_grad():
            if valid_mask.any():
                self._last_align_mean = float(align[valid_mask].mean().item())
                self._last_gate_mean = float(gate[valid_mask].mean().item())
            self._last_valid_frac = float(valid_mask.float().mean().item())

        return align_out, gate_out, L_align

    def _infonce(self, u, v, valid_mask):
        if int(valid_mask.sum().item()) < 2:
            return u.new_tensor(0.0)
        uv, vv = u[valid_mask], v[valid_mask]
        logits = (uv @ vv.t()) / self.temperature
        labels = torch.arange(uv.shape[0], device=uv.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
