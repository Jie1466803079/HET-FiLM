"""Text-only edge prediction head (Component 2 C2-4 — ablation appendix).

Standalone auxiliary head that predicts edge existence from `abs_proj(E_f)` and
stock features alone, with no GNN involvement. Trained with BCE on the same
Stage-1 candidate edges (positive + negative) that supervise the main model.

Gated entirely behind `--use_text_only_aux_loss`. Default off.

Two purposes:
1. Provides a "text-only AUC" diagnostic for the paper — quantifies how much
   of the prediction signal lives in text alone.
2. Provides a direct BCE gradient to `abs_proj` complementing the contrastive
   gradient from Spatial Alignment's InfoNCE.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TextOnlyEdgeHead(nn.Module):
    """Bilinear edge-existence predictor using (text, stock) features only.

    Inputs:
        abs_proj_per_edge:   (E, abs_proj_dim) text projection for the SRC fund of each edge
        stock_feat_per_edge: (E, stock_feat_dim) raw stock features for the DST stock

    Returns: (E,) per-edge logit (raw, no sigmoid). Use with BCEWithLogitsLoss.
    """

    DEFAULT_STOCK_FEAT_DIM = 15  # 2005Q3 dataset

    def __init__(
        self,
        abs_proj_dim: int = 128,
        stock_feat_dim: int = DEFAULT_STOCK_FEAT_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.abs_proj_dim = abs_proj_dim
        self.stock_feat_dim = stock_feat_dim
        self.hidden_dim = hidden_dim

        self.text_proj = nn.Sequential(
            nn.Linear(abs_proj_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.stock_proj = nn.Sequential(
            nn.Linear(stock_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

        # Monitoring for the firing-OK log line
        self._last_logit_mean: float = 0.0
        self._last_n_edges: int = 0

    def forward(
        self,
        abs_proj_per_edge: torch.Tensor,
        stock_feat_per_edge: torch.Tensor,
    ) -> torch.Tensor:
        t = self.text_proj(abs_proj_per_edge)
        s = self.stock_proj(stock_feat_per_edge)
        logits = self.bilinear(t, s).squeeze(-1)              # (E,)
        with torch.no_grad():
            self._last_logit_mean = float(logits.mean().item())
            self._last_n_edges = int(logits.shape[0])
        return logits
