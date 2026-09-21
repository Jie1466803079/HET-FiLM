"""Concat-based fund fusion: text embedding directly concatenated with numerical.

Sibling of ``prospectus_fusion.ProspectusTextFusion`` (gated / convex-residual).
Selected via ``--prospectus_fusion_mode concat`` and swapped in by run_model.py
when that flag is set. Existing gated path is untouched.

Fusion op:

    abs_proj_out = abs_proj(abs_emb)                       # (B, ABS_PROJ_DIM=128)
    text_input   = 0 for cold-start rows, abs_proj_out otherwise  (no staleness)
    num_input    = num_proj(numerical)                     # (B, OUTPUT_DIM=128)
    h_fund       = concat_proj([text_input, num_input])    # 256 → OUTPUT_DIM=128

No gate, no fund-aware prior, no Jacobian-ratio input, no staleness sigmoid
(fixed by design — this variant is the raw concat baseline). Cold-start funds
(delta_t == -1, including Case-3 zero-strategy cells zeroed by the loader) get
a hard-zero on the text branch so they never leak the shared abs_proj(0)
constant into h_fund.

Class attributes ``ABS_PROJ_DIM`` and ``OUTPUT_DIM`` are pinned at 128 to
match ``ProspectusTextFusion`` so downstream heads (SpatialAlignment,
SpatialAttentionBias, TextOnlyHead, TBA, ToA) that read those constants keep
working unchanged.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class ProspectusTextConcatFusion(nn.Module):
    ABS_PROJ_DIM = 128
    OUTPUT_DIM = 128

    def __init__(
        self,
        input_dim: int = 300,
        numerical_dim: int = 11,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.INPUT_DIM = int(input_dim)
        self.numerical_dim = int(numerical_dim)
        self.no_staleness = True
        self.staleness_scale = 1.0

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
        self.num_proj = nn.Linear(self.numerical_dim, self.OUTPUT_DIM)
        self.concat_proj = nn.Linear(
            self.ABS_PROJ_DIM + self.OUTPUT_DIM, self.OUTPUT_DIM
        )

        self.use_fund_aware_gate = False
        self.use_jacobian_ratio_gate = False
        self.no_text_fund_feature = False
        self.fusion_mode = "concat"

        self._last_gate_mean: float = 0.0
        self._last_staleness_mean: float = 1.0
        self._last_log_ratio_mean: float = 0.0

        print(
            f"[PROSPECTUS][CONCAT] ProspectusTextConcatFusion: "
            f"input_dim={self.INPUT_DIM}, numerical_dim={self.numerical_dim}, "
            f"OUTPUT_DIM={self.OUTPUT_DIM}, no_staleness=True (hardcoded), "
            f"cold-start hard-zero on text branch"
        )

    def forward(
        self,
        text_features: Dict[str, torch.Tensor],
        numerical: torch.Tensor,
        behavior_gate: torch.Tensor = None,
        log_ratio: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        abs_emb = text_features["abs_emb"]
        delta_t = text_features["delta_t"]

        abs_proj_out = self.abs_proj(abs_emb)
        cold_mask = (delta_t == -1).unsqueeze(-1)
        text_input = abs_proj_out.masked_fill(cold_mask, 0.0)

        if behavior_gate is not None:
            text_input = behavior_gate.unsqueeze(-1) * text_input

        num_input = self.num_proj(numerical)
        h_fund = self.concat_proj(torch.cat([text_input, num_input], dim=-1))

        return h_fund, abs_proj_out
