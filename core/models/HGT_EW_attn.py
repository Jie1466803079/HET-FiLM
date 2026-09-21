"""HGT+EW variant with attention-bias edge weighting (Strategy B) + log1p (C).

Subclasses :class:`HGTConvEW` and overrides ``message`` ONLY — ``forward`` is
inherited verbatim, so no unintended logic changes vs the existing HGT_EW
implementation.

Difference from HGT_EW.HGTConvEW:
- HGT_EW pre-multiplies value vectors:  v_j' = v_j * edge_w  (before softmax)
- This module instead biases attention LOGITS:
        alpha_ij = (q_i k_j) / sqrt(D) + gamma_h * log1p(edge_w_ij)
    then softmax. Lets attention learn whether/how much to weight edge-weight
    information rather than baking the multiplier into v.

The per-head scalar ``gamma`` is initialised to 0 so that at init the layer
behaves identically to base HGT (no EW influence). Gradient descent then
chooses whether to use the EW signal.

Combined with ``STAGE1_WEIGHTED_BCE=1`` this is the A+B+C configuration of
the edge-weight design experiment. Registered as ``"HGT+EWattn"``.
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax

from .HGT_EW import HGTConvEW, HGTEW


class HGTConvEWAttn(HGTConvEW):
    """HGTConvEW with attention-bias edge weighting (replaces v scaling)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-head learnable bias gain; init 0 → identical to base HGT at init.
        self.gamma = nn.Parameter(torch.zeros(self.heads))

    def message(
        self,
        k_j: Tensor,
        q_i: Tensor,
        v_j: Tensor,
        rel: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
        edge_time_k=None,
        edge_time_v=None,
        edge_w=None,
    ) -> Tensor:
        if self.use_RTE and edge_time_k is not None:
            k_j = k_j + edge_time_k
        if self.use_RTE and edge_time_v is not None:
            v_j = v_j + edge_time_v

        # Standard HGT attention score
        alpha = (q_i * k_j).sum(dim=-1) * rel
        alpha = alpha / math.sqrt(q_i.size(-1))

        # Strategy B+C: additive attention bias from log1p(edge_w),
        # per-head learnable gain (init 0 → identical to base at init).
        if edge_w is not None:
            ew_log = torch.log1p(torch.clamp(edge_w, min=0.0))
            alpha = alpha + self.gamma.view(1, -1) * ew_log.view(-1, 1)

        alpha = softmax(alpha, index, ptr, size_i)
        # NOTE: no v_j *= edge_w pre-multiplication (replaced by alpha bias).
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)


class HGTEWAttn(HGTEW):
    """HGT variant using HGTConvEWAttn (attention-bias edge weighting)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace each HGTConvEW with HGTConvEWAttn (same params)
        new_convs = nn.ModuleList()
        for conv in self.convs:
            ew_conv = HGTConvEWAttn(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                metadata=(list(conv.k_lin.keys()), [
                    tuple(k.split("__")) for k in conv.a_rel.keys()
                ]),
                heads=conv.heads,
                group=conv.group,
                use_RTE=conv.use_RTE,
            )
            new_convs.append(ew_conv)
        self.convs = new_convs
