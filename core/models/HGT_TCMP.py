"""HGT + Text-Conditioned Message Passing (TCMP).

ASYMMETRIC architecture for graphs where only ONE node type has text
(here: fund has prospectus text; stock does not).

Standard HGT (HGT.py) projects fund features via `lin_dict['fund']` once at
the input, then runs `num_layers` HGTConv layers. Text influences the model
ONLY through the input projection (via TF's gated h_fund in
ProspectusTextFusion). Aux losses (TBA/SpAB/SpatialAlign) shape the text
encoder's projection matrices but the gradient never reaches the GNN's
per-layer parameters — which is the empirical bottleneck this code addresses.

TCMP injects text into the GNN at EVERY layer via per-layer FiLM modulation
applied to fund features:

    h_fund_l+1 = γ_l(text_f) * h_fund_l + β_l(text_f)        (after HGTConv layer l)

where γ_l, β_l ∈ R^hid_dim are learned linear maps of fund-side text.
Stock features are NOT modulated (asymmetric — stocks have no text). The
FiLM modules are zero-initialized (γ bias=1, β bias=0), so at init TCMP
is byte-identical to HGT, then deviates as training discovers useful
text-conditional interactions.

This module is opt-in: it activates ONLY when --use_tcmp is set AND
text_per_snapshot is passed through encode(). With either condition off,
HGT_TCMP behaves identically to HGT.
"""
from typing import List, Optional

import torch
import torch.nn as nn

from .HGT import HGT


class TCMPFiLMLayer(nn.Module):
    """Per-layer FiLM modulator on fund features.

    Init: γ ≈ 1, β ≈ 0 → layer is identity at construction time.
    Stock features pass through unchanged (asymmetric one-sided text).
    """

    def __init__(self, text_dim: int = 128, hid_dim: int = 64):
        super().__init__()
        self.gamma_proj = nn.Linear(text_dim, hid_dim)
        self.beta_proj = nn.Linear(text_dim, hid_dim)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h_fund: torch.Tensor, text_fund: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(text_fund)
        beta = self.beta_proj(text_fund)
        return gamma * h_fund + beta


class HGT_TCMP(HGT):
    """HGT with Text-Conditioned Message Passing on fund features.

    The text vector is expected to be a per-snapshot tensor of shape
    (N_funds_in_snapshot, text_dim). If text is provided to encode()
    via `text_per_snapshot=[t_0, t_1, ..., t_T-1]`, FiLM applies per-layer
    after each HGTConv layer. Otherwise behaves identically to HGT.
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_heads,
        num_layers,
        metadata,
        predict_type,
        use_RTE=False,
        featemb=None,
        nclf_linear=None,
        dropout: float = 0.0,
        time_window: int = 1,
        tcmp_text_dim: int = 128,
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            metadata=metadata,
            predict_type=predict_type,
            use_RTE=use_RTE,
            featemb=featemb,
            nclf_linear=nclf_linear,
            dropout=dropout,
            time_window=time_window,
        )
        self.tcmp_text_dim = int(tcmp_text_dim)
        self.tcmp_film_layers = nn.ModuleList([
            TCMPFiLMLayer(text_dim=self.tcmp_text_dim, hid_dim=hidden_channels)
            for _ in range(num_layers)
        ])
        print(f"[TCMP] Text-conditioned message passing enabled "
              f"(text_dim={self.tcmp_text_dim}, hid_dim={hidden_channels}, "
              f"n_layers={num_layers}); init=identity (γ=1,β=0).")

    def _encode_snapshot(self, data, edge_bias_dict=None, edge_attr_traj_dict=None,
                        text_for_fund: Optional[torch.Tensor] = None):
        """Forward pass with optional per-layer FiLM on fund features.

        If text_for_fund is None, behavior is identical to HGT._encode_snapshot.
        """
        x_dict = self.featemb(data.x_dict)
        edge_index_dict = data.edge_index_dict

        edge_time_dict = None
        if self.convs[0].use_RTE:
            et = {}
            for etype in edge_index_dict:
                if hasattr(data[etype], 'edge_time'):
                    et[etype] = data[etype].edge_time.squeeze(-1)
            if et:
                edge_time_dict = et

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        # TCMP guard: only apply if text matches the snapshot's fund count
        apply_tcmp = (
            text_for_fund is not None
            and 'fund' in x_dict
            and x_dict['fund'] is not None
            and x_dict['fund'].shape[0] == text_for_fund.shape[0]
        )

        for l, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict, edge_time_dict,
                          edge_bias_dict=edge_bias_dict,
                          edge_attr_traj_dict=edge_attr_traj_dict)
            if apply_tcmp:
                x_dict['fund'] = self.tcmp_film_layers[l](
                    x_dict['fund'], text_for_fund
                )
            if self.drop is not None:
                for k in x_dict:
                    if x_dict[k] is not None:
                        x_dict[k] = self.drop(x_dict[k])

        return x_dict

    def encode(self, data, *args, return_dict=False,
               edge_bias_dict_per_snapshot=None,
               edge_attr_traj_dict_per_snapshot=None,
               text_per_snapshot: Optional[List[torch.Tensor]] = None,
               **kwargs):
        """Same signature as HGT.encode plus `text_per_snapshot` for TCMP.

        text_per_snapshot: optional list of length T, where each entry is
            a (N_funds_in_snapshot, text_dim) tensor providing the per-fund
            text vector for FiLM modulation in that snapshot. If None, no
            FiLM is applied — behaves like vanilla HGT.
        """
        if isinstance(data, (list, tuple)):
            graphs = data
        else:
            graphs = [data]

        T = len(graphs)
        if edge_bias_dict_per_snapshot is not None:
            assert len(edge_bias_dict_per_snapshot) == T, \
                f"edge_bias_dict_per_snapshot len {len(edge_bias_dict_per_snapshot)} != #graphs {T}"
        if edge_attr_traj_dict_per_snapshot is not None:
            assert len(edge_attr_traj_dict_per_snapshot) == T, \
                f"edge_attr_traj_dict_per_snapshot len {len(edge_attr_traj_dict_per_snapshot)} != #graphs {T}"
        if text_per_snapshot is not None:
            assert len(text_per_snapshot) == T, \
                f"text_per_snapshot len {len(text_per_snapshot)} != #graphs {T}"

        per_snap = []
        for t, g in enumerate(graphs):
            ebd = edge_bias_dict_per_snapshot[t] if edge_bias_dict_per_snapshot is not None else None
            etd = edge_attr_traj_dict_per_snapshot[t] if edge_attr_traj_dict_per_snapshot is not None else None
            tt = text_per_snapshot[t] if text_per_snapshot is not None else None
            per_snap.append(self._encode_snapshot(g, edge_bias_dict=ebd,
                                                 edge_attr_traj_dict=etd,
                                                 text_for_fund=tt))

        predict_type = self.predict_type

        if return_dict:
            return {k: v for k, v in per_snap[-1].items() if v is not None}

        if self.temporal_attn is not None and len(graphs) > 1:
            if isinstance(predict_type, list):
                stacks = [torch.stack([snap[pt] for snap in per_snap], dim=1)
                          for pt in predict_type]
                aggregated = [self.temporal_attn(s).mean(dim=1) for s in stacks]
                return aggregated
            stack = torch.stack([snap[predict_type] for snap in per_snap], dim=1)
            aggregated = self.temporal_attn(stack).mean(dim=1)
            return aggregated

        if isinstance(predict_type, list):
            return [torch.stack([snap[pt] for snap in per_snap], dim=0).mean(dim=0)
                    for pt in predict_type]
        return torch.stack([snap[predict_type] for snap in per_snap], dim=0).mean(dim=0)
