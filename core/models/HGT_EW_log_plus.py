"""HGT with log1p-scaled edge-weight message-passing (the EWlog mechanism
from HGT_EW_log.py) plus a wrapper-compatible `_encode_snapshot` signature.

Motivation
----------
The existing `HGTEWLog._encode_snapshot(self, data)` in HGT_EW_log.py does
not accept `edge_bias_dict` / `edge_attr_traj_dict` kwargs, but
`HGT.encode` unconditionally forwards those kwargs at line 174. When the
outer `MultiTaskEdgePredictor.encode` path is used (as with text-fusion
`--use_prospectus` runs) this composition currently `TypeError`s.

`HGTEWLogPlus` copies the exact EWlog mechanism (log1p(clamp(w, 0)) on
per-edge-type edge_attr, then `HGTConvEW` message-passing that multiplies
`v_j` by the log-scaled weight) into a class whose `_encode_snapshot`
accepts and silently ignores the extra kwargs — matching HGT._encode_snapshot's
signature. This makes the EWlog mechanism composable with
ProspectusTextFusion at the outer wrapper without touching any existing
file.

Baseline behavior of `HGT+EWlog` / `HGT_EW_log.py` is not modified.

Registered as model name `"HGT+EWlogPlus"` in load_model.py.
"""
from typing import Optional, Dict

import torch
import torch.nn as nn

from .HGT import HGT
from .HGT_EW import HGTConvEW


class HGTEWLogPlus(HGT):
    """HGT with `v_j *= log1p(clamp(w, min=0))` on every edge type carrying
    an `edge_attr`. Wrapper-compatible `_encode_snapshot` signature.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        new_convs = nn.ModuleList()
        for conv in self.convs:
            ew_conv = HGTConvEW(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                metadata=(
                    list(conv.k_lin.keys()),
                    [tuple(k.split("__")) for k in conv.a_rel.keys()],
                ),
                heads=conv.heads,
                group=conv.group,
                use_RTE=conv.use_RTE,
            )
            new_convs.append(ew_conv)
        self.convs = new_convs
        print(f"[EWlogPlus] HGTConv → HGTConvEW substitution complete "
              f"({len(self.convs)} layers); v_j *= log1p(w) per edge type.")

    def _encode_snapshot(self, data, edge_bias_dict=None, edge_attr_traj_dict=None):
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

        edge_attr_dict: Dict = {}
        for etype in edge_index_dict:
            store = data[etype]
            if (hasattr(store, 'edge_attr')
                    and store.edge_attr is not None
                    and store.edge_attr.numel() > 0):
                ea = store.edge_attr.float()
                if ea.dim() > 1:
                    ea = ea.squeeze(-1)
                edge_attr_dict[etype] = torch.log1p(torch.clamp(ea, min=0.0))

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_time_dict,
                          edge_attr_dict=edge_attr_dict)
            if self.drop is not None:
                for k in x_dict:
                    if x_dict[k] is not None:
                        x_dict[k] = self.drop(x_dict[k])

        return x_dict
