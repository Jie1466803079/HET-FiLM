"""HGT with Edge-Weight scaling + TCETF FiLM on v_j.

Combines the two mechanisms from existing files (each unchanged):
  - HGT_EW.py  : HGTConvEW.message applies v_j *= edge_w
  - HGT.py     : HGTConv.message applies v_j = v_j*(1+γ) + β when FiLM (γ,β) given

This module composes both in a single message-passing step, in that order:

    v_j = v_j * edge_w                             (HGT+EW scaling)
    v_j = v_j * (1 + film_gamma) + film_beta       (TCETF FiLM on top)

New classes:
  HGTConvEWFiLM  — subclasses HGTConv; forward+message accept both edge_w and (γ,β).
  HGTEWTCETF    — subclasses HGT; _encode_snapshot builds edge_attr_dict (raw weights,
                   like HGTEW) AND forwards edge_attr_traj_dict (TCETF pipeline).

Registered as model name "HGT+EW+TCETF" in scripts/run/run_model.py.

No existing files modified; existing HGT+EW / HGT+EWlogPlus behavior preserved.
"""
import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.utils import softmax

from .HGT import HGT, HGTConv, group as _group


class HGTConvEWFiLM(HGTConv):
    """HGTConv with per-edge-weight scaling AND per-edge FiLM (γ, β) on v_j.

    Extends HGTConv by accepting two additional per-edge signals in forward():
      - edge_attr_dict[etype] → per-edge scalar weight (like HGTConvEW)
      - edge_attr_traj_dict[etype] → (γ, β) tuple of (E, heads, head_dim) tensors
        (from EdgeTrajectoryFiLM, i.e. TCETF pipeline)

    Applied in message():
      v_j *= edge_w                       (if edge_w given)
      v_j = v_j*(1+film_gamma) + film_beta (if TCETF FiLM given)
    """

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[
            Dict[EdgeType, Tensor], Dict[EdgeType, SparseTensor]
        ],
        edge_time_dict=None,
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
        edge_attr_traj_dict: Optional[Dict[EdgeType, Tuple[Tensor, Tensor]]] = None,
    ) -> Dict[NodeType, Optional[Tensor]]:
        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}
        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)
            q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
            v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        use_rte = self.use_RTE and (edge_time_dict is not None)

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_time = None
            if use_rte and edge_type in edge_time_dict:
                edge_time = edge_time_dict[edge_type].squeeze(-1)

            edge_w = None
            if edge_attr_dict is not None and edge_type in edge_attr_dict:
                ew = edge_attr_dict[edge_type]
                if ew is not None and ew.numel() > 0:
                    edge_w = ew.float()
                    if edge_w.dim() > 1:
                        edge_w = edge_w.squeeze(-1)

            film_gamma, film_beta = None, None
            if edge_attr_traj_dict is not None and edge_type in edge_attr_traj_dict:
                film_gamma, film_beta = edge_attr_traj_dict[edge_type]

            edge_type_str = "__".join(edge_type)
            a_rel = self.a_rel[edge_type_str]
            m_rel = self.m_rel[edge_type_str]

            time_emb_k, time_emb_v = None, None
            if use_rte and edge_time is not None:
                time_emb = self.rte(edge_time)
                time_emb_k = (
                    self.k_lin[src_type](time_emb).view(-1, H, D).transpose(0, 1) @ a_rel
                ).transpose(1, 0)
                time_emb_v = (
                    self.v_lin[src_type](time_emb).view(-1, H, D).transpose(0, 1) @ m_rel
                ).transpose(1, 0)

            k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)
            v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)

            prop_kwargs = dict(
                k=k, q=q_dict[dst_type], v=v,
                rel=self.p_rel[edge_type_str],
                edge_w=edge_w,
                film_gamma=film_gamma,
                film_beta=film_beta,
                size=None,
            )
            if use_rte and (time_emb_k is not None or time_emb_v is not None):
                prop_kwargs['edge_time_k'] = time_emb_k
                prop_kwargs['edge_time_v'] = time_emb_v

            out = self.propagate(edge_index, **prop_kwargs)
            out_dict[dst_type].append(out)

        # Node-level aggregation + skip connection (unchanged from HGTConv / HGTConvEW)
        for node_type, outs in out_dict.items():
            out = _group(outs, self.group)
            if out is None:
                out_dict[node_type] = None
                continue
            out = self.a_lin[node_type](F.gelu(out))
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

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
        film_gamma=None,
        film_beta=None,
    ) -> Tensor:
        if self.use_RTE and edge_time_k is not None:
            k_j = k_j + edge_time_k
        if self.use_RTE and edge_time_v is not None:
            v_j = v_j + edge_time_v

        # 1) HGT+EW-style value-vector scaling by raw edge weight
        if edge_w is not None:
            v_j = v_j * edge_w.view(-1, 1, 1)

        # 2) TCETF FiLM on top (γ, β from EdgeTrajectoryFiLM). γ, β zero at init
        #    when both etraj + TCETF are freshly enabled, so this block is a
        #    no-op initially and behavior matches HGT+EW baseline until FiLM learns.
        if film_gamma is not None or film_beta is not None:
            assert film_gamma is not None and film_beta is not None, \
                "FiLM γ and β must be supplied together"
            v_j = v_j * (1.0 + film_gamma) + film_beta

        alpha = (q_i * k_j).sum(dim=-1) * rel
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)


class HGTEWTCETF(HGT):
    """HGT with EW value-scaling + TCETF FiLM support.

    `_encode_snapshot` extracts per-edge-type edge_attr (raw holdings weight,
    like HGTEW does) AND forwards edge_attr_traj_dict (from TCETF pipeline
    at MultiTaskEdgePredictor). Both are passed into HGTConvEWFiLM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        new_convs = nn.ModuleList()
        for conv in self.convs:
            ewf_conv = HGTConvEWFiLM(
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
            new_convs.append(ewf_conv)
        self.convs = new_convs
        print(f"[HGT+EW+TCETF] HGTConv → HGTConvEWFiLM substitution complete "
              f"({len(self.convs)} layers); v_j *= edge_w then FiLM on top when TCETF is on.")

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

        # Per-edge-type raw weights (used by v_j *= edge_w)
        edge_attr_dict: Dict = {}
        for etype in edge_index_dict:
            store = data[etype]
            if (hasattr(store, 'edge_attr')
                    and store.edge_attr is not None
                    and store.edge_attr.numel() > 0):
                edge_attr_dict[etype] = store.edge_attr

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(
                x_dict, edge_index_dict, edge_time_dict,
                edge_attr_dict=edge_attr_dict,
                edge_attr_traj_dict=edge_attr_traj_dict,
            )
            if self.drop is not None:
                for k in x_dict:
                    if x_dict[k] is not None:
                        x_dict[k] = self.drop(x_dict[k])

        return x_dict
