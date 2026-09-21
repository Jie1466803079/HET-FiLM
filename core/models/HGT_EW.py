"""HGT with Edge-Weight-scaled value vectors (Option 2: Value Scaling).

New classes:
  HGTConvEW  — subclasses HGTConv, scales v_j by edge weight before attention.
  HGTEW      — subclasses HGT, wires edge_attr_dict through conv layers.

Registered as model name "HGT+EW" in load_model.py.
"""

import math
from typing import Dict, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.utils import softmax

from .HGT import HGT, HGTConv


class HGTConvEW(HGTConv):
    """HGTConv with edge-weight scaling on value vectors.

    For each edge type that carries an ``edge_w`` tensor, the value vector
    ``v_j`` is multiplied element-wise by the scalar weight *before* the
    attention softmax is applied.  Edge types without weights pass through
    unchanged (identical to the base HGTConv behaviour).
    """

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[
            Dict[EdgeType, Tensor], Dict[EdgeType, SparseTensor]
        ],
        edge_time_dict=None,
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
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

            # Extract per-edge weight for this relation (may be None)
            edge_w = None
            if edge_attr_dict is not None and edge_type in edge_attr_dict:
                ew = edge_attr_dict[edge_type]
                if ew is not None and ew.numel() > 0:
                    edge_w = ew.float()
                    if edge_w.dim() > 1:
                        edge_w = edge_w.squeeze(-1)

            edge_type_str = "__".join(edge_type)
            a_rel = self.a_rel[edge_type_str]
            m_rel = self.m_rel[edge_type_str]

            time_emb_k = None
            time_emb_v = None
            if use_rte and edge_time is not None:
                time_emb = self.rte(edge_time)
                time_emb_k = (
                    self.k_lin[src_type](time_emb).view(-1, H, D).transpose(0, 1)
                    @ a_rel
                ).transpose(1, 0)
                time_emb_v = (
                    self.v_lin[src_type](time_emb).view(-1, H, D).transpose(0, 1)
                    @ m_rel
                ).transpose(1, 0)

            k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)
            v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)

            if use_rte and (time_emb_k is not None or time_emb_v is not None):
                out = self.propagate(
                    edge_index,
                    edge_time_k=time_emb_k,
                    edge_time_v=time_emb_v,
                    k=k,
                    q=q_dict[dst_type],
                    v=v,
                    rel=self.p_rel[edge_type_str],
                    edge_w=edge_w,
                    size=None,
                )
            else:
                out = self.propagate(
                    edge_index,
                    k=k,
                    q=q_dict[dst_type],
                    v=v,
                    rel=self.p_rel[edge_type_str],
                    edge_w=edge_w,
                    size=None,
                )
            out_dict[dst_type].append(out)

        # Node-level aggregation + skip connection (unchanged from base)
        from .HGT import group as _group
        import torch.nn.functional as F

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
    ) -> Tensor:
        if self.use_RTE and edge_time_k is not None:
            k_j = k_j + edge_time_k
        if self.use_RTE and edge_time_v is not None:
            v_j = v_j + edge_time_v

        # --- Edge-weight scaling (Option 2) ---
        if edge_w is not None:
            v_j = v_j * edge_w.view(-1, 1, 1)

        alpha = (q_i * k_j).sum(dim=-1) * rel
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)


class HGTEW(HGT):
    """HGT variant that feeds edge_attr_dict into HGTConvEW layers.

    Drop-in replacement for HGT / HGT+ — accepts identical constructor args.
    During ``_encode_snapshot`` it extracts ``edge_attr`` from the data object
    and forwards it to the conv layers so that holding-percentage weights
    modulate the message-passing value vectors.
    """

    def __init__(self, *args, **kwargs):
        # Temporarily prevent HGT.__init__ from creating HGTConv layers
        # by intercepting after super().__init__ and replacing them.
        super().__init__(*args, **kwargs)

        # Replace each HGTConv with HGTConvEW (same params)
        new_convs = torch.nn.ModuleList()
        for conv in self.convs:
            ew_conv = HGTConvEW(
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

    def _encode_snapshot(self, data):
        """Encode a single snapshot, passing edge_attr_dict to EW conv layers."""
        x_dict = self.featemb(data.x_dict)
        edge_index_dict = data.edge_index_dict

        # Build edge_time_dict from graph attributes for RTE
        edge_time_dict = None
        if self.convs[0].use_RTE:
            et = {}
            for etype in edge_index_dict:
                if hasattr(data[etype], 'edge_time'):
                    et[etype] = data[etype].edge_time.squeeze(-1)
            if et:
                edge_time_dict = et

        # Build edge_attr_dict — extract per-edge-type weights
        edge_attr_dict = {}
        for etype in edge_index_dict:
            store = data[etype]
            if hasattr(store, 'edge_attr') and store.edge_attr is not None and store.edge_attr.numel() > 0:
                edge_attr_dict[etype] = store.edge_attr

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(
                x_dict, edge_index_dict, edge_time_dict,
                edge_attr_dict=edge_attr_dict,
            )
            if self.drop is not None:
                for k in x_dict:
                    if x_dict[k] is not None:
                        x_dict[k] = self.drop(x_dict[k])

        return x_dict
