"""
DHSpaceMGGlobal: Global meta-graph channels integrated as synthetic relations
that participate in AR/To/n_alpha like primitives. Edges for meta relations are
composed on-the-fly (macro meta-path ops), row-normalized, per-target top-k
pruned, and attention is biased by log(weights). No separate fusion gate — the
global relation machinery mixes meta with primitives.
"""

from typing import Dict, List, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, ones

from .DHSpaceSearch import DHSpace


class DHSpaceMGGlobal(DHSpace):
    def __init__(self, *args, **kwargs):
        args_in = kwargs.get("args", None)
        # Original metadata
        hid_dim = args[0]; metadata = args[1]
        node_types, edge_types = metadata
        # Decide target types to host meta relations (default: predict_type only)
        targets = []
        if args_in is not None and hasattr(args_in, 'predict_type'):
            pt = getattr(args_in, 'predict_type')
            if isinstance(pt, (list, tuple)):
                targets = [t for t in pt if t in node_types]
            elif isinstance(pt, str) and pt in node_types:
                targets = [pt]
        if not targets:
            targets = list(node_types)
        # Build synthetic meta relation per target type
        meta_rels: List[Tuple[str, str, str]] = []
        self.meta_rel_names: Dict[str, str] = {}
        for nt in targets:
            name = f"meta_mg:{nt}"
            meta_rels.append((nt, name, nt))
            self.meta_rel_names[nt] = name

        # Extend metadata
        ext_metadata = (node_types, edge_types + meta_rels)
        new_args = (hid_dim, ext_metadata) + tuple(args[2:])
        super().__init__(*new_args, **kwargs)

        # Config
        self.mg_meta_paths_str = ""
        self.mg_topk = int(getattr(args_in, 'mp_topk', 0)) if args_in is not None else 0
        self.mg_row_norm = str(getattr(args_in, 'mp_row_norm', 'src')) if args_in is not None else 'src'
        self.mg_include_metapath_ops = bool(getattr(args_in, 'mg_include_metapath_ops', 0)) if args_in is not None else 0
        self.mg_meta_paths_str = str(getattr(args_in, 'mg_meta_paths', '')) if args_in is not None else ''
        # Optional temperature for macro gate softmax
        self.mg_macro_tau = float(getattr(args_in, 'mg_macro_tau', 1.0)) if args_in is not None else 1.0

        # Candidate macro ops per target type (optional)
        self._macro_ops_by_target: Dict[str, List[List[str]]] = {}
        if self.mg_include_metapath_ops and self.mg_meta_paths_str.strip():
            for seg in self.mg_meta_paths_str.split(';'):
                hops = [h.strip() for h in seg.split('+') if h.strip()]
                if len(hops) < 2:
                    continue
                try:
                    s0 = next(s for (s, rel, d) in self.edge_types if rel == hops[0])
                    dL = next(d for (s, rel, d) in self.edge_types if rel == hops[-1])
                except StopIteration:
                    continue
                self._macro_ops_by_target.setdefault(dL, []).append(hops)

        # Architecture logits over macro candidates per target (if multiple)
        self.alpha_macro = nn.ParameterDict()
        for nt, cands in self._macro_ops_by_target.items():
            if len(cands) > 1:
                self.alpha_macro[nt] = nn.Parameter(torch.zeros(len(cands)))

        # Per-Δt mg projections (used to bias attention for meta relations)
        if self.rel_time_type == 'independent':
            rel_time_len = self.twin * self.twin
        elif self.rel_time_type == 'relative':
            rel_time_len = self.twin if self.causal_mask else 2 * self.twin
        elif self.rel_time_type in ('source','target'):
            rel_time_len = self.twin
        else:
            rel_time_len = self.twin
        self.mg_relation_att = nn.Parameter(torch.Tensor(rel_time_len, self.n_heads, self.d_k, self.d_k))
        self.mg_relation_msg = nn.Parameter(torch.Tensor(rel_time_len, self.n_heads, self.d_k, self.d_k))
        self.mg_relation_pri = nn.Parameter(torch.ones(rel_time_len, self.n_heads))
        glorot(self.mg_relation_att); glorot(self.mg_relation_msg)

        # Simple cache for composed edges per (t_src, target ntype)
        self._mg_cache: Dict[Tuple[int,str,int], Tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _compose_two(ei1: torch.Tensor, ei2: torch.Tensor) -> torch.Tensor:
        s1 = ei1[0].detach().cpu().numpy(); m1 = ei1[1].detach().cpu().numpy()
        s2 = ei2[0].detach().cpu().numpy(); d2 = ei2[1].detach().cpu().numpy()
        from collections import defaultdict
        idx = defaultdict(list)
        for i in range(len(s2)):
            idx[s2[i]].append(d2[i])
        out_s, out_d = [], []
        for i in range(len(s1)):
            mid = m1[i]
            if mid in idx:
                for dst in idx[mid]:
                    out_s.append(s1[i]); out_d.append(dst)
        if not out_s:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor([out_s, out_d], dtype=torch.long)

    def _compose_metapath_edges(self, graph, hops: List[str]) -> torch.Tensor:
        if not hops or hops[0] not in graph:
            return torch.zeros((2, 0), dtype=torch.long)
        ei = graph[hops[0]].edge_index
        for hop in hops[1:]:
            if hop not in graph:
                return torch.zeros((2, 0), dtype=torch.long)
            ei = self._compose_two(ei, graph[hop].edge_index)
            if ei.numel() == 0:
                break
        return ei

    def _compute_dt(self, t_tar: int, t_src: int) -> int:
        if self.rel_time_type == 'relative':
            return t_src - t_tar + self.twin - 1
        elif self.rel_time_type == 'independent':
            return t_tar * self.twin + t_src
        elif self.rel_time_type == 'source':
            return t_src
        elif self.rel_time_type == 'target':
            return t_tar
        return 0

    def _aggregate_with_bias(self, x_tar, topos):
        # Similar to DHAttnOne2Multi but supports attention log-bias per edge
        res_atts: Dict[str, List[torch.Tensor]] = {}
        res_msgs: Dict[str, List[torch.Tensor]] = {}
        ei_tars: Dict[str, List[torch.Tensor]] = {}
        to_weights: Dict[str, List[torch.Tensor]] = {}
        for item in topos:
            if len(item) == 6:
                (x_tar_rel, x_src_rel, t_tar, t_src, rel, ei_rel_tar) = item
                edge_logw = None
            else:
                (x_tar_rel, x_src_rel, t_tar, t_src, rel, ei_rel_tar, edge_w) = item
                edge_logw = torch.log(edge_w.clamp(min=1e-8)).view(-1,1).expand(-1, self.n_heads)
            _, _, target_type = self.id2etype[self.etype2id[rel]]
            for d in (res_atts, res_msgs, ei_tars, to_weights):
                if target_type not in d:
                    d[target_type] = []
            # Use standard DHAttn for primitives, mg W for meta relations
            if rel.startswith('meta_mg:'):
                # compute attention with mg W
                # Already projected x_src_rel / x_tar_rel embeddings; reuse q/k/v linears
                q_mat = self.q_linears[0](x_tar_rel).view(-1, self.n_heads, self.d_k)
                k_mat = self.k_linears[0](x_src_rel).view(-1, self.n_heads, self.d_k)
                v_mat = self.v_linears[0](x_src_rel).view(-1, self.n_heads, self.d_k)
                dt = self._compute_dt(t_tar, t_src)
                att_w = self.mg_relation_att[dt]
                msg_w = self.mg_relation_msg[dt]
                pri = self.mg_relation_pri[dt]
                k_proj = torch.bmm(k_mat.transpose(1, 0), att_w).transpose(1, 0)
                att = (q_mat * k_proj).sum(dim=-1) * pri / math.sqrt(self.d_k)
            else:
                att, msg = self.DHAttn(x_tar_rel, x_src_rel, t_tar, t_src, rel)
                if edge_logw is not None:
                    att = att + edge_logw.to(att.device)
                res_atts[target_type].append(att); res_msgs[target_type].append(msg); ei_tars[target_type].append(ei_rel_tar)
                if not self.fix_To:
                    to_weights[target_type].append(self.to_alpha[t_tar, t_src, self.etype2id[rel]].expand(att.shape))
                continue
            if edge_logw is not None:
                att = att + edge_logw.to(att.device)
            res = torch.bmm(v_mat.transpose(1, 0), msg_w).transpose(1, 0)
            res_atts[target_type].append(att); res_msgs[target_type].append(res); ei_tars[target_type].append(ei_rel_tar)
            if not self.fix_To:
                to_weights[target_type].append(self.to_alpha[t_tar, t_src, self.etype2id[rel]].expand(att.shape))

        x_dict = {}
        for ntype in self.node_types:
            if ntype in res_atts:
                att = torch.cat(res_atts[ntype], dim=0)
                msg = torch.cat(res_msgs[ntype], dim=0)
                ei_tar = torch.cat(ei_tars[ntype])
                att = softmax(att, ei_tar)
                if not self.fix_To and len(to_weights[ntype])>0:
                    att = att * softmax(torch.cat(to_weights[ntype],dim=0), ei_tar)
                res = msg * att.view(-1, self.n_heads, 1)
                res = res.view(-1, self.hid_dim)
                from torch_scatter import scatter
                res = scatter(res, ei_tar, dim=self.node_dim, dim_size=x_tar[ntype].shape[0], reduce=self.aggr)
                if self.hupdate:
                    res = self.update_lin[ntype](F.gelu(res))
                res = x_tar[ntype] + res
            else:
                res = x_tar[ntype]
            x_dict[ntype] = res
        return x_dict

    def forward(self, xs, graphs):
        twin = self.twin
        device = xs[0][self.id2ntype[0]].device
        ATo = self.A[0].to(device)
        x_res = []
        for t_tar in range(twin):
            ATo_tar = ATo[t_tar]
            x_tar = xs[t_tar]
            if ATo_tar.sum() == 0:
                x_res.append(x_tar); continue
            topos = []
            for t_src in range(twin):
                if ATo_tar[t_src].sum() == 0:
                    continue
                graph_src = graphs[t_src]
                x_src = xs[t_src]
                # primitives
                for rel_idx in ATo_tar[t_src].nonzero():
                    nsrc, rel, ntar = self.id2etype[rel_idx]
                    ei_rel = graph_src[rel].edge_index
                    x_tar_rel = x_tar[ntar].index_select(self.node_dim, ei_rel[1, :])
                    x_src_rel = x_src[nsrc].index_select(self.node_dim, ei_rel[0, :])
                    ei_rel_tar = ei_rel[1, :].T
                    topos.append((x_tar_rel, x_src_rel, t_tar, t_src, rel, ei_rel_tar))
                # meta global per target type
                for nt in self.meta_rel_names:
                    meta_name = self.meta_rel_names[nt]
                    # build (or reuse) composed edges for this snapshot t_src
                    candidates = self._macro_ops_by_target.get(nt, [])
                    if not candidates:
                        continue
                    # soft weights over macros (or 1 if single)
                    if nt in self.alpha_macro:
                        probs = F.softmax(self.alpha_macro[nt] / max(self.mg_macro_tau, 1e-8), dim=-1)
                    else:
                        probs = torch.ones(len(candidates), device=device)
                    # emit entries for each macro candidate, weighted by probs
                    for k_idx, hops in enumerate(candidates):
                        key = (t_src, nt, k_idx)
                        if key in self._mg_cache:
                            ei_meta, w = self._mg_cache[key]
                        else:
                            ei_meta = self._compose_metapath_edges(graph_src, hops)
                            if ei_meta.numel() == 0:
                                continue
                            src = ei_meta[0]; tar = ei_meta[1]
                            if self.mg_row_norm == 'tar':
                                deg = torch.bincount(tar, minlength=int(tar.max().item())+1).clamp_min(1)
                                w = 1.0/deg[tar].float()
                            else:
                                deg = torch.bincount(src, minlength=int(src.max().item())+1).clamp_min(1)
                                w = 1.0/deg[src].float()
                            if self.mg_topk>0:
                                import numpy as np
                                tar_cpu = tar.detach().cpu().numpy(); w_cpu = w.detach().cpu().numpy()
                                keep = torch.zeros(src.size(0), dtype=torch.bool)
                                groups: Dict[int,List[int]] = {}
                                for i,t in enumerate(tar_cpu.tolist()):
                                    groups.setdefault(t, []).append(i)
                                for t, idxs in groups.items():
                                    if len(idxs)<=self.mg_topk:
                                        keep[idxs]=True
                                    else:
                                        vals=np.array([w_cpu[i] for i in idxs])
                                        tk=np.argpartition(-vals, kth=self.mg_topk-1)[:self.mg_topk]
                                        for j in tk:
                                            keep[idxs[j]]=True
                                ei_meta = ei_meta[:, keep]; w = w[keep]
                            self._mg_cache[key]=(ei_meta,w)
                        # scale weights by macro prob (learnable gate over macro candidates)
                        prob = probs[k_idx]
                        w_meta = w * prob
                        ntar = nt; nsrc = nt
                        x_tar_rel = x_tar[ntar].index_select(self.node_dim, ei_meta[1, :])
                        x_src_rel = x_tar[nsrc].index_select(self.node_dim, ei_meta[0, :])
                        ei_rel_tar = ei_meta[1, :].T
                        topos.append((x_tar_rel, x_src_rel, t_tar, t_src, meta_name, ei_rel_tar, w_meta))
            # aggregate with bias
            x_dict = self._aggregate_with_bias(x_tar, topos)
            x_res.append(x_dict)
        return x_res
