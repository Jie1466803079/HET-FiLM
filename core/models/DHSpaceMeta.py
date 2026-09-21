"""
DHSpaceMeta: Extend DHSpace to support learned meta-paths as synthetic edge types
integrated into the same DHGA attention mechanism, with a simple gate to handle
overlap between primitive and meta-path channels.

Key ideas
- Treat each configured meta-path P_k as a synthetic relation id r_meta_k.
- Compute Q/K/V exactly like primitive edges; add separate relation-time
  projections (through the same FR machinery) for synthetic relations by
  enlarging the relation set.
- On each layer step, aggregate primitive and meta-path messages separately and
  combine with a softmax gate so branches compete, preventing double counting.

Notes
- Meta-path neighbor sets are computed on-the-fly per snapshot by composing
  primitive edge indices in the current HeteroData graph.
- This is a straightforward, unoptimized implementation intended for small
  candidate meta-path sets and length-2/3 paths.
"""

from typing import Dict, List, Tuple
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax

from .DHSpaceSearch import DHSpace


class DHSpaceMeta(DHSpace):
    """DHSpace with meta-paths as synthetic relations and branch gating.

    Args:
        meta_paths (List[List[str]]): list of meta-path schemas, each as a list
          of relation-name strings, e.g., ["holds_stock", "rev_holds_stock"].
          Paths must be type-consistent; start/end node type must match.
        meta_prefix (str): string prefix for synthetic relation names.
    """

    def __init__(self, *args, meta_paths: List[List[str]] = None, meta_prefix: str = "meta:", **kwargs):
        # We need to extend metadata (edge_types) with synthetic meta relations
        # before letting DHSpace initialize its tensors. So we rebuild metadata
        # here, then call super().__init__ with the augmented metadata.
        if meta_paths is None:
            meta_paths = []

        # Unpack metadata from kwargs or args
        # args signature is: (hid_dim, metadata, twin, ...)
        assert len(args) >= 2, "DHSpaceMeta expects (hid_dim, metadata, ...)"
        hid_dim = args[0]
        metadata = args[1]
        node_types, edge_types = metadata

        # Keep a copy of primitive relations
        prim_edge_types: List[Tuple[str, str, str]] = list(edge_types)
        prim_rel_names = {et for _, et, _ in prim_edge_types}

        # Build a registry mapping relation name to (src_type, dst_type)
        rel2sd: Dict[str, Tuple[str, str]] = {rel: (s, d) for (s, rel, d) in prim_edge_types}

        # Build synthetic meta relations
        meta_relations: List[Tuple[str, str, str]] = []
        meta_registry: Dict[str, List[str]] = {}
        for idx, hops in enumerate(meta_paths):
            if not hops:
                continue
            # Check type consistency using primitive relation endpoints
            try:
                s0, d0 = rel2sd[hops[0]]
            except KeyError:
                raise KeyError(f"Unknown relation in meta-path[0]: {hops[0]}")
            src_type = s0
            cur_dst = d0
            for hop in hops[1:]:
                if hop not in rel2sd:
                    raise KeyError(f"Unknown relation in meta-path: {hop}")
                s, d = rel2sd[hop]
                if s != cur_dst:
                    raise ValueError(
                        f"Type mismatch in meta-path {hops}: expected {cur_dst} -> *, got {s}"
                    )
                cur_dst = d
            dst_type = cur_dst
            # For our use cases, we typically use start=end target type; we do not enforce here,
            # but the aggregator will naturally target dst_type.
            meta_name = f"{meta_prefix}{idx}"
            meta_relations.append((src_type, meta_name, dst_type))
            meta_registry[meta_name] = list(hops)

        # Augment metadata
        ext_edge_types = prim_edge_types + meta_relations
        ext_metadata = (node_types, ext_edge_types)

        # Rebuild args tuple to inject extended metadata for DHSpace
        new_args = (hid_dim, ext_metadata) + tuple(args[2:])
        super().__init__(*new_args, **kwargs)

        # Record primitive/meta partitions and hop registry
        self.primitive_edge_types = prim_edge_types
        self.primitive_rel_set = {rel for _, rel, _ in prim_edge_types}
        self.meta_relations = meta_relations
        self.meta_rel_set = {rel for _, rel, _ in meta_relations}
        self.meta_registry = meta_registry  # meta_name -> hop list

        # Simple per-layer branch gate (primitive vs meta) as two scalars
        # Gate weights = softmax(eta) over [prim, meta]
        self.branch_gate = nn.Parameter(torch.zeros(2))

        # Meta composition controls (pruning/normalization)
        self.mp_topk = 0
        self.mp_row_norm = "src"
        args_in = kwargs.get("args", None)
        try:
            if args_in is not None:
                self.mp_topk = int(getattr(args_in, "mp_topk", 0))
                self.mp_row_norm = str(getattr(args_in, "mp_row_norm", "src"))
        except Exception:
            pass

    # ---- Meta-path edge composition utilities ----
    @staticmethod
    def _compose_two(ei1: torch.Tensor, ei2: torch.Tensor) -> torch.Tensor:
        """Compose two directed edge_index tensors by joining ei1.dst == ei2.src.
        Returns a new edge_index [2, E_new] for src(ei1)->dst(ei2).
        Simple CPU implementation for small paths.
        """
        s1 = ei1[0].detach().cpu().numpy()
        m1 = ei1[1].detach().cpu().numpy()  # middle
        s2 = ei2[0].detach().cpu().numpy()
        d2 = ei2[1].detach().cpu().numpy()

        from collections import defaultdict

        idx = defaultdict(list)
        for i in range(len(s2)):
            idx[s2[i]].append(d2[i])

        out_s = []
        out_d = []
        for i in range(len(s1)):
            mid = m1[i]
            if mid in idx:
                for dst in idx[mid]:
                    out_s.append(s1[i])
                    out_d.append(dst)
        if len(out_s) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        ei = torch.tensor([out_s, out_d], dtype=torch.long)
        return ei

    def _compose_metapath_edges(self, graph, hops: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compose primitive edges for the given hop sequence within one snapshot graph.
        Returns (edge_index [2,E], edge_weight [E]) where edge_weight reflects
        row-normalization (1/deg) over the chosen row axis.
        """
        assert len(hops) >= 2
        # Start with the first hop edges
        if hops[0] not in graph:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)
        ei = graph[hops[0]].edge_index
        for hop in hops[1:]:
            if hop not in graph:
                return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)
            ei_next = graph[hop].edge_index
            ei = self._compose_two(ei, ei_next)
            if ei.numel() == 0:
                break
        if ei.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)

        # Compute row-normalized weights (default: src-row)
        src = ei[0]
        tar = ei[1]
        if self.mp_row_norm == "tar":
            # weight = 1 / deg_tar
            deg = torch.bincount(tar, minlength=int(tar.max().item()) + 1).clamp_min(1)
            w = 1.0 / deg[tar].float()
        else:
            # weight = 1 / deg_src (default)
            deg = torch.bincount(src, minlength=int(src.max().item()) + 1).clamp_min(1)
            w = 1.0 / deg[src].float()

        # Optional per-target top-k pruning by weight (keep largest weights)
        if isinstance(self.mp_topk, int) and self.mp_topk > 0:
            # group indices per target
            # Move to CPU for simple grouping if needed
            device = ei.device
            tar_cpu = tar.detach().cpu().numpy()
            import numpy as np
            groups = {}
            for idx, t in enumerate(tar_cpu.tolist()):
                groups.setdefault(t, []).append(idx)
            keep_mask = torch.zeros(src.size(0), dtype=torch.bool)
            k = self.mp_topk
            w_cpu = w.detach().cpu().numpy()
            for t, idxs in groups.items():
                if len(idxs) <= k:
                    keep_mask[idxs] = True
                else:
                    # select top-k by weight
                    vals = np.array([w_cpu[i] for i in idxs])
                    topk_idx = np.argpartition(-vals, kth=k-1)[:k]
                    for j in topk_idx:
                        keep_mask[idxs[j]] = True
            if keep_mask.sum().item() < keep_mask.numel():
                ei = ei[:, keep_mask]
                w = w[keep_mask]
                src = ei[0]; tar = ei[1]
        return ei, w

    # ---- Aggregation helpers (messages only; no residual/update) ----
    def _aggregate_messages(self, x_tar, topos):
        res_atts = {}
        res_msgs = {}
        ei_tars = {}
        to_weights = {}
        for item in topos:
            # Support extended tuple with optional edge weights
            if len(item) == 6:
                (x_tar_rel, x_src_rel, t_tar, t_src, rel_name, ei_rel_tar) = item
                edge_logw = None
            else:
                (x_tar_rel, x_src_rel, t_tar, t_src, rel_name, ei_rel_tar, edge_w) = item
                # Add log-weight before softmax for stability
                edge_logw = torch.log(edge_w.clamp(min=1e-8)).view(-1, 1).expand(-1, self.n_heads)
            # Collect dicts per target type
            _, _, target_type = self.id2etype[self.etype2id[rel_name]]
            for collect in [res_atts, res_msgs, ei_tars, to_weights]:
                if target_type not in collect:
                    collect[target_type] = []
            att, msg = self.DHAttn(x_tar_rel, x_src_rel, t_tar, t_src, rel_name)
            if edge_logw is not None:
                try:
                    att = att + edge_logw.to(att.device)
                except Exception:
                    pass
            res_atts[target_type].append(att)
            res_msgs[target_type].append(msg)
            ei_tars[target_type].append(ei_rel_tar)
            if not self.fix_To:
                to_weights[target_type].append(
                    self.to_alpha[t_tar, t_src, self.etype2id[rel_name]].expand(att.shape)
                )

        msg_dict = {}
        for ntype in self.node_types:
            if ntype in res_atts:
                res_att = torch.cat(res_atts[ntype], dim=0)
                res_msg = torch.cat(res_msgs[ntype], dim=0)
                ei_tar = torch.cat(ei_tars[ntype])
                res_att = softmax(res_att, ei_tar)
                if not self.fix_To and len(to_weights[ntype]) > 0:
                    to_weight = torch.cat(to_weights[ntype], dim=0)
                    res_att = res_att.mul(softmax(to_weight, ei_tar))
                res = res_msg * res_att.view(-1, self.n_heads, 1)
                res = res.view(-1, self.hid_dim)
                from torch_scatter import scatter

                res = scatter(
                    res,
                    ei_tar,
                    dim=self.node_dim,
                    dim_size=x_tar[ntype].shape[0],
                    reduce=self.aggr,
                )
            else:
                res = torch.zeros_like(x_tar[ntype])
            msg_dict[ntype] = res
        return msg_dict

    def forward(self, xs, graphs):
        twin = self.twin
        device = xs[0][self.id2ntype[0]].device

        x_win = xs
        x_res = []
        ATo, _, _ = self.A
        ATo = ATo.to(device)

        # Precompute sets of relation indices for primitive vs meta
        prim_ids = []
        meta_ids = []
        for rid, (s, r, d) in enumerate(self.id2etype):
            if r in self.meta_rel_set:
                meta_ids.append(rid)
            else:
                prim_ids.append(rid)

        for t_tar in range(twin):
            ATo_tar = ATo[t_tar]
            x_tar = x_win[t_tar]
            if ATo_tar.sum() == 0:
                # No temporal edges selected; carry forward
                x_out = x_tar
            else:
                topos_prim = []
                topos_meta = []
                for t_src in range(twin):
                    if ATo_tar[t_src].sum() == 0:
                        continue
                    graph_src = graphs[t_src]
                    x_src = x_win[t_src]
                    # Gather selected relation indices at (t_tar, t_src)
                    sel = ATo_tar[t_src].nonzero().flatten().tolist()
                    # Primitive relations
                    for rid in sel:
                        if rid not in prim_ids:
                            continue
                        nsrc, rel_name, ntar = self.id2etype[rid]
                        ei_rel = graph_src[rel_name].edge_index
                        x_tar_rel = x_tar[ntar].index_select(self.node_dim, ei_rel[1, :])
                        x_src_rel = x_src[nsrc].index_select(self.node_dim, ei_rel[0, :])
                        ei_rel_tar = ei_rel[1, :].T
                        topos_prim.append((x_tar_rel, x_src_rel, t_tar, t_src, rel_name, ei_rel_tar))
                    # Meta-path synthetic relations
                    for rid in sel:
                        if rid not in meta_ids:
                            continue
                        nsrc, meta_name, ntar = self.id2etype[rid]
                        hops = self.meta_registry.get(meta_name, [])
                        if not hops:
                            continue
                        ei_meta, w_meta = self._compose_metapath_edges(graph_src, hops)
                        if ei_meta.numel() == 0:
                            continue
                        x_tar_rel = x_tar[ntar].index_select(self.node_dim, ei_meta[1, :])
                        x_src_rel = x_src[nsrc].index_select(self.node_dim, ei_meta[0, :])
                        ei_rel_tar = ei_meta[1, :].T
                        topos_meta.append((x_tar_rel, x_src_rel, t_tar, t_src, meta_name, ei_rel_tar, w_meta))

                # Aggregate per branch (messages only)
                msg_prim = self._aggregate_messages(x_tar, topos_prim)
                msg_meta = self._aggregate_messages(x_tar, topos_meta)

                # Branch gate (primitive vs meta)
                gp = F.softmax(self.branch_gate, dim=0)  # [2]
                x_out = {}
                for ntype in self.node_types:
                    comb_msg = gp[0] * msg_prim[ntype] + gp[1] * msg_meta[ntype]
                    if self.hupdate:
                        comb_msg = self.update_lin[ntype](F.gelu(comb_msg))
                    res = x_tar[ntype] + comb_msg
                    x_out[ntype] = res
                if self.norm:
                    x_out = self.update_norm(x_out)
            x_res.append(x_out)
        return x_res
