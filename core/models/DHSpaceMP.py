from torch import nn
import torch
import torch.nn.functional as F
try:
    from torch_scatter import scatter
except ImportError:
    scatter = None
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, ones

from .DHSpace import DHNet
from .DHSpace import DHSpace


class MetaPathGTN(nn.Module):
    def __init__(self, metadata, in_relations, mid_relations, k_channels=4):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.in_relations = [r for r in self.edge_types if r[1] in in_relations]
        self.mid_relations = [r for r in self.edge_types if r[1] in mid_relations]
        self.k = k_channels
        self.alpha1 = nn.Parameter(torch.zeros(self.k, len(self.in_relations)))
        self.alpha2 = nn.Parameter(torch.zeros(self.k, len(self.mid_relations)))

    def _sparse_from_edge(self, graph, rel):
        src, _, dst = rel
        ei = graph[rel].edge_index
        n_src = graph[src].num_nodes
        n_dst = graph[dst].num_nodes
        # USE edge weights if available: build weighted adjacency for composition
        # Falls back to 1.0 where no edge_attr is present
        values = torch.ones(ei.size(1), device=ei.device)
        if hasattr(graph[rel], 'edge_attr') and graph[rel].edge_attr is not None:
            v = graph[rel].edge_attr
            # allow [E, 1] or [E]
            if v.dim() > 1:
                v = v.squeeze(-1)
            try:
                values = v.to(ei.device)
            except Exception:
                values = values
        return torch.sparse_coo_tensor(ei, values, (n_src, n_dst)).coalesce()

    def _weighted_sum(self, mats, weights):
        out = None
        for m, w in zip(mats, weights):
            if out is None:
                out = w * m
            else:
                out = out + w * m
        return out

    def forward(self, graph):
        # Build per-mid-type groups so matrix shapes match for composition.
        # For in_relations (fund -> mid), group by mid node type (dst).
        mats1 = []
        mats1_mid = []
        for r in self.in_relations:
            m = self._sparse_from_edge(graph, r)
            mats1.append(m)
            mats1_mid.append(r[2])  # dst node type as mid
        # For mid_relations (mid -> fund), group by mid node type (src).
        mats2 = []
        mats2_mid = []
        for r in self.mid_relations:
            m = self._sparse_from_edge(graph, r)
            mats2.append(m)
            mats2_mid.append(r[0])  # src node type as mid

        # Indices per mid type
        mid_types1 = {}
        for idx, mid in enumerate(mats1_mid):
            mid_types1.setdefault(mid, []).append(idx)
        mid_types2 = {}
        for idx, mid in enumerate(mats2_mid):
            mid_types2.setdefault(mid, []).append(idx)

        # Only compose through common mid types present in both steps
        common_mids = set(mid_types1.keys()).intersection(set(mid_types2.keys()))

        w1 = torch.softmax(self.alpha1, dim=-1)  # [k, len(in_rel)]
        w2 = torch.softmax(self.alpha2, dim=-1)  # [k, len(mid_rel)]
        channels = []
        for k in range(self.k):
            comp_sum = None
            for mid in common_mids:
                idxs1 = mid_types1[mid]
                idxs2 = mid_types2[mid]
                mats1_sel = [mats1[i] for i in idxs1]
                mats2_sel = [mats2[i] for i in idxs2]
                w1_sel = w1[k][torch.tensor(idxs1, device=w1.device)]
                w2_sel = w2[k][torch.tensor(idxs2, device=w2.device)]
                s1 = self._weighted_sum(mats1_sel, w1_sel)
                s2 = self._weighted_sum(mats2_sel, w2_sel)
                try:
                    comp_mid = torch.sparse.mm(s1, s2)  # fund->fund
                except Exception:
                    comp_mid = None
                if comp_mid is not None:
                    if comp_sum is None:
                        comp_sum = comp_mid
                    else:
                        comp_sum = comp_sum + comp_mid
            channels.append(comp_sum)
        return channels


class DHSpaceMP(nn.Module):
    def __init__(self, hid_dim, metadata, twin, K_To, K_N, K_R, n_heads=4, norm=True, args=None, mp_channels=4, **kwargs):
        super().__init__()
        self.metadata = metadata
        self.twin = twin
        self.args = args
        # Expose mp_channels from args if provided
        if args is not None and hasattr(args, 'mp_channels'):
            mp_channels = int(getattr(args, 'mp_channels', mp_channels))
        self.mp_channels = mp_channels
        # cap edges per meta-path channel to avoid OOM during gather
        self.mp_topk = int(getattr(args, 'mp_topk', 200000) if args is not None else 200000)
        # Normalization/weighting strategy for composed edges
        self.mp_weight_mode = str(getattr(args, 'mp_weight_mode', 'mul') if args is not None else 'mul').lower()
        self.mp_row_norm = str(getattr(args, 'mp_row_norm', 'src') if args is not None else 'src').lower()
        self.mpgen = MetaPathGTN(metadata,
                                 in_relations={'holds_stock', 'holds_other', 'managed_by'},
                                 mid_relations={'rev_holds_stock', 'rev_holds_other', 'rev_managed_by'},
                                 k_channels=self.mp_channels)
        # Base DHSpace backbone
        # Forward any extra kwargs to DHSpace (e.g., rel_time_type, hupdate)
        self.space = DHSpace(hid_dim, metadata, twin, K_To, K_N, K_R, n_heads=n_heads, norm=norm, args=args, **kwargs)
        # Params for meta-path message passing (single relation template shared across channels)
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.d_k = hid_dim // n_heads
        self.sqrt_dk = self.d_k ** 0.5
        self.q_mp = nn.Linear(hid_dim, hid_dim)
        self.k_mp = nn.Linear(hid_dim, hid_dim)
        self.v_mp = nn.Linear(hid_dim, hid_dim)
        self.mp_pri = nn.Parameter(torch.ones(n_heads))
        self.mp_att = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.mp_msg = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        # Per-channel gates to mitigate redundancy (sigmoid-applied at use)
        self.mp_gates = nn.Parameter(torch.ones(self.mp_channels))
        # Expose dropout from args if present
        self.dropout = nn.Dropout(getattr(args, 'dropout', 0.0) if args is not None else 0.0)

    # Compatibility helpers for DHSearcher/DHNet
    @property
    def A(self):
        return self.space.A

    @A.setter
    def A(self, value):
        self.space.A = value

    def assign_arch(self, A):
        self.space.assign_arch(A)
        return self

    def get_arch(self):
        return self.space.get_arch()

    def count_space(self):
        return self.space.count_space()

    def assign_basic_arch(self, atype):
        # Mirror DHSpace API for search scripts; return self for chaining
        self.space.assign_basic_arch(atype)
        return self

    def set_stage(self, stage):
        # DHSpaceSearch has set_stage; DHSpace may not. Tolerate no-op.
        if hasattr(self.space, 'set_stage'):
            self.space.set_stage(stage)
        return self

    def reset_parameters(self, *args, **kwargs):
        # Accept arbitrary args to be compatible with searcher's selective resets
        self.space.reset_parameters()
        nn.init.zeros_(self.mpgen.alpha1)
        nn.init.zeros_(self.mpgen.alpha2)
        self.q_mp.reset_parameters()
        self.k_mp.reset_parameters()
        self.v_mp.reset_parameters()
        ones(self.mp_pri)
        glorot(self.mp_att)
        glorot(self.mp_msg)

    def forward(self, xs, graphs):
        # First compute baseline DHSpace outputs
        base_out = self.space.forward(xs, graphs)  # list of x_dict per t_tar
        # Optional disable flag to bypass meta-path channels (for stability/ablation)
        if self.args is not None and getattr(self.args, 'mp_disable', 0):
            return base_out

        # Inject meta-path channel messages into 'fund' targets
        twin = self.twin
        device = xs[0][self.space.id2ntype[0]].device
        ATo, _, _ = self.space.A
        ATo = ATo.to(device)

        if scatter is None:
            # torch_scatter optional dependency; if missing, just return baseline
            return base_out

        for t_tar in range(twin):
            ATo_tar = ATo[t_tar]
            if ATo_tar.sum() == 0:
                continue
            x_tar = xs[t_tar]
            # meta-path messages only for 'fund' nodes
            if 'fund' not in x_tar:
                continue
            fund_target = x_tar['fund']
            res_atts = []
            res_msgs = []
            ei_tars = []

            for t_src in range(twin):
                if ATo_tar[t_src].sum() == 0:
                    continue
                graph_src = graphs[t_src]
                # Generate K channels (sparse fund->fund matrices)
                channels = self.mpgen(graph_src)
                for ch_idx, comp in enumerate(channels):
                    if comp is None:
                        continue
                    comp = comp.coalesce()
                    # prune edges to top-k by value to control memory
                    if self.mp_topk > 0 and comp._nnz() > self.mp_topk:
                        vals = comp.values()
                        # guard: if vals is not floating, cast for topk
                        if not torch.is_floating_point(vals):
                            vals = vals.float()
                        k = self.mp_topk
                        topk = torch.topk(vals, k=k, largest=True)
                        idx = comp.indices()[:, topk.indices]
                        val = comp.values()[topk.indices]
                        comp = torch.sparse_coo_tensor(idx, val, comp.size(), device=idx.device).coalesce()
                    ei = comp.indices()  # [2,E]
                    vv = comp.values()   # [E]
                    if ei.numel() == 0:
                        continue
                    # x_src from fund
                    x_src_fund = xs[t_src]['fund']
                    x_tar_rel = fund_target.index_select(0, ei[1, :])
                    x_src_rel = x_src_fund.index_select(0, ei[0, :])
                    # attention/message with meta-path params
                    q_mat = self.q_mp(x_tar_rel).view(-1, self.n_heads, self.d_k)
                    k_mat = self.k_mp(x_src_rel).view(-1, self.n_heads, self.d_k)
                    v_mat = self.v_mp(x_src_rel).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.mp_att).transpose(1, 0)
                    att = (q_mat * k_mat).sum(dim=-1) * self.mp_pri / self.sqrt_dk
                    # Row-normalize weights (by src or tar) if available
                    w_norm = vv
                    try:
                        from torch_scatter import scatter as _scatter
                    except ImportError:
                        _scatter = None
                    if _scatter is not None and vv.numel() > 0:
                        if self.mp_row_norm == 'src':
                            denom = _scatter(vv, ei[0, :], dim=0, dim_size=x_src_fund.size(0), reduce='sum')
                            denom = denom.index_select(0, ei[0, :]).clamp(min=1e-8)
                            w_norm = (vv / denom)
                        elif self.mp_row_norm == 'tar':
                            denom = _scatter(vv, ei[1, :], dim=0, dim_size=fund_target.size(0), reduce='sum')
                            denom = denom.index_select(0, ei[1, :]).clamp(min=1e-8)
                            w_norm = (vv / denom)
                    # Integrate weights: multiply messages or bias attention
                    if self.mp_weight_mode == 'att_log':
                        att = att + torch.log(w_norm.clamp(min=1e-8)).unsqueeze(1)
                    msg = torch.bmm(v_mat.transpose(1, 0), self.mp_msg).transpose(1, 0)
                    if self.mp_weight_mode == 'mul':
                        msg = msg * w_norm.view(-1, 1, 1)
                    res_atts.append(att)
                    res_msgs.append(msg)
                    ei_tars.append(ei[1, :].T)

            if res_atts:
                res_att = torch.cat(res_atts, dim=0)
                res_msg = torch.cat(res_msgs, dim=0)
                ei_tar = torch.cat(ei_tars)
                res_att = softmax(res_att, ei_tar)
                res = res_msg * res_att.view(-1, self.n_heads, 1)
                res = res.view(-1, self.hid_dim)
                res = scatter(res, ei_tar, dim=0, dim_size=fund_target.shape[0], reduce=self.space.aggr)
                # Apply average channel gate (simple global gate across channels)
                if self.mp_gates is not None and self.mp_gates.numel() == self.mp_channels:
                    res = res * torch.sigmoid(self.mp_gates).mean()
                if self.space.hupdate:
                    # use fund update linear if available
                    res = self.space.update_lin['fund'](F.gelu(res))
                res = fund_target + self.dropout(res)
                base_out[t_tar]['fund'] = res

            if self.space.norm:
                base_out[t_tar] = self.space.update_norm(base_out[t_tar])

        return base_out
