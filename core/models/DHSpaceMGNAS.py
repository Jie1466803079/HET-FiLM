from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, ones

from .DHSpace import DHSpace


class MetaGraphCandidates:
    """
    Build a tiny vocabulary of fund->fund meta-graphs by combining multiple
    length-2 meta-path branches that share the same endpoints (DiffMG-style):

      Branch types (if present in metadata):
        P_s: fund -> holds_stock -> stock -> rev_holds_stock -> fund
        P_o: fund -> holds_other -> other_asset -> rev_holds_other -> fund
        P_m: fund -> managed_by -> manager -> rev_managed_by -> fund

      Meta-graphs (examples):
        G1 = {P_s, P_o}
        G2 = {P_s, P_m}
        G3 = {P_o, P_m}
        G4 = {P_s, P_o, P_m}

    Each meta-graph composes its branch adjacencies and aggregates them with
    learned (softmax-normalized) internal weights (DiffMG).
    """

    def __init__(self, metadata):
        node_types, edge_types = metadata
        self.edge_types = set(edge_types)
        # Availability flags for length-2 branches
        self.has_s = (
            ("fund", "holds_stock", "stock") in self.edge_types
            and ("stock", "rev_holds_stock", "fund") in self.edge_types
        )
        self.has_o = (
            ("fund", "holds_other", "other_asset") in self.edge_types
            and ("other_asset", "rev_holds_other", "fund") in self.edge_types
        )
        self.has_m = (
            ("fund", "managed_by", "manager") in self.edge_types
            and ("manager", "rev_managed_by", "fund") in self.edge_types
        )
        # Define meta-graphs (each as a list of branch relation pairs)
        self.vocab = []
        def add_graph(use_s, use_o, use_m):
            branches = []
            if use_s and self.has_s:
                branches.append((("fund", "holds_stock", "stock"), ("stock", "rev_holds_stock", "fund")))
            if use_o and self.has_o:
                branches.append((("fund", "holds_other", "other_asset"), ("other_asset", "rev_holds_other", "fund")))
            if use_m and self.has_m:
                branches.append((("fund", "managed_by", "manager"), ("manager", "rev_managed_by", "fund")))
            if len(branches) >= 2:
                self.vocab.append(branches)

        add_graph(True, True, False)
        add_graph(True, False, True)
        add_graph(False, True, True)
        add_graph(True, True, True)

    @staticmethod
    def _sparse_from_edge(graph, rel):
        ei = graph[rel].edge_index
        src, _, dst = rel
        n_src = graph[src].num_nodes
        n_dst = graph[dst].num_nodes
        # Use weights if available, else 1.0
        values = torch.ones(ei.size(1), device=ei.device)
        if hasattr(graph[rel], 'edge_attr') and graph[rel].edge_attr is not None:
            v = graph[rel].edge_attr
            if isinstance(v, torch.Tensor) and v.numel() == ei.size(1):
                if v.dim() > 1:
                    v = v.squeeze(-1)
                values = v.to(ei.device)
        return torch.sparse_coo_tensor(ei, values, (n_src, n_dst)).coalesce()

    def compose_branches(self, graph, branches):
        """Compose each length-2 branch and return list of fund->fund sparse matrices."""
        comps = []
        for r1, r2 in branches:
            try:
                a = self._sparse_from_edge(graph, r1)  # fund->mid
                b = self._sparse_from_edge(graph, r2)  # mid->fund
                comp = torch.sparse.mm(a, b)           # fund->fund
            except Exception:
                comp = None
            comps.append(comp)
        return comps


class DHSpaceMGNAS(nn.Module):
    """
    Meta-Graph NAS layer on top of DHSpace. Enumerates a small, typed vocabulary
    of fund->fund meta-graphs (multi-branch length-2 compositions) and learns
    gates over them (DiffMG-style). Messages from composed edges are integrated
    as residuals to DHSpace outputs. Edge weights affect message magnitude only
    (not attention), and row-normalization + top-k pruning stabilize density.
    """

    def __init__(
        self,
        hid_dim,
        metadata,
        twin,
        K_To,
        K_N,
        K_R,
        n_heads=4,
        norm=True,
        args=None,
        mg_budget=2,
        hard_topk=False,
        tau=1.0,
    ):
        super().__init__()
        self.metadata = metadata
        self.twin = twin
        self.args = args

        # Backbone
        self.space = DHSpace(hid_dim, metadata, twin, K_To, K_N, K_R, n_heads=n_heads, norm=norm, args=args)
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.d_k = hid_dim // n_heads
        self.sqrt_dk = self.d_k ** 0.5

        # Meta-graph vocab + gating
        self.mggen = MetaGraphCandidates(metadata)
        self.num_mg = max(1, len(self.mggen.vocab))
        self.gate_logits = nn.Parameter(torch.zeros(self.num_mg))
        # Temperature for softmax gating (DiffMG-style annealing handled by search script)
        self.gate_temp = 1.0
        # DiffMG-style: use differentiable softmax gates (no hard top-k/Gumbel)
        self.mg_budget = int(getattr(args, 'mg_budget', mg_budget) if args is not None else mg_budget)

        # Intra meta-graph branch weights (softmax per meta-graph)
        # We parameterize a weight per branch position up to 3 (pad where absent)
        self.max_branches = 3
        self.branch_logits = nn.Parameter(torch.zeros(self.num_mg, self.max_branches))

        # Stabilizers
        # Per-target top-k sampling and symmetric normalization
        self.mg_topk = int(getattr(args, 'mg_topk', 200000) if args is not None else 200000)

        # Message parameters (separate from DHSpace); DiffMG-style linear aggregation
        self.v_mg = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(getattr(args, 'dropout', 0.0) if args is not None else 0.0)

        self.reset_parameters()
        # Optional target batching (DiffMG-like mini-batch over targets)
        self.target_batch_idx = None

    def set_target_batch(self, idx: "torch.Tensor | None"):
        """Set target batch indices for the last target time (t_tar=twin-1).
        When set, meta-graph edges are filtered to only those with targets in idx.
        """
        self.target_batch_idx = idx

    def reset_parameters(self):
        self.space.reset_parameters()
        nn.init.zeros_(self.gate_logits)
        nn.init.zeros_(self.branch_logits)
        self.v_mg.reset_parameters()

    # ---------------- Compatibility with DHSearcher/DHNet ----------------
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
        self.space.assign_basic_arch(atype)
        return self

    def set_stage(self, stage):
        if hasattr(self.space, 'set_stage'):
            self.space.set_stage(stage)
        return self
    # -------------------------------------------------------------------

    def _gate_weights(self):
        # Softmax gating with temperature (set via search script)
        temp = max(self.gate_temp, 1e-6) if hasattr(self, 'gate_temp') else 1.0
        return torch.softmax(self.gate_logits / temp, dim=-1)
    def _sym_normalize(self, vv, ei, n_src, n_tar):
        # Symmetric normalization w_ij / sqrt(deg_i * deg_j)
        try:
            from torch_scatter import scatter as _scatter
        except ImportError:
            return vv
        deg_src = _scatter(vv, ei[0, :], dim=0, dim_size=n_src, reduce='sum').clamp(min=1e-8)
        deg_tar = _scatter(vv, ei[1, :], dim=0, dim_size=n_tar, reduce='sum').clamp(min=1e-8)
        dsrc = deg_src.index_select(0, ei[0, :])
        dtar = deg_tar.index_select(0, ei[1, :])
        return vv / (dsrc.sqrt() * dtar.sqrt() + 1e-8)

    def _per_target_topk(self, ei, vv, n_tar, k):
        if k <= 0 or vv.numel() <= k:
            return ei, vv
        tar = ei[1, :]
        uniq = torch.unique(tar)
        keep_idx = []
        for t in uniq.tolist():
            mask = (tar == t)
            idx = torch.nonzero(mask, as_tuple=True)[0]
            if idx.numel() <= k:
                keep_idx.append(idx)
            else:
                vals = vv.index_select(0, idx)
                top = torch.topk(vals, k=k, largest=True).indices
                keep_idx.append(idx.index_select(0, top))
        keep_idx = torch.cat(keep_idx) if len(keep_idx) > 0 else torch.arange(vv.numel(), device=vv.device)
        return ei.index_select(1, keep_idx), vv.index_select(0, keep_idx)

    def forward(self, xs, graphs):
        # Baseline DHSpace outputs per time
        base_out = self.space.forward(xs, graphs)

        twin = self.twin
        device = xs[0][self.space.id2ntype[0]].device
        ATo, _, _ = self.space.A
        ATo = ATo.to(device)

        for t_tar in range(twin):
            ATo_tar = ATo[t_tar]
            if ATo_tar.sum() == 0:
                continue
            if 'fund' not in xs[t_tar]:
                continue
            fund_target = xs[t_tar]['fund']
            res_total = None

            for t_src in range(twin):
                if ATo_tar[t_src].sum() == 0:
                    continue
                graph_src = graphs[t_src]
                w_gate = self._gate_weights()
                for mg_idx, branches in enumerate(self.mggen.vocab):
                    comps = self.mggen.compose_branches(graph_src, branches)
                    # Compose branches with softmax weights within meta-graph
                    bs = len(branches)
                    if bs == 0:
                        continue
                    beta = torch.softmax(self.branch_logits[mg_idx, :bs], dim=-1)
                    comp_sum = None
                    for j, comp in enumerate(comps):
                        if comp is None:
                            continue
                        comp = comp.coalesce()
                        if self.mg_topk > 0 and comp._nnz() > self.mg_topk:
                            vals = comp.values()
                            if not torch.is_floating_point(vals):
                                vals = vals.float()
                            topk = torch.topk(vals, k=self.mg_topk, largest=True)
                            idx = comp.indices()[:, topk.indices]
                            val = comp.values()[topk.indices]
                            comp = torch.sparse_coo_tensor(idx, val, comp.size(), device=idx.device).coalesce()
                        comp = torch.sparse_coo_tensor(comp.indices(), beta[j] * comp.values(), comp.size(), device=comp.device).coalesce()
                        comp_sum = comp if comp_sum is None else (comp_sum + comp)
                    if comp_sum is None:
                        continue
                    ei = comp_sum.indices()
                    vv = comp_sum.values()
                    if ei.numel() == 0:
                        continue
                    # If batching targets at last time step, filter edges to batch targets
                    if t_tar == (self.twin - 1) and self.target_batch_idx is not None and self.target_batch_idx.numel() > 0:
                        device = vv.device
                        mask_tar = torch.zeros(fund_target.size(0), dtype=torch.bool, device=device)
                        mask_tar[self.target_batch_idx.to(device)] = True
                        keep = mask_tar.index_select(0, ei[1, :])
                        if keep.sum() == 0:
                            continue
                        ei = ei[:, keep]
                        vv = vv.index_select(0, torch.nonzero(keep, as_tuple=True)[0])
                    # DiffMG normalization + per-target sampling
                    if self.mg_topk > 0:
                        ei, vv = self._per_target_topk(ei, vv, fund_target.size(0), self.mg_topk)
                    vv = self._sym_normalize(vv, ei, xs[t_src]['fund'].size(0), fund_target.size(0))
                    # Features
                    x_src_fund = xs[t_src]['fund']
                    x_src_rel = x_src_fund.index_select(0, ei[0, :])
                    # DiffMG-style linear aggregation: message = (X_src W) scaled by edge weight and gate
                    msg = self.v_mg(x_src_rel) * (vv.view(-1, 1) * w_gate[mg_idx].clamp(min=0.0))
                    # scatter-add to targets
                    from torch_scatter import scatter
                    res = scatter(msg, ei[1, :].T, dim=0, dim_size=fund_target.shape[0], reduce=self.space.aggr)
                    res_total = res if res_total is None else (res_total + res)

            if res_total is not None:
                if self.space.hupdate:
                    res_total = self.space.update_lin['fund'](F.gelu(res_total))
                base_out[t_tar]['fund'] = fund_target + self.dropout(res_total)
                if self.space.norm:
                    base_out[t_tar] = self.space.update_norm(base_out[t_tar])

        return base_out
