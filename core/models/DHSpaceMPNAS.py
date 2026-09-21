from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, ones

from .DHSpace import DHSpace


class MetaPathCandidates:
    """
    Build a small, typed vocabulary of fund->fund meta-path candidates by
    composing length-2 relations through mid node types present in metadata.

    Candidates (if relations exist in the graph):
      - fund -> stock -> fund
      - fund -> other_asset -> fund
      - fund -> manager -> fund
    """

    def __init__(self, metadata):
        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.relset = set(edge_types)
        # Map mid-type to (in_rel, mid_rel)
        self.rules = []
        # stock
        s_in = ("fund", "holds_stock", "stock")
        s_mid = ("stock", "rev_holds_stock", "fund")
        if s_in in self.relset and s_mid in self.relset:
            self.rules.append(("stock", s_in, s_mid))
        # other_asset
        o_in = ("fund", "holds_other", "other_asset")
        o_mid = ("other_asset", "rev_holds_other", "fund")
        if o_in in self.relset and o_mid in self.relset:
            self.rules.append(("other_asset", o_in, o_mid))
        # manager
        m_in = ("fund", "managed_by", "manager")
        m_mid = ("manager", "rev_managed_by", "fund")
        if m_in in self.relset and m_mid in self.relset:
            self.rules.append(("manager", m_in, m_mid))

    @staticmethod
    def _sparse_from_edge(graph, rel):
        ei = graph[rel].edge_index
        src, _, dst = rel
        n_src = graph[src].num_nodes
        n_dst = graph[dst].num_nodes
        # Use weights if present; else ones
        values = torch.ones(ei.size(1), device=ei.device)
        if hasattr(graph[rel], 'edge_attr') and graph[rel].edge_attr is not None:
            v = graph[rel].edge_attr
            if isinstance(v, torch.Tensor):
                if v.dim() > 1:
                    v = v.squeeze(-1)
                values = v.to(ei.device)
        return torch.sparse_coo_tensor(ei, values, (n_src, n_dst)).coalesce()

    def compose_candidates(self, graph):
        """Return list[(name, sparse_coo fund->fund)] for available rules."""
        comps = []
        for name, rel_in, rel_mid in self.rules:
            try:
                a = self._sparse_from_edge(graph, rel_in)      # fund->mid
                b = self._sparse_from_edge(graph, rel_mid)     # mid->fund
                comp = torch.sparse.mm(a, b)                   # fund->fund
            except Exception:
                comp = None
            comps.append((name, comp))
        return comps


class DHSpaceMPNAS(nn.Module):
    """
    DHSpace with NAS-style meta-path selection (length-2 typed paths).

    - Builds a small candidate set of fund->fund meta-path adjacencies per
      snapshot (stock/other_asset/manager), if present in metadata.
    - Learns gating logits over candidates; supports soft or hard top-k (Gumbel).
    - Injects gated meta-path messages (fund targets) as a residual alongside
      the standard DHSpace aggregation.
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
        mp_budget=2,
        hard_topk=False,
        tau=1.0,
    ):
        super().__init__()
        self.metadata = metadata
        self.twin = twin
        self.space = DHSpace(hid_dim, metadata, twin, K_To, K_N, K_R, n_heads=n_heads, norm=norm, args=args)
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.d_k = hid_dim // n_heads
        self.sqrt_dk = self.d_k ** 0.5

        # Meta-path candidate generator and gating
        self.mpgen = MetaPathCandidates(metadata)
        self.num_candidates = max(1, len(self.mpgen.rules))
        self.gate_logits = nn.Parameter(torch.zeros(self.num_candidates))
        self.hard_topk = bool(getattr(args, 'mp_hard_topk', hard_topk) if args is not None else hard_topk)
        self.mp_budget = int(getattr(args, 'mp_budget', mp_budget) if args is not None else mp_budget)
        self.tau = float(getattr(args, 'mp_tau', tau) if args is not None else tau)
        self.mp_topk = int(getattr(args, 'mp_topk', 200000) if args is not None else 200000)

        # Parameters for meta-path message passing (separate from DHSpace)
        self.q_mp = nn.Linear(hid_dim, hid_dim)
        self.k_mp = nn.Linear(hid_dim, hid_dim)
        self.v_mp = nn.Linear(hid_dim, hid_dim)
        self.mp_pri = nn.Parameter(torch.ones(n_heads))
        self.mp_att = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.mp_msg = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.dropout = nn.Dropout(getattr(args, 'dropout', 0.0) if args is not None else 0.0)

        self.reset_parameters()

    # ---------------- Compatibility helpers for DHSearcher/DHNet ----------------
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

    # ---------------------------------------------------------------------------

    def reset_parameters(self):
        self.space.reset_parameters()
        nn.init.zeros_(self.gate_logits)
        self.q_mp.reset_parameters()
        self.k_mp.reset_parameters()
        self.v_mp.reset_parameters()
        ones(self.mp_pri)
        glorot(self.mp_att)
        glorot(self.mp_msg)

    def _gumbel_softmax_topk(self, logits, k, tau=1.0):
        """Straight-through top-k selection using Gumbel-Softmax.
        Returns a (soft) weight vector w summing to ~k, with hard top-k in forward pass.
        """
        if logits.numel() == 1:
            return torch.ones_like(logits)
        g = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-9, max=1-1e-9)))
        y = (logits + g) / max(tau, 1e-6)
        # softmax -> probabilities
        p = torch.softmax(y, dim=-1)
        if k >= p.numel():
            return p
        # hard mask for top-k
        topk_idx = torch.topk(p, k=k, dim=-1).indices
        hard = torch.zeros_like(p)
        hard[topk_idx] = 1.0
        # straight-through
        w = hard + (p - p.detach())
        return w

    def _get_gate_weights(self):
        if self.hard_topk and self.mp_budget > 0:
            return self._gumbel_softmax_topk(self.gate_logits, self.mp_budget, self.tau)
        # soft weights; optionally mask to top-k
        w = torch.softmax(self.gate_logits, dim=-1)
        if self.mp_budget > 0 and self.mp_budget < w.numel():
            topk_idx = torch.topk(w, k=self.mp_budget, dim=-1).indices
            mask = torch.zeros_like(w)
            mask[topk_idx] = 1.0
            w = w * mask
        return w

    def forward(self, xs, graphs):
        # Baseline DHSpace outputs per time
        base_out = self.space.forward(xs, graphs)

        twin = self.twin
        device = xs[0][self.space.id2ntype[0]].device
        ATo, _, _ = self.space.A
        ATo = ATo.to(device)

        # Early exit if no candidates
        if len(self.mpgen.rules) == 0:
            return base_out

        w = self._get_gate_weights()  # size num_candidates

        for t_tar in range(twin):
            ATo_tar = ATo[t_tar]
            if ATo_tar.sum() == 0:
                continue
            # Fund target features at t_tar
            if 'fund' not in xs[t_tar]:
                continue
            x_tar_fund = xs[t_tar]['fund']
            res_total = None

            for t_src in range(twin):
                if ATo_tar[t_src].sum() == 0:
                    continue
                graph_src = graphs[t_src]
                # Build candidate fund->fund adjacencies
                cand = self.mpgen.compose_candidates(graph_src)
                for idx, (_, comp) in enumerate(cand):
                    if comp is None:
                        continue
                    comp = comp.coalesce()
                    # prune edges if too dense
                    if self.mp_topk > 0 and comp._nnz() > self.mp_topk:
                        vals = comp.values()
                        if not torch.is_floating_point(vals):
                            vals = vals.float()
                        k = self.mp_topk
                        topk = torch.topk(vals, k=k, largest=True)
                        ei = comp.indices()[:, topk.indices]
                        vv = comp.values()[topk.indices]
                        comp = torch.sparse_coo_tensor(ei, vv, comp.size(), device=ei.device).coalesce()

                    ei = comp.indices()
                    if ei.numel() == 0:
                        continue
                    # Gather fund features along edges u->v
                    x_src_fund = xs[t_src]['fund']
                    x_tar_rel = x_tar_fund.index_select(0, ei[1, :])
                    x_src_rel = x_src_fund.index_select(0, ei[0, :])
                    q_mat = self.q_mp(x_tar_rel).view(-1, self.n_heads, self.d_k)
                    k_mat = self.k_mp(x_src_rel).view(-1, self.n_heads, self.d_k)
                    v_mat = self.v_mp(x_src_rel).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.mp_att).transpose(1, 0)
                    att = (q_mat * k_mat).sum(dim=-1) * self.mp_pri / self.sqrt_dk
                    msg = torch.bmm(v_mat.transpose(1, 0), self.mp_msg).transpose(1, 0)
                    ei_tar = ei[1, :].T
                    att = softmax(att, ei_tar)
                    res = msg * att.view(-1, self.n_heads, 1)
                    res = res.view(-1, self.hid_dim)
                    # scatter-add to fund targets
                    try:
                        from torch_scatter import scatter
                    except ImportError:
                        scatter = None
                    if scatter is None:
                        continue
                    res = scatter(res, ei_tar, dim=0, dim_size=x_tar_fund.shape[0], reduce=self.space.aggr)
                    # Weight by gate
                    res = res * w[idx].clamp(min=0.0).to(res.device)
                    res_total = res if res_total is None else (res_total + res)

            if res_total is not None:
                if self.space.hupdate:
                    res_total = self.space.update_lin['fund'](F.gelu(res_total))
                base_out[t_tar]['fund'] = xs[t_tar]['fund'] + self.dropout(res_total)
                if self.space.norm:
                    base_out[t_tar] = self.space.update_norm(base_out[t_tar])

        return base_out

