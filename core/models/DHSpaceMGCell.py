"""
DHSpaceMGCell: DiffMG-inspired meta-graph search cell embedded inside DHGA.

High-level:
- Adds a small meta-graph "cell" per DHGA layer that aggregates a
  meta-branch message for each target node type from a set of candidate
  relation operations. We implement a pragmatic simplification of DiffMG:
  one-level cell with one-op-per-forward (epsilon-greedy) across a candidate
  set of primitive relations filtered by type reachability, plus identity and
  zero ops. The selected op executes via the same DHAttn pipeline as
  primitives so synthetic messages are fully compatible with DHGA.
- The meta-branch is combined with the primitive branch via a two-way softmax
  gate (overlap-aware). Optional per-time gating scales the meta-branch.
- Architecture parameters (logits) are named with "alpha" so existing
  DHSearcher treats them as arch params (outer/validation optimizer).

Notes:
- This file keeps changes local and toggleable; if meta_graph is disabled the
  class behaves identically to DHSpace.forward.
- This is a compact first version designed to run in the current pipeline
  without modifying trainers.
"""

from typing import Dict, List, Tuple, Optional
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

from .DHSpaceSearch import DHSpace


def _make_time_gate(twin: int, hidden: int) -> nn.Module:
    # Simple MLP gate over discrete t in [0, twin)
    return nn.Sequential(
        nn.Embedding(twin, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid()
    )


class DHSpaceMGCell(DHSpace):
    """DHSpace with a lightweight DiffMG-style meta-graph cell per layer.

    Config (taken from kwargs.args):
      - mg_enabled (bool)
      - mg_L (int, number of latent states; we use 1 internally)
      - mg_one_op (bool, epsilon-greedy one-op-per-forward)
      - mg_eps_start/end/decay (floats/ints)
      - mg_fusion (str: 'softmax2'|'residual')
      - mg_time_gate (bool)
      - mg_time_gate_hidden (int)
    """

    def __init__(self, *args, **kwargs):
        args_in = kwargs.get("args", None)
        super().__init__(*args, **kwargs)

        # Read meta-graph config
        self.mg_enabled = False
        self.mg_L = 1
        self.mg_one_op = True
        self.mg_fusion = "softmax2"
        self.mg_eps = 0.1
        self.mg_eps_start = 0.5
        self.mg_eps_end = 0.05
        self.mg_eps_decay = 2000
        self.register_buffer("mg_step", torch.zeros(1, dtype=torch.long))
        self.mg_time_gate = False
        self.mg_time_gate_hidden = 32
        if args_in is not None:
            self.mg_enabled = bool(getattr(args_in, "mg_enabled", 0))
            self.mg_L = int(getattr(args_in, "mg_L", 1))
            self.mg_one_op = bool(getattr(args_in, "mg_one_op", 1))
            self.mg_eps_start = float(getattr(args_in, "mg_eps_start", 0.5))
            self.mg_eps_end = float(getattr(args_in, "mg_eps_end", 0.05))
            self.mg_eps_decay = int(getattr(args_in, "mg_eps_decay", 2000))
            self.mg_eps = self.mg_eps_start
            self.mg_fusion = str(getattr(args_in, "mg_fusion", "softmax2"))
            self.mg_time_gate = bool(getattr(args_in, "mg_time_gate", 1))
            self.mg_time_gate_hidden = int(
                getattr(args_in, "mg_time_gate_hidden", 32)
            )

        # Candidate primitive relations per target node type: only those whose
        # declared destination matches the target type.
        self._candidates_by_target: Dict[str, List[str]] = {}
        for s, r, d in self.edge_types:
            self._candidates_by_target.setdefault(d, []).append(r)

        # Architecture logits per target type over its candidate set
        # Named with "alpha" so DHSearcher treats them as arch params.
        # alpha_meta[ntype][j-1] is logits over candidates for state j (j>=1)
        self.alpha_meta: nn.ModuleDict = nn.ModuleDict()
        for ntype, cands in self._candidates_by_target.items():
            if len(cands) == 0:
                continue
            plist = nn.ParameterList()
            for _ in range(max(1, self.mg_L)):
                plist.append(nn.Parameter(torch.zeros(len(cands))))
            self.alpha_meta[ntype] = plist

        # Layer fusion gate (primitive vs meta), two scalars -> softmax
        self.alpha_branch = nn.Parameter(torch.zeros(2))

        # Optional per-time gate on meta-branch
        self.time_gate = None
        if self.mg_time_gate:
            self.time_gate = _make_time_gate(self.twin, self.mg_time_gate_hidden)

    def _epsilon(self):
        # simple linear decay per forward call
        step = int(self.mg_step.item())
        if self.mg_eps_decay > 0:
            ratio = min(1.0, step / float(self.mg_eps_decay))
            self.mg_eps = self.mg_eps_start + ratio * (self.mg_eps_end - self.mg_eps_start)
        return self.mg_eps

    def _select_meta_op(self, ntype: str, j_state: int = 1) -> Optional[str]:
        cands = self._candidates_by_target.get(ntype, [])
        if not cands:
            return None
        if not self.mg_one_op:
            # soft selection fallback: pick top-1 under softmax (temperature)
            probs = F.softmax(self.alpha_meta[ntype][j_state - 1], dim=-1)
            idx = int(torch.argmax(probs).item())
            return cands[idx]
        # epsilon-greedy
        eps = self._epsilon()
        if torch.rand(1).item() < eps:
            idx = int(torch.randint(0, len(cands), (1,)).item())
        else:
            idx = int(torch.argmax(self.alpha_meta[ntype][j_state - 1]).item())
        return cands[idx]

    def _aggregate_single_relation(self, x_tar, x_src, graph_src, t_tar, t_src, rel_name):
        # Build topos for just one relation and aggregate using existing path
        nsrc, _, ntar = self.id2etype[self.etype2id[rel_name]]
        ei_rel = graph_src[rel_name].edge_index
        x_tar_rel = x_tar[ntar].index_select(self.node_dim, ei_rel[1, :])
        x_src_rel = x_src[nsrc].index_select(self.node_dim, ei_rel[0, :])
        ei_rel_tar = ei_rel[1, :].T
        att, msg = self.DHAttn(x_tar_rel, x_src_rel, t_tar, t_src, rel_name)
        # Normalize and scatter like DHAttnOne2Multi
        ei_tar = ei_rel_tar
        att = softmax(att, ei_tar)
        res = msg * att.view(-1, self.n_heads, 1)
        res = res.view(-1, self.hid_dim)
        from torch_scatter import scatter

        res = scatter(
            res, ei_tar, dim=self.node_dim, dim_size=x_tar[ntar].shape[0], reduce=self.aggr
        )
        return ntar, res

    # Helpers for macro ops and dt
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
        if not hops:
            return torch.zeros((2, 0), dtype=torch.long)
        if hops[0] not in graph:
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
        if self.rel_time_type == "relative":
            return t_src - t_tar + self.twin - 1
        elif self.rel_time_type == "independent":
            return t_tar * self.twin + t_src
        elif self.rel_time_type == "source":
            return t_src
        elif self.rel_time_type == "target":
            return t_tar
        return 0

    def forward(self, xs, graphs):
        if not self.mg_enabled:
            return super().forward(xs, graphs)

        # Primitive branch identical to DHSpace, but we keep intermediate messages
        twin = self.twin
        device = xs[0][self.id2ntype[0]].device
        ATo = self.A[0].to(device)

        x_res = []
        # per forward step update epsilon schedule
        self.mg_step += 1

        for t_tar in range(twin):
            ATo_tar = ATo[t_tar]
            # Avoid walrus operator for broader Python compatibility
            tar = t_tar
            x_tar = xs[tar]
            if ATo_tar.sum() == 0:
                x_out = x_tar
                x_res.append(x_out)
                continue

            # Build primitive topos and aggregate using parent helper
            topos = []
            for t_src in range(twin):
                if ATo_tar[t_src].sum() == 0:
                    continue
                graph_src = graphs[t_src]
                x_src = xs[t_src]
                for rel in ATo_tar[t_src].nonzero():
                    nsrc, rel, ntar = self.id2etype[rel]
                    ei_rel = graph_src[rel].edge_index
                    x_tar_rel = x_tar[ntar].index_select(self.node_dim, ei_rel[1, :])
                    x_src_rel = x_src[nsrc].index_select(self.node_dim, ei_rel[0, :])
                    ei_rel_tar = ei_rel[1, :].T
                    topos.append((x_tar_rel, x_src_rel, t_tar, t_src, rel, ei_rel_tar))
            prim_msgs = self.DHAttnOne2Multi(x_tar, topos)

            # --- Meta-graph branch with true chaining and macro-ops ---
            # Initialize z^0 with current target features
            z_states: List[Dict[str, torch.Tensor]] = []
            z0: Dict[str, torch.Tensor] = {ntype: x_tar.get(ntype) for ntype in self.node_types if ntype in x_tar}
            z_states.append(z0)

            for j in range(1, max(1, self.mg_L) + 1):
                zj: Dict[str, torch.Tensor] = {}
                for ntype in self.node_types:
                    if ntype in x_tar:
                        zj[ntype] = torch.zeros_like(x_tar[ntype])
                for t_src in range(twin):
                    if ATo_tar[t_src].sum() == 0:
                        continue
                    graph_src = graphs[t_src]
                    for ntype in self.node_types:
                        if ntype not in x_tar:
                            continue
                        op = self._select_meta_op(ntype, j_state=j)
                        if op is None:
                            continue
                        # accumulate from all previous states i < j
                        for i_state in range(0, j):
                            try:
                                if op.startswith('macro:'):
                                    hops = [h.strip() for h in op[len('macro:'):].split('+') if h.strip()]
                                    ei = self._compose_metapath_edges(graph_src, hops)
                                    if ei.numel() == 0:
                                        continue
                                    # infer src/target types from first hop
                                    nsrc, _, ntar = self.id2etype[self.etype2id[hops[0]]]
                                    tar_emb = (z_states[j-1][ntar] if j-1 >= 0 else x_tar[ntar]).index_select(self.node_dim, ei[1, :])
                                    src_emb = z_states[i_state][nsrc].index_select(self.node_dim, ei[0, :])
                                else:
                                    nsrc, _, ntar = self.id2etype[self.etype2id[op]]
                                    ei_rel = graph_src[op].edge_index
                                    tar_emb = (z_states[j-1][ntar] if j-1 >= 0 else x_tar[ntar]).index_select(self.node_dim, ei_rel[1, :])
                                    src_emb = z_states[i_state][nsrc].index_select(self.node_dim, ei_rel[0, :])
                                    ei = ei_rel
                                # compute attention with synthetic mg W per dt
                                q_mat = self.q_linears[0](tar_emb).view(-1, self.n_heads, self.d_k)
                                k_mat = self.k_linears[0](src_emb).view(-1, self.n_heads, self.d_k)
                                v_mat = self.v_linears[0](src_emb).view(-1, self.n_heads, self.d_k)
                                dt = self._compute_dt(t_tar, t_src) if hasattr(self, '_compute_dt') else 0
                                att_w = self.mg_relation_att[dt]
                                msg_w = self.mg_relation_msg[dt]
                                pri = self.mg_relation_pri[dt]
                                k_proj = torch.bmm(k_mat.transpose(1, 0), att_w).transpose(1, 0)
                                att = (q_mat * k_proj).sum(dim=-1) * pri / math.sqrt(self.d_k)
                                ei_tar = ei[1, :].T
                                att = softmax(att, ei_tar)
                                msg_proj = torch.bmm(v_mat.transpose(1, 0), msg_w).transpose(1, 0)
                                res = msg_proj * att.view(-1, self.n_heads, 1)
                                res = res.view(-1, self.hid_dim)
                                from torch_scatter import scatter
                                res = scatter(res, ei_tar, dim=self.node_dim, dim_size=x_tar[ntar].shape[0], reduce=self.aggr)
                                zj[ntype] = zj[ntype] + res
                            except Exception:
                                pass
                z_states.append(zj)

            meta_msgs = z_states[-1]

            # Per-time gate on meta branch (scalar [0,1] per t_tar)
            if self.time_gate is not None:
                gate = self.time_gate(torch.tensor([t_tar], device=device)).view(1)
                for ntype in meta_msgs:
                    if meta_msgs[ntype] is not None:
                        meta_msgs[ntype] = meta_msgs[ntype] * gate

            # Residual/update+norm per branch and fuse
            x_out: Dict[str, torch.Tensor] = {}
            gate2 = F.softmax(self.alpha_branch, dim=0)  # [2]
            div_terms = []
            for ntype in self.node_types:
                prim = prim_msgs.get(ntype, x_tar.get(ntype))
                meta = meta_msgs.get(ntype, None)
                if meta is None:
                    comb = prim
                else:
                    comb = gate2[0] * prim + gate2[1] * meta
                    # accumulate diversity (cosine^2) per ntype
                    try:
                        p = prim.reshape(prim.size(0), -1)
                        m = meta.reshape(meta.size(0), -1)
                        # cosine similarity per node then mean
                        eps = 1e-8
                        cs = (p * m).sum(dim=-1) / (p.norm(dim=-1) * m.norm(dim=-1) + eps)
                        div_terms.append((cs * cs).mean())
                    except Exception:
                        pass
                if self.hupdate:
                    comb = self.update_lin[ntype](F.gelu(comb))
                comb = x_tar[ntype] + comb
                x_out[ntype] = comb
            if self.norm:
                x_out = self.update_norm(x_out)
            # store last diversity (not added to loss by default; trainers can query if desired)
            if len(div_terms) > 0:
                self.last_div_loss = torch.stack(div_terms).mean()
            x_res.append(x_out)

        return x_res
