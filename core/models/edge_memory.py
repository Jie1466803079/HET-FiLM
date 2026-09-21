"""Temporal Edge Memory Network components.

EdgeMemoryModule: GRU-based memory tracking per-edge weight trajectories.
PeerAttentionInit: Attention over peer holders' memories for new edge initialization.
compute_weight_profiles: Per-fund weight distribution features from support window.
compute_edge_weight_profiles: Per-edge historical weight features [Abl2].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class EdgeMemoryModule(nn.Module):
    """GRU-based memory for fund-stock edge weight trajectories.

    For each fund-stock edge in the support window, tracks the weight
    evolution via a GRU cell with input [w(t), delta_w(t)] at each snapshot.
    Optionally enriched with node features [Abl4: gru_rich_input].
    """

    def __init__(self, memory_dim=32, rich_input_dim=0):
        super().__init__()
        self.memory_dim = memory_dim
        self.rich_input_dim = rich_input_dim
        gru_input_size = 2 + rich_input_dim
        self.gru_cell = nn.GRUCell(input_size=gru_input_size, hidden_size=memory_dim)
        if rich_input_dim > 0:
            self.node_proj = nn.Linear(rich_input_dim, rich_input_dim)
        self.all_snapshots_holders = False

    def process_support_window(
        self,
        snapshots: List[Dict[str, torch.Tensor]],
        node_feats: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[Tuple[int, int], int], Dict[int, torch.Tensor]]:
        """Process all support snapshots to build edge memories.

        Uses tensor-based registry for O(1) index lookup instead of Python dicts.
        Edge keys are encoded as fund_id * MAX_STOCK + stock_id for hash-free indexing.

        Args:
            snapshots: list of dicts with 'edge_index' and 'edge_weight'.
            node_feats: [Abl4] optional per-snapshot node feature tensors
                        (n_edges_t, rich_input_dim) aligned with edge_index.
        """
        device = snapshots[0]["edge_weight"].device if snapshots else torch.device("cpu")

        all_fids = []
        all_sids = []
        for snap in snapshots:
            ei_cpu = snap["edge_index"].cpu()
            all_fids.append(ei_cpu[0])
            all_sids.append(ei_cpu[1])

        if not all_fids:
            empty = torch.zeros(0, self.memory_dim, device=device)
            return empty, {}, {}

        cat_fids = torch.cat(all_fids)
        cat_sids = torch.cat(all_sids)
        max_stock = int(cat_sids.max().item()) + 1
        edge_keys_all = cat_fids.long() * max_stock + cat_sids.long()
        unique_keys, inverse_all = torch.unique(edge_keys_all, return_inverse=True)
        n_edges = unique_keys.shape[0]

        if n_edges == 0:
            empty = torch.zeros(0, self.memory_dim, device=device)
            return empty, {}, {}

        sorted_keys, sort_perm = unique_keys.sort()
        memory = torch.zeros(n_edges, self.memory_dim, device=device)
        prev_w_tensor = torch.zeros(n_edges)

        offset = 0
        for t_idx, snap in enumerate(snapshots):
            n_t = snap["edge_index"].size(1)
            if n_t == 0:
                continue

            ew = snap["edge_weight"].float().to(device)
            snap_inverse = inverse_all[offset:offset + n_t]
            mem_indices = snap_inverse.to(device)
            ew_cpu = ew.cpu()
            prev_w = prev_w_tensor[snap_inverse.long()]
            dw = ew_cpu - prev_w

            if self.rich_input_dim > 0 and node_feats is not None and t_idx < len(node_feats):
                nf = node_feats[t_idx].to(device)
                nf = self.node_proj(nf)
                gru_input = torch.cat([ew.unsqueeze(-1), dw.to(device).unsqueeze(-1), nf], dim=-1)
            elif self.rich_input_dim > 0:
                pad = torch.zeros(ew.shape[0], self.rich_input_dim, device=device)
                gru_input = torch.cat([ew.unsqueeze(-1), dw.to(device).unsqueeze(-1), pad], dim=-1)
            else:
                gru_input = torch.stack([ew, dw.to(device)], dim=-1)

            current_mem = memory[mem_indices]
            new_mem = self.gru_cell(gru_input, current_mem)
            new_mem = new_mem.to(dtype=memory.dtype)
            memory = memory.clone()
            memory.index_copy_(0, mem_indices, new_mem)
            prev_w_tensor.index_copy_(0, snap_inverse.long(), ew_cpu)
            offset += n_t

        # Build stock_to_holders mapping
        if self.all_snapshots_holders:
            holder_fids = cat_fids.long()
            holder_sids = cat_sids.long()
            holder_keys = holder_fids * max_stock + holder_sids
            unique_holder_keys, holder_inv = torch.unique(holder_keys, return_inverse=True)
            holder_positions = torch.searchsorted(sorted_keys, unique_holder_keys)
            holder_positions = holder_positions.clamp(max=len(sorted_keys) - 1)
            holder_mem_idx_all = sort_perm[holder_positions].to(device)
            unique_holder_sids = (unique_holder_keys % max_stock).long()
            sorted_sids, sid_sort_order = unique_holder_sids.sort()
            sorted_mem_idx = holder_mem_idx_all[sid_sort_order]
            unique_stocks, counts = torch.unique_consecutive(sorted_sids, return_counts=True)
        else:
            last_ei_cpu = snapshots[-1]["edge_index"].cpu()
            last_fids_t = last_ei_cpu[0].long()
            last_sids_t = last_ei_cpu[1].long()
            last_keys = last_fids_t * max_stock + last_sids_t
            positions = torch.searchsorted(sorted_keys, last_keys)
            positions = positions.clamp(max=len(sorted_keys) - 1)
            last_mem_idx = sort_perm[positions].to(device)
            sorted_sids, sid_sort_order = last_sids_t.sort()
            sorted_mem_idx = last_mem_idx[sid_sort_order]
            unique_stocks, counts = torch.unique_consecutive(sorted_sids, return_counts=True)

        stock_to_holders = {}
        split_idx = 0
        for sid, cnt in zip(unique_stocks.tolist(), counts.tolist()):
            holder_mem_idx = sorted_mem_idx[split_idx:split_idx + cnt]
            stock_to_holders[sid] = memory[holder_mem_idx]
            split_idx += cnt

        registry = {}
        return memory, registry, stock_to_holders


class PeerAttentionInit(nn.Module):
    """Initialize edge memory for new edges via attention over peer holders.

    Two-phase design for performance:
      precompute_kv(): called in _build_edge_memory per support window.
        Projects holders' memories through W_k/W_v, pads to fixed MAX_PEERS=50,
        stacks into dense tensors indexed by stock ID.

      forward(): called per batch in decode_weight. ZERO Python loops.
        Q = W_q([z_fund; z_stock]) per edge, then tensor index ops to
        look up precomputed K, V, mask by stock ID, then batched attention.

    v2 extensions (gated by constructor flags, default OFF = backward compat):
      stratified_peers: sample evenly across weight-magnitude quantiles instead
        of random sampling when a stock has >MAX_PEERS holders.
      profile_query: enrich attention query with fund portfolio profile features.
      film_conditioning: output used as scale/shift modulation on GNN prediction
        instead of concatenation into the MLP (handled externally in decode_weight).
    """
    def __init__(self, node_embed_dim, memory_dim, num_heads=4,
                 profile_dim=0, stratified_peers=False, max_peers=50):
        super().__init__()
        self.MAX_PEERS = max_peers
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = memory_dim // num_heads
        assert memory_dim % num_heads == 0
        self.stratified_peers = stratified_peers
        self.profile_dim = profile_dim
        query_dim = node_embed_dim * 2 + profile_dim
        self.W_q = nn.Linear(query_dim, memory_dim)
        self.W_k = nn.Linear(memory_dim, memory_dim)
        self.W_v = nn.Linear(memory_dim, memory_dim)
        self.W_o = nn.Linear(memory_dim, memory_dim)
        self._all_K = None
        self._all_V = None
        self._all_raw = None
        self._all_mask = None
        self._max_sid = 0

    def _select_peers(self, mem: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Select up to MAX_PEERS holders from mem (n_holders, memory_dim).

        When stratified_peers is True, sorts holders by L2 norm of their GRU
        memory (a proxy for weight magnitude since the GRU input is [w, dw])
        and samples evenly across quantiles to ensure the attention sees the
        full spectrum of allocation strategies. Falls back to random when off.
        """
        MP = self.MAX_PEERS
        if mem.shape[0] <= MP:
            return mem
        if not self.stratified_peers:
            idx = torch.randperm(mem.shape[0], device=device)[:MP]
            return mem[idx]
        norms = mem.norm(dim=-1)
        sorted_idx = norms.argsort()
        step = mem.shape[0] / MP
        selected = torch.tensor(
            [int(i * step) for i in range(MP)],
            dtype=torch.long, device=device,
        )
        return mem[sorted_idx[selected]]

    def _pack_holders(self, stock_to_holders: Dict[int, torch.Tensor]):
        """Gather holder memories per stock, cap at MAX_PEERS, build dense tensor + mask."""
        if not stock_to_holders:
            return None, None, 0
        max_sid = max(stock_to_holders.keys()) + 1
        device = next(iter(stock_to_holders.values())).device
        MP = self.MAX_PEERS
        all_raw = torch.zeros(max_sid, MP, self.memory_dim, device=device)
        all_mask = torch.ones(max_sid, MP, dtype=torch.bool, device=device)

        sid_list = []
        mem_list = []
        lengths = []
        for sid, mem in stock_to_holders.items():
            mem = self._select_peers(mem, device)
            sid_list.append(sid)
            mem_list.append(mem)
            lengths.append(mem.shape[0])

        if mem_list:
            cat_mem = torch.cat(mem_list, dim=0)
            row_idx_list = []
            col_idx_list = []
            for sid, n in zip(sid_list, lengths):
                row_idx_list.append(torch.full((n,), sid, dtype=torch.long))
                col_idx_list.append(torch.arange(n, dtype=torch.long))
            row_idx = torch.cat(row_idx_list).to(device)
            col_idx = torch.cat(col_idx_list).to(device)
            all_raw[row_idx, col_idx] = cat_mem.to(dtype=all_raw.dtype)
            all_mask[row_idx, col_idx] = False

        return all_raw, all_mask, max_sid

    def precompute_kv(self, stock_to_holders: Dict[int, torch.Tensor]) -> None:
        """Project and pad K, V for all stocks into dense tensors (frozen W_k/W_v)."""
        self._all_raw = None
        if not stock_to_holders:
            self._all_K = None
            self._all_V = None
            self._all_mask = None
            self._max_sid = 0
            return
        all_raw, all_mask, max_sid = self._pack_holders(stock_to_holders)
        self._all_mask = all_mask
        self._max_sid = max_sid
        valid = ~all_mask
        raw_flat = all_raw[valid]
        self._all_K = torch.zeros_like(all_raw)
        self._all_V = torch.zeros_like(all_raw)
        if raw_flat.shape[0] > 0:
            self._all_K[valid] = self.W_k(raw_flat).to(dtype=self._all_K.dtype)
            self._all_V[valid] = self.W_v(raw_flat).to(dtype=self._all_V.dtype)

    def precompute_raw(self, stock_to_holders: Dict[int, torch.Tensor]) -> None:
        """Cache raw memories WITHOUT applying W_k/W_v (Abl1: trainable projections).

        W_k/W_v are applied live in forward() so they receive gradients.
        """
        self._all_K = None
        self._all_V = None
        if not stock_to_holders:
            self._all_raw = None
            self._all_mask = None
            self._max_sid = 0
            return
        all_raw, all_mask, max_sid = self._pack_holders(stock_to_holders)
        self._all_raw = all_raw
        self._all_mask = all_mask
        self._max_sid = max_sid

    def forward(
        self,
        z_fund: torch.Tensor,
        z_stock: torch.Tensor,
        stock_to_holders: Dict[int, torch.Tensor],
        query_stock_idx: torch.Tensor,
        fund_profiles: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = z_fund.shape[0]
        device = z_fund.device

        has_kv = self._all_K is not None
        has_raw = getattr(self, "_all_raw", None) is not None
        if not has_kv and not has_raw:
            return torch.zeros(B, self.memory_dim, device=device)

        q_parts = [z_fund, z_stock]
        if self.profile_dim > 0 and fund_profiles is not None:
            q_parts.append(fund_profiles)
        q = self.W_q(torch.cat(q_parts, dim=-1))
        sid_clamped = query_stock_idx.clamp(max=self._max_sid - 1)
        mask = self._all_mask[sid_clamped]

        if has_kv:
            k = self._all_K[sid_clamped]
            v = self._all_V[sid_clamped]
        else:
            raw = self._all_raw[sid_clamped]
            k = self.W_k(raw)
            v = self.W_v(raw)

        q_mh = q.view(B, self.num_heads, 1, self.head_dim)
        k_mh = k.view(B, self.num_heads, self.MAX_PEERS, self.head_dim)
        v_mh = v.view(B, self.num_heads, self.MAX_PEERS, self.head_dim)

        attn = q_mh @ k_mh.transpose(-2, -1) / self.head_dim ** 0.5
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(mask_expanded, float("-inf"))
        all_masked = mask.all(dim=1)
        if all_masked.any():
            attn[all_masked] = 0.0
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = (attn @ v_mh).view(B, self.memory_dim)
        out = self.W_o(out)
        if all_masked.any():
            out[all_masked] = 0.0
        return out


def compute_weight_profiles(
    snapshots: List[Dict[str, torch.Tensor]], n_funds: int
) -> torch.Tensor:
    """Per-fund weight distribution features from support window.

    Features (4-d per fund):
        0: herfindahl — sum(w^2), portfolio concentration
        1: n_holdings — log(1 + count of held stocks)
        2: mean_weight — average holding weight
        3: avg_new_weight — mean weight of edges first appearing in the window

    Returns: (n_funds, 4) tensor, z-score normalized across funds.
    """
    profiles = torch.zeros(n_funds, 4)
    seen_edges = set()
    new_edge_weights = {}

    for t, snap in enumerate(snapshots):
        ei = snap["edge_index"]
        ew = snap["edge_weight"].float()
        for e in range(ei.size(1)):
            fid = int(ei[0, e].item())
            sid = int(ei[1, e].item())
            key = (fid, sid)
            if key not in seen_edges:
                seen_edges.add(key)
                if t > 0:
                    if fid not in new_edge_weights:
                        new_edge_weights[fid] = []
                    new_edge_weights[fid].append(float(ew[e].item()))

    last = snapshots[-1]
    ei = last["edge_index"]
    ew = last["edge_weight"].float()
    fund_weights = {}
    for e in range(ei.size(1)):
        fid = int(ei[0, e].item())
        if fid not in fund_weights:
            fund_weights[fid] = []
        fund_weights[fid].append(float(ew[e].item()))

    for fid, ws in fund_weights.items():
        if fid >= n_funds:
            continue
        ws_t = torch.tensor(ws)
        profiles[fid, 0] = (ws_t ** 2).sum()
        profiles[fid, 1] = torch.log1p(torch.tensor(float(len(ws))))
        profiles[fid, 2] = ws_t.mean()

    for fid, ws in new_edge_weights.items():
        if fid >= n_funds:
            continue
        profiles[fid, 3] = sum(ws) / len(ws)

    for col in range(4):
        vals = profiles[:, col]
        nonzero = vals[vals != 0]
        if nonzero.numel() > 1:
            mean = nonzero.mean()
            std = nonzero.std().clamp(min=1e-6)
            profiles[:, col] = (vals - mean) / std

    return profiles


def compute_edge_weight_profiles(
    snapshots: List[Dict[str, torch.Tensor]], n_funds: int, n_stocks: int
) -> torch.Tensor:
    """[Abl2] Per-edge historical weight features from support window.

    For each (fund, stock) pair, computes 4 features from the weight trajectory:
        0: last_weight — most recent observed weight
        1: mean_weight — average weight across snapshots where the edge existed
        2: weight_std — std of weight across snapshots (volatility)
        3: weight_trend — (last - first) / n_snapshots (linear trend)

    Returns: dense (n_funds, n_stocks, 4) tensor. Edges never observed get zeros.
    Normalized per-feature across non-zero entries.
    """
    MAX_T = len(snapshots)
    edge_sums = torch.zeros(n_funds, n_stocks)
    edge_sq_sums = torch.zeros(n_funds, n_stocks)
    edge_counts = torch.zeros(n_funds, n_stocks)
    edge_first = torch.zeros(n_funds, n_stocks)
    edge_last = torch.zeros(n_funds, n_stocks)
    edge_first_set = torch.zeros(n_funds, n_stocks, dtype=torch.bool)

    for snap in snapshots:
        ei = snap["edge_index"]
        ew = snap["edge_weight"].float().cpu()
        fids = ei[0].long().cpu()
        sids = ei[1].long().cpu()
        valid = (fids < n_funds) & (sids < n_stocks)
        fids, sids, ew = fids[valid], sids[valid], ew[valid]

        edge_sums[fids, sids] += ew
        edge_sq_sums[fids, sids] += ew ** 2
        edge_counts[fids, sids] += 1

        not_set = ~edge_first_set[fids, sids]
        if not_set.any():
            edge_first[fids[not_set], sids[not_set]] = ew[not_set]
            edge_first_set[fids[not_set], sids[not_set]] = True
        edge_last[fids, sids] = ew

    has_data = edge_counts > 0
    mean_w = torch.where(has_data, edge_sums / edge_counts.clamp(min=1), torch.zeros_like(edge_sums))
    var_w = torch.where(has_data, edge_sq_sums / edge_counts.clamp(min=1) - mean_w ** 2, torch.zeros_like(edge_sums))
    std_w = var_w.clamp(min=0).sqrt()
    trend_w = torch.where(has_data, (edge_last - edge_first) / max(MAX_T, 1), torch.zeros_like(edge_sums))

    profiles = torch.stack([edge_last, mean_w, std_w, trend_w], dim=-1)
    flat = profiles.view(-1, 4)
    for col in range(4):
        vals = flat[:, col]
        nonzero = vals[vals != 0]
        if nonzero.numel() > 1:
            m = nonzero.mean()
            s = nonzero.std().clamp(min=1e-6)
            flat[:, col] = (vals - m) / s

    return profiles
