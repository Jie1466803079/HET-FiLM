"""
CasMLN (Cascaded Multi-Level Network) adapted for the Funds framework.

Ported from CasMLN/cmln/model/CMLN.py (SIGIR 2024).
Key adaptation: encode() accepts a list of HeteroData snapshots
(matching HGT/HTGNN interface) instead of a dict indexed by integer.

Architecture:
  - RGCNConv layers on homogeneous-converted graphs
  - Multi-level feature extraction: node, category (degree-weighted), graph (learnable)
  - LLM modulation: log(|MLP(llm_emb)|) element-wise scaling on category/graph levels
  - Temporal attention across snapshots
  - Degree-based amplification via configurable amplifier base
"""
import os
import torch
from torch import nn
from torch_geometric.nn import RGCNConv, Linear
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
from torch_scatter import scatter_add


def make_hodata(x_dict, e_dict, predict_type):
    """Convert heterogeneous node/edge dicts to homogeneous format for RGCNConv.

    Ported from CasMLN/cmln/data/utils.py:449-471.

    Returns:
        x: [N_total, feat_dim] concatenated node features
        e: [2, E_total] concatenated edge indices (remapped to global IDs)
        predict_mask: boolean mask(s) for target node type(s)
        cate_mask: list of boolean masks, one per node type
        hodata: the homogeneous Data object (carries edge_type)
    """
    hodata = HeteroData()
    for ntype, x in x_dict.items():
        hodata[ntype].x = x
    for etype, e in e_dict.items():
        hodata[etype].edge_index = e
    hodata = hodata.to_homogeneous()
    node_type = hodata.node_type
    node_type_names = hodata._node_type_names
    name2id = dict(zip(node_type_names, range(len(node_type_names))))
    if isinstance(predict_type, (list, tuple)):
        predict_mask = [node_type == name2id[pt] for pt in predict_type]
    else:
        predict_mask = node_type == name2id[predict_type]
    x = hodata.x
    e = hodata.edge_index
    cate_mask = [node_type == name2id[cate] for cate in x_dict.keys()]
    return x, e, predict_mask, cate_mask, hodata


class CateLevelExtraction(nn.Module):
    """Ported from CasMLN/cmln/model/CMLN.py:cate_level_extraction."""

    def __init__(self, hid_dim, metadata):
        super().__init__()
        self.cate_num = len(metadata[0])

    def get_cate_emb(self, x, degrees):
        """Degree-weighted mean embedding per node type."""
        cate_emb = []
        for i in range(self.cate_num):
            cate_i_emb = torch.mean(
                x[i] * torch.unsqueeze(degrees[i], -1), dim=0
            )
            cate_emb.append(cate_i_emb)
        return cate_emb


class GraphLevelExtraction(nn.Module):
    """Ported from CasMLN/cmln/model/CMLN.py:graph_level_extraction."""

    def __init__(self, hid_dim, metadata):
        super().__init__()
        self.cate_num = len(metadata[0])
        self.graph_cate_w = nn.Sequential(
            nn.Linear(hid_dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-5, elementwise_affine=True)

    def get_graph_emb(self, cate_emb):
        cate_w = self.graph_cate_w(torch.stack(cate_emb))
        weighted_cate_emb = torch.stack(cate_emb) * cate_w
        graph_emb = torch.mean(weighted_cate_emb, dim=0)
        graph_emb = self.layer_norm(graph_emb)
        return graph_emb, cate_w


class TemporalAttention(nn.Module):
    """Ported from CasMLN/cmln/model/CMLN.py:temporal_attention.

    The original has a `if dataset!='covid': x = self.norm(x)` branch.
    Since our dataset is Funds (not covid), we always apply LayerNorm.
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.alpha = nn.Sequential(
            nn.Linear(hid_dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hid_dim, eps=1e-5, elementwise_affine=True)

    def temporal_encode(self, feats):
        temporal_w = self.alpha(torch.stack(feats))
        feats = torch.stack(feats) * torch.softmax(temporal_w, dim=0)
        x = torch.mean(feats, dim=0)
        x = self.norm(x)
        return x


class EdgeWeightRGCNConv(RGCNConv):
    """RGCNConv with per-edge weight support for scaling messages.

    When edge_weight is None, delegates to parent RGCNConv (identical behavior).
    When edge_weight is provided, scales source node messages by per-edge
    weights before aggregation and relation-specific transform.

    Math: out[v] = sum_r sum_{u in N_r(v)} w(u,v) * x_u @ W_r  (+ root + bias)
    Standard RGCNConv uses w(u,v)=1 for all edges.

    Optional env var EDGE_WEIGHT_LOG=1 wraps the incoming weight in
    ``torch.log1p(clamp(w, min=0))`` before scaling — the CMLN analog of
    ``HGT+EWlogPlus``'s log1p transform.  Default (env unset or "0") is
    unchanged raw-weight behavior.

    Only the fallback loop-over-relations path is overridden (pyg_lib path
    is not available in the dhgas environment).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_edge_weight = os.environ.get("EDGE_WEIGHT_LOG", "0") != "0"

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        if edge_weight is None:
            return super().forward(x, edge_index, edge_type)

        # --- Edge-weight-aware path (fallback loop over relations) ---
        x_l = x[0] if isinstance(x, tuple) else x
        x_r = x[1] if isinstance(x, tuple) else x_l
        size = (x_l.size(0), x_r.size(0))

        assert edge_type is not None

        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:
            # Block-diagonal decomposition — fall back to parent (no ew support)
            return super().forward(x, edge_index, edge_type)

        for i in range(self.num_relations):
            mask = edge_type == i
            tmp = edge_index[:, mask]
            tmp_w = edge_weight[mask]
            h = self.propagate(tmp, x=x_l, edge_weight=tmp_w,
                               edge_type_ptr=None, size=size)
            out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            out = out + x_r @ root

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_weight=None, edge_type_ptr=None):
        if edge_weight is not None:
            if self._log_edge_weight:
                edge_weight = torch.log1p(torch.clamp(edge_weight, min=0.0))
            x_j = x_j * edge_weight.unsqueeze(-1)
        return x_j


class CMLN(nn.Module):
    """CasMLN adapted for the Funds framework.

    Ported from CasMLN/cmln/model/CMLN.py.
    Only adaptation: encode() accepts a list of HeteroData snapshots
    instead of a dict indexed by integer, matching HGT/HTGNN interface.
    All architecture components are identical to the original.

    Args:
        in_dim: Input feature dimension (after FeatEmbed projection).
        hid_dim: Hidden dimension for RGCNConv and MLPs.
        num_layers: Number of RGCNConv layers.
        dropout: Dropout rate (stored, not used in forward — same as original).
        time_window: Number of temporal snapshots.
        metadata: (node_types, edge_types) tuple from dataset.
        predict_type: Target node type(s) for link prediction, e.g. ['fund', 'stock'].
        device: torch device.
        featemb: Optional FeatEmbed module for learnable node embeddings.
        amplifier: Degree amplification base (default 5.0).
        llm_graph_emb: Pre-computed graph-level LLM embedding [1536].
        llm_cate_embs: Pre-computed category-level LLM embeddings (list of [1536]).
        init_ratio: [node_w, cate_w, graph_w] initial values (default [0.5, 0.25, 0.25]).
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_layers,
        dropout,
        time_window,
        metadata,
        predict_type,
        device,
        featemb=None,
        amplifier=5.0,
        llm_graph_emb=None,
        llm_cate_embs=None,
        init_ratio=None,
        prospectus_loader=None,  # NEW: enables node-level text modulation
    ):
        super().__init__()
        if init_ratio is None:
            init_ratio = [0.5, 0.25, 0.25]

        num_relations = len(metadata[1])
        self.num_cates = len(metadata[0])
        self.cates = metadata[0]
        self.timeframe = list(range(time_window))
        self.predict_type = predict_type
        self.featemb = featemb if featemb else lambda x: x
        self.device = device
        self.dropout = dropout

        # Whether to use edge-weight-aware degree amplification.
        # Controlled by EDGE_WEIGHT_DEGREE env var (default "0" = disabled).
        self._has_edge_weights = True

        # Whether to use edge-weight-scaled message passing (Strategy 1).
        # Controlled by EDGE_WEIGHT_MESSAGE env var (default "0" = disabled).
        self._use_ew_message = os.environ.get("EDGE_WEIGHT_MESSAGE", "0") != "0"

        # Per-type linear projection to in_dim (needed because different node types
        # have different raw feature dimensions; to_homogeneous requires uniform dim).
        # Matches the commented-out self.hlinear in original CasMLN.
        self.adapt_ws = nn.ModuleDict()
        for nt in metadata[0]:
            self.adapt_ws[nt] = Linear(-1, in_dim)

        # RGCNConv layers (EdgeWeightRGCNConv when EDGE_WEIGHT_MESSAGE=1)
        ConvClass = EdgeWeightRGCNConv if self._use_ew_message else RGCNConv
        convs = nn.ModuleList()
        convs.append(ConvClass(in_dim, hid_dim, num_relations))
        for _ in range(num_layers - 1):
            convs.append(ConvClass(hid_dim, hid_dim, num_relations))
        self.convs = convs

        # Multi-level weights (learnable) — original uses dataset-specific init_ratios,
        # we use a configurable default [0.5, 0.25, 0.25] for Funds.
        self.node_w = nn.Parameter(torch.FloatTensor([init_ratio[0]]))
        self.cate_w = nn.Parameter(torch.FloatTensor([init_ratio[1]]))
        self.graph_w = nn.Parameter(torch.FloatTensor([init_ratio[2]]))

        # Multi-level extraction modules
        self.cate_emb_module = CateLevelExtraction(hid_dim, metadata)
        self.graph_emb_module = GraphLevelExtraction(hid_dim, metadata)
        self.temporal_attn = TemporalAttention(hid_dim)

        # Degree amplification
        self.amplifier = amplifier

        # LLM embeddings (pre-computed, frozen)
        # Original stores as plain attributes; we use register_buffer for proper
        # device tracking when .to(device) is called.
        self.register_buffer("llm_graph_emb", llm_graph_emb)
        self._register_cate_embs(llm_cate_embs)

        # LLM projection MLPs (1536 -> hid_dim)
        self.graph_mlp = nn.Sequential(
            nn.Linear(1536, hid_dim * 2, bias=True),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim, bias=True),
        )
        self.cate_mlp = nn.Sequential(
            nn.Linear(1536, hid_dim * 2, bias=True),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, hid_dim, bias=True),
        )

        # NODE-LEVEL PROSPECTUS MODULATION (paper-style MLP_S / MLP_R at Level 1).
        # When prospectus_loader is None, these stay None and modulation is skipped.
        self.prospectus_loader = prospectus_loader
        self.strategy_mlp = None
        self.risk_mlp = None
        self._dataset_keys = None  # set externally by run_model.py before training
        if prospectus_loader is not None:
            # Mirror MLP_G / MLP_C exactly: Linear → ReLU → Linear, same shapes,
            # only the input dim differs (1024 for text-embedding-3-large vs 1536 for ada-002).
            self.strategy_mlp = nn.Sequential(
                nn.Linear(1024, hid_dim * 2, bias=True),
                nn.ReLU(),
                nn.Linear(hid_dim * 2, hid_dim, bias=True),
            )
            self.risk_mlp = nn.Sequential(
                nn.Linear(1024, hid_dim * 2, bias=True),
                nn.ReLU(),
                nn.Linear(hid_dim * 2, hid_dim, bias=True),
            )
            # Diagnostic counters (printed once per epoch via log_prospectus_diagnostics)
            self._cold_n_total = 0
            self._cold_n_valid = 0
            self._mod_strat_sum = 0.0
            self._mod_risk_sum = 0.0
            self._mod_n_obs = 0

            # COLD-START TOKEN FOLLOW-UP (opt-in via --use_prospectus_node_cold_token).
            # Two learnable 1024-d "no text" embeddings substituted for cold-start funds
            # before the MLP. Initialized to zeros so that at init, log|MLP(0)| = log|bias|
            # = some non-trivial constant — the model learns the proper cold-start prior.
            self.strategy_cold = nn.Parameter(torch.zeros(1, 1024))
            self.risk_cold = nn.Parameter(torch.zeros(1, 1024))
            # Whether to substitute learned cold tokens for delta_t==-1 funds.
            # Set externally by run_model.py based on --use_prospectus_node_cold_token flag.
            self._use_cold_token = False

    def _register_cate_embs(self, emb_list):
        """Register a list of tensors as numbered buffers."""
        if emb_list is None:
            self._n_cate_embs = 0
            return
        self._n_cate_embs = len(emb_list)
        for i, emb in enumerate(emb_list):
            self.register_buffer(f"llm_cate_emb_{i}", emb)

    def _get_llm_cate_embs(self):
        return [getattr(self, f"llm_cate_emb_{i}") for i in range(self._n_cate_embs)]

    @staticmethod
    def _extract_ew_edges(graph, hodata):
        """Extract edges with ``edge_attr`` from *graph* and remap their indices
        to the homogeneous node IDs produced by ``make_hodata`` / ``to_homogeneous()``.

        This mirrors the way :class:`SEHTGNNFundsWrapper._build_dgl_graph` attaches
        ``_ew`` edge data from PyG ``edge_attr`` so that the downstream
        :class:`EdgeWeightGCN` can scale messages by fund-stock holding weights.

        Returns
        -------
        ew_edge_index : Tensor [2, E_w] or *None*
        ew_weights    : Tensor [E_w]    or *None*
        """
        node_type_names = hodata._node_type_names
        # Cumulative node offset per type (to_homogeneous concatenates in this order)
        offsets = {}
        cum = 0
        for idx, name in enumerate(node_type_names):
            offsets[name] = cum
            cum += int((hodata.node_type == idx).sum())

        ew_edges = []
        ew_weights = []
        for etype in graph.edge_types:
            store = graph[etype]
            if (hasattr(store, "edge_attr")
                    and store.edge_attr is not None
                    and store.edge_attr.numel() > 0):
                src_type, _rel, dst_type = etype
                ei = store.edge_index
                src_global = ei[0] + offsets[src_type]
                dst_global = ei[1] + offsets[dst_type]
                ew_edges.append(torch.stack([src_global, dst_global], dim=0))
                ew = store.edge_attr.float()
                if ew.dim() > 1:
                    ew = ew.squeeze(-1)
                ew_weights.append(ew)

        if ew_edges:
            return torch.cat(ew_edges, dim=1), torch.cat(ew_weights, dim=0)
        return None, None

    @staticmethod
    def _build_edge_weight(graph, e_dict, device):
        """Build per-edge weight tensor for the homogeneous graph.

        Edges with edge_attr get their actual weight; others get 1.0.
        Edge order matches to_homogeneous() concatenation order (same
        as e_dict iteration order in Python 3.8+).
        """
        weights = []
        for etype_key in e_dict.keys():
            n_edges = e_dict[etype_key].shape[1]
            store = graph[etype_key]
            if (hasattr(store, 'edge_attr') and store.edge_attr is not None
                    and store.edge_attr.numel() > 0):
                ew = store.edge_attr.float()
                if ew.dim() > 1:
                    ew = ew.squeeze(-1)
                weights.append(ew.to(device))
            else:
                weights.append(torch.ones(n_edges, device=device))
        return torch.cat(weights)

    def encode(self, data, *args, **kwargs):
        """Encode temporal snapshots -> node embeddings.

        Args:
            data: list of HeteroData snapshots (length == time_window).
                  Adapted from original which uses dict[int] -> HeteroData.

        Returns:
            [z_fund, z_stock] list of node embeddings for link prediction,
            or z tensor for single predict_type.
        """
        if isinstance(data, (list, tuple)):
            snapshots = list(data)
        else:
            snapshots = [data]

        feats = []

        # Project LLM embeddings (same as original lines 171-179)
        llm_graph_emb = self.graph_mlp(self.llm_graph_emb)
        llm_cate_embs_raw = self._get_llm_cate_embs()
        llm_cate_embs = [self.cate_mlp(e) for e in llm_cate_embs_raw]

        for ttype in self.timeframe:
            graph = snapshots[ttype]

            x_dict = self.featemb(graph.x_dict)
            # Project each node type to in_dim so to_homogeneous() can concatenate
            x_dict = {nt: self.adapt_ws[nt](x) for nt, x in x_dict.items()}
            e_dict = graph.edge_index_dict

            x, e, predict_mask, cate_mask, hodata = make_hodata(
                x_dict, e_dict, self.predict_type
            )

            # Degree-based amplification with edge-weight awareness.
            # For edges with edge_attr (fund-stock holdings), each edge
            # contributes its weight to the node's degree instead of 1.0.
            # Edges without edge_attr contribute 1.0 (standard count).
            # Controlled by EDGE_WEIGHT_DEGREE env var (default "0" = disabled).
            num_nodes = x.shape[0]
            if (os.environ.get("EDGE_WEIGHT_DEGREE", "0") != "0"
                    and self._has_edge_weights):
                ew_ei, ew_w = self._extract_ew_edges(graph, hodata)
                if ew_ei is not None:
                    ew_ei = ew_ei.to(x.device)
                    ew_w = ew_w.to(x.device)
                    # Weighted out-degree and in-degree from edge-weight edges
                    ew_out = scatter_add(ew_w, ew_ei[0], dim=0, dim_size=num_nodes)
                    ew_in = scatter_add(ew_w, ew_ei[1], dim=0, dim_size=num_nodes)
                    # Uniform degree from remaining (non-weighted) edges
                    all_out = degree(e[0], num_nodes=num_nodes)
                    all_in = degree(e[1], num_nodes=num_nodes)
                    n_ew_out = degree(ew_ei[0], num_nodes=num_nodes)
                    n_ew_in = degree(ew_ei[1], num_nodes=num_nodes)
                    unw_out = all_out - n_ew_out  # count of unweighted edges
                    unw_in = all_in - n_ew_in
                    degrees = (ew_out + unw_out) + (ew_in + unw_in)
                else:
                    degrees = degree(e[0], num_nodes=num_nodes) + degree(
                        e[1], num_nodes=num_nodes
                    )
            else:
                degrees = degree(e[0], num_nodes=num_nodes) + degree(
                    e[1], num_nodes=num_nodes
                )
            degrees = torch.softmax(degrees, dim=0)
            degrees = torch.pow(self.amplifier, degrees)

            # RGCNConv message passing
            edge_type = hodata.edge_type
            if self._use_ew_message:
                ew = self._build_edge_weight(graph, e_dict, x.device)
                for i, conv in enumerate(self.convs):
                    x = conv(x, e, edge_type, edge_weight=ew)
                    if i != len(self.convs) - 1:
                        x = x.relu()
            else:
                for i, conv in enumerate(self.convs):
                    x = conv(x, e, edge_type)
                    if i != len(self.convs) - 1:
                        x = x.relu()

            # Split by node type (same as original lines 202-203)
            x = [x[m] for m in cate_mask]
            degrees = [degrees[m] for m in cate_mask]

            # Category-level extraction + LLM modulation (same as original line 207)
            cate_emb = self.cate_emb_module.get_cate_emb(x, degrees)
            for i in range(len(cate_emb)):
                cate_emb[i] = cate_emb[i] * (torch.log(torch.abs(llm_cate_embs[i]))).to(cate_emb[i].device)

            # Graph-level extraction + LLM modulation (same as original line 210)
            graph_emb, cate_w = self.graph_emb_module.get_graph_emb(cate_emb)
            graph_emb = graph_emb * (torch.log(torch.abs(llm_graph_emb))).to(graph_emb.device)

            # NODE-LEVEL PROSPECTUS MODULATION (Level 1, paper-style)
            # x_fund <- x_fund * log|MLP_strat(s)| * log|MLP_risk(r)|
            # Cold-start (delta_t == -1) → identity modulation.
            if (self.prospectus_loader is not None
                    and self.strategy_mlp is not None
                    and self._dataset_keys is not None
                    and 'fund' in self.cates):
                fund_pos = self.cates.index('fund')
                # Recover the snapshot index from edge_time (same trick as
                # core/models/multitask_edge.py:160 in the existing prospectus integration)
                _etype = ('fund', 'holds_stock', 'stock')
                if (_etype in graph.edge_types
                        and hasattr(graph[_etype], 'edge_time')
                        and graph[_etype].edge_time.numel() > 0):
                    snapshot_idx = int(graph[_etype].edge_time[0].item())
                else:
                    snapshot_idx = ttype
                if snapshot_idx < len(self._dataset_keys):
                    qi = self.prospectus_loader.timestamp_to_quarter_idx(
                        self._dataset_keys[snapshot_idx]
                    )
                else:
                    qi = -1
                if 0 <= qi < self.prospectus_loader.N_QUARTERS:
                    # The dataset's _select_snapshot (core/data/funds_edge_weight.py:407)
                    # remaps every snapshot to a unified-row coordinate system where
                    # x[fund] has shape (N_unified, feat_dim) and row index i IS the
                    # unified fund ID. The H5's fund_ids array uses the same 0..N-1
                    # numbering, so torch.arange(N_unified) is the correct lookup tensor.
                    # (This mirrors the existing prospectus integration at
                    #  core/models/multitask_edge.py:174.)
                    # Note: graph['fund'].id_idx is consumed by _select_snapshot and
                    # is NOT preserved on the output snapshot — do not try to read it.
                    fund_ids = torch.arange(x[fund_pos].shape[0], dtype=torch.long)
                    _sr_feats = self.prospectus_loader.get_fund_strategy_risk(fund_ids, qi)
                    strategy_emb_t = _sr_feats['strategy_emb'].to(x[fund_pos].device).float()
                    risk_emb_t = _sr_feats['risk_emb'].to(x[fund_pos].device).float()
                    delta_t = _sr_feats['delta_t'].to(x[fund_pos].device)

                    EPS = 1e-6
                    valid = (delta_t != -1).unsqueeze(-1)                  # (N_fund, 1)

                    if self._use_cold_token:
                        # COLD-START TOKEN PATH: substitute learned cold tokens for cold funds
                        # BEFORE the MLP. Then run MLP on the resulting tensor and apply the
                        # modulation to all funds (no identity fallback — cold funds get the
                        # learned cold-token modulation instead).
                        cold_strat = self.strategy_cold.expand_as(strategy_emb_t).to(strategy_emb_t.device)
                        cold_risk = self.risk_cold.expand_as(risk_emb_t).to(risk_emb_t.device)
                        strategy_emb_in = torch.where(valid, strategy_emb_t, cold_strat)
                        risk_emb_in = torch.where(valid, risk_emb_t, cold_risk)
                        strat_proj = self.strategy_mlp(strategy_emb_in)
                        risk_proj = self.risk_mlp(risk_emb_in)
                        strat_mod = torch.log(torch.abs(strat_proj) + EPS)
                        risk_mod = torch.log(torch.abs(risk_proj) + EPS)
                        # No identity fallback — every fund gets a real modulation now
                    else:
                        # ORIGINAL PATH (identity fallback for cold funds, unchanged from v1)
                        strat_proj = self.strategy_mlp(strategy_emb_t)         # (N_fund, hid_dim)
                        risk_proj = self.risk_mlp(risk_emb_t)                  # (N_fund, hid_dim)
                        strat_mod = torch.log(torch.abs(strat_proj) + EPS)
                        risk_mod = torch.log(torch.abs(risk_proj) + EPS)
                        strat_mod = torch.where(valid, strat_mod, torch.ones_like(strat_mod))
                        risk_mod = torch.where(valid, risk_mod, torch.ones_like(risk_mod))

                    x[fund_pos] = x[fund_pos] * strat_mod * risk_mod

                    # Diagnostic accounting
                    self._cold_n_total += int(delta_t.shape[0])
                    self._cold_n_valid += int(valid.sum().item())
                    self._mod_strat_sum += float(strat_mod.detach().abs().mean().item())
                    self._mod_risk_sum += float(risk_mod.detach().abs().mean().item())
                    self._mod_n_obs += 1

            # Multi-level fusion (same as original line 213)
            for i in range(len(x)):
                x[i] = x[i] * self.node_w + cate_emb[i] * self.cate_w + graph_emb * self.graph_w

            x = torch.cat(x, dim=0)

            if ttype == self.timeframe[0]:
                feats = [x]
            else:
                feats.append(x)

        # Temporal attention aggregation (same as original line 223, without covid branch)
        x = self.temporal_attn.temporal_encode(feats)

        if isinstance(predict_mask, list):
            x = [x[predict_mask[0]], x[predict_mask[1]]]
        else:
            x = x[predict_mask]

        return x

    def log_prospectus_diagnostics(self, prefix: str = "") -> None:
        """Print per-epoch cold-start fraction and mean modulation magnitudes.

        Resets the counters after printing. No-op if modulation is disabled
        (loader is None) or if no observations have accumulated since the last
        call (e.g., the first epoch hasn't run any forward passes yet).
        """
        if self.prospectus_loader is None or self._mod_n_obs == 0:
            return
        cold_frac = 1.0 - (self._cold_n_valid / max(1, self._cold_n_total))
        strat = self._mod_strat_sum / self._mod_n_obs
        risk = self._mod_risk_sum / self._mod_n_obs
        print(
            f"[NODE_TEXT]{prefix} cold-start frac = {cold_frac:.3f} "
            f"({self._cold_n_total - self._cold_n_valid}/{self._cold_n_total}); "
            f"mean |log strat_mod| = {strat:.3f}; mean |log risk_mod| = {risk:.3f}"
        )
        # Reset counters for the next epoch
        self._cold_n_total = 0
        self._cold_n_valid = 0
        self._mod_strat_sum = 0.0
        self._mod_risk_sum = 0.0
        self._mod_n_obs = 0

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
