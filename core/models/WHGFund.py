"""
WHGFund: Weighted Heterogeneous Graph for Fund Performance Prediction

Adapted from WHGDroid (Huang et al., 2023):
"WHGDroid: Effective android malware detection based on weighted heterogeneous graph"
Journal of Information Security and Applications 77 (2023) 103556

Original paper methodology:
- Constructs weighted heterogeneous graph with multiple entity types
- Assigns weights to entities based on occurrence frequency differences
- Uses metapaths to establish implicit associations
- Applies GraphSAGE for node embedding
- Uses attention mechanism to fuse multiple metapath embeddings

Adaptation to Funds dataset:
- Entity types: fund, stock, other_asset, manager (instead of App, API, Permission, Intent)
- Relations: holds_stock, holds_other, managed_by (instead of App-API, App-Permission, etc.)
- Edge weights: Use existing holding amounts from graph (weighted edges) or 1.0 (unweighted edges)
- Metapaths: fund-stock-fund, fund-manager-fund, fund-other_asset-fund
- Target: Fund performance prediction (regression instead of classification)

Key simplification:
- Entity weight calculation (Formula 1 from paper) is NOT needed because the Funds dataset
  already has meaningful edge weights (holding amounts). For unweighted edges like fund-manager,
  we use weight=1.0, which is standard practice in graph learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter
import math


# Note: Entity weight calculation (Formula 1 from WHGDroid) is not needed for Funds dataset
# because edges already have meaningful weights (holding amounts for stocks/other_assets).
# For unweighted edges (e.g., fund-manager), we use weight=1.0, which is standard practice.


class MetapathBuilder:
    """
    Constructs metapaths following WHGDroid methodology (Section 3.2.2).
    
    STATIC METAPATHS (within same timestep):
    MP1: fund-holds_stock-stock-rev_holds_stock-fund (funds holding same stocks)
    MP2: fund-holds_other-other_asset-rev_holds_other-fund (funds holding same other assets)
    MP3: fund-managed_by-manager-rev_managed_by-fund (funds with same manager)
    """
    def __init__(self, relation_matrices):
        """
        Args:
            relation_matrices: dict with keys like 'holds_stock', 'holds_other', 'managed_by'
                               Each matrix N is fund x entity, where N[i,j] = weight if fund_i uses entity_j
        """
        self.relation_matrices = relation_matrices
    
    def compute_metapath_adjacency(self, metapath_name):
        """
        Compute adjacency matrix for a metapath using Formula (2) from paper:
        phi_MP = R_A1A2 * R_A2A3 * ... * R_AL A_{L+1}
        
        Args:
            metapath_name: one of 'MP1' (stock), 'MP2' (other_asset), 'MP3' (manager)
        
        Returns:
            torch.Tensor: fund x fund adjacency matrix
        """
        if metapath_name == 'MP1':  # fund-stock-fund
            N = self.relation_matrices['holds_stock']  # fund x stock
            phi = torch.mm(N, N.t())  # fund x fund
        elif metapath_name == 'MP2':  # fund-other_asset-fund
            P = self.relation_matrices['holds_other']  # fund x other_asset
            phi = torch.mm(P, P.t())  # fund x fund
        elif metapath_name == 'MP3':  # fund-manager-fund
            M = self.relation_matrices['managed_by']  # fund x manager
            phi = torch.mm(M, M.t())  # fund x fund
        else:
            raise ValueError(f"Unknown metapath: {metapath_name}")
        
        return phi
    
    def compute_pathsim(self, phi):
        """
        Compute PathSim similarity using Formula (3) from paper:
        Sim_MP(App_i, App_j) = 2 * phi_MP[i,j] / (phi_MP[i,i] + phi_MP[j,j])
        
        Args:
            phi: fund x fund adjacency matrix from metapath
        
        Returns:
            torch.Tensor: fund x fund similarity matrix
        """
        # Get diagonal elements
        diag = torch.diag(phi).unsqueeze(1)  # (num_funds, 1)
        
        # Compute pairwise sums of diagonals
        denom = diag + diag.t()  # (num_funds, num_funds)
        
        # Avoid division by zero
        denom = torch.clamp(denom, min=1e-8)
        
        # Compute PathSim
        sim = 2.0 * phi / denom
        
        return sim


class TemporalMetapathBuilder:
    """
    Constructs TEMPORAL METAPATHS across different time snapshots.
    
    This extends WHGDroid to handle time-series data by creating metapaths that
    track how funds evolve over time.
    
    TEMPORAL METAPATHS:
    ==================
    MP_temporal_self: fund(t) --evolves_to--> fund(t+1) --evolves_to--> fund(t+2)
        Semantic: A fund's temporal continuity and evolution patterns
        
    MP_temporal_stock: fund_i(t) --stock(t)-- fund_j(t) --evolves_to--> fund_j(t+1)
        Semantic: Funds with similar holdings at time t, then track evolution of similar fund
        
    MP_temporal_cross: fund_i(t) --evolves_to--> fund_i(t+1) --stock(t+1)-- fund_j(t+1)
        Semantic: A fund's evolution + its similarity to other funds at next timestep
    
    This approach is inspired by dynamic graph learning:
    - DySAT: Dynamic Self-Attention Network (ICLR 2019)
    - EvolveGCN: Evolving GCN (AAAI 2020)
    - TGAT: Temporal Graph Attention (ICLR 2020)
    """
    def __init__(self, relation_matrices_temporal, temporal_decay=0.9):
        """
        Args:
            relation_matrices_temporal: list of dicts, one per timestep
                                       Each dict has keys 'holds_stock', 'holds_other', 'managed_by'
            temporal_decay: weight decay for older timesteps (default 0.9)
                           fund(t) --0.9--> fund(t+1) --0.81--> fund(t+2)
        """
        self.relation_matrices_temporal = relation_matrices_temporal
        self.temporal_decay = temporal_decay
        self.num_time = len(relation_matrices_temporal)
    
    def compute_temporal_self_similarity(self):
        """
        MP_temporal_self: fund(t) --evolves_to--> fund(t+1) --evolves_to--> fund(t+2)
        
        Creates adjacency matrix where fund_i(t) connects to fund_i(t+1) with decaying weight.
        Then aggregates across all timesteps to get fund-fund similarity based on temporal co-evolution.
        
        Returns:
            torch.Tensor: (num_funds, num_funds) similarity matrix
        """
        # Get number of funds from first timestep
        first_rel_matrix = next(iter(self.relation_matrices_temporal[0].values()))
        num_funds = first_rel_matrix.shape[0]
        device = first_rel_matrix.device
        
        # Initialize temporal co-evolution matrix
        temporal_sim = torch.zeros(num_funds, num_funds, device=device)
        
        # For each consecutive pair of timesteps
        for t in range(self.num_time - 1):
            # Create temporal edge: fund_i(t) connects to fund_i(t+1)
            # This is an identity matrix with temporal decay
            weight = self.temporal_decay ** (t + 1)  # Decay for distance from current time
            
            # Self-connection across time (fund_i at different timesteps)
            temporal_edge = torch.eye(num_funds, device=device) * weight
            
            # Accumulate: funds that exist across multiple timesteps are similar
            temporal_sim += temporal_edge
        
        # Normalize
        temporal_sim = temporal_sim / (self.num_time - 1) if self.num_time > 1 else temporal_sim
        
        return temporal_sim
    
    def compute_temporal_stock_similarity(self):
        """
        MP_temporal_stock: fund_i(t) --stock(t)-- fund_j(t) --evolves_to--> fund_j(t+1)
        
        At time t, find funds with similar stock holdings.
        Then track how those similar funds evolve to t+1.
        
        Returns:
            torch.Tensor: (num_funds, num_funds) similarity matrix
        """
        first_rel_matrix = next(iter(self.relation_matrices_temporal[0].values()))
        num_funds = first_rel_matrix.shape[0]
        device = first_rel_matrix.device
        
        temporal_stock_sim = torch.zeros(num_funds, num_funds, device=device)
        
        for t in range(self.num_time - 1):
            # Spatial similarity at time t (fund-stock-fund)
            if 'holds_stock' in self.relation_matrices_temporal[t]:
                N_t = self.relation_matrices_temporal[t]['holds_stock']
                spatial_sim_t = torch.mm(N_t, N_t.t())  # (num_funds, num_funds)
                
                # Temporal evolution weight
                weight = self.temporal_decay ** (self.num_time - 1 - t)  # Recent times more important
                
                # Combine spatial and temporal
                temporal_stock_sim += spatial_sim_t * weight
        
        # Normalize
        temporal_stock_sim = temporal_stock_sim / (self.num_time - 1) if self.num_time > 1 else temporal_stock_sim
        
        return temporal_stock_sim
    
    def compute_temporal_cross_similarity(self):
        """
        MP_temporal_cross: fund_i(t) --evolves_to--> fund_i(t+1) --stock(t+1)-- fund_j(t+1)
        
        Track a fund's evolution, then find similar funds at the next timestep.
        
        Returns:
            torch.Tensor: (num_funds, num_funds) similarity matrix
        """
        first_rel_matrix = next(iter(self.relation_matrices_temporal[0].values()))
        num_funds = first_rel_matrix.shape[0]
        device = first_rel_matrix.device
        
        temporal_cross_sim = torch.zeros(num_funds, num_funds, device=device)
        
        for t in range(self.num_time - 1):
            # Temporal self-connection: fund_i(t) -> fund_i(t+1)
            temporal_edge = torch.eye(num_funds, device=device)
            
            # Spatial similarity at time t+1
            if 'holds_stock' in self.relation_matrices_temporal[t + 1]:
                N_t1 = self.relation_matrices_temporal[t + 1]['holds_stock']
                spatial_sim_t1 = torch.mm(N_t1, N_t1.t())
                
                # Combine: temporal_edge @ spatial_sim_t1
                cross_sim = torch.mm(temporal_edge, spatial_sim_t1)
                
                weight = self.temporal_decay ** (self.num_time - 1 - t)
                temporal_cross_sim += cross_sim * weight
        
        # Normalize
        temporal_cross_sim = temporal_cross_sim / (self.num_time - 1) if self.num_time > 1 else temporal_cross_sim
        
        return temporal_cross_sim


class WeightedSAGEConv(nn.Module):
    """
    Weighted GraphSAGE-style convolution with mean aggregation.
    Neighbor messages are weighted by edge weights (adjacency weights).
    """
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.lin.weight)
        if bias:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        # Ensure edge_index and edge_weight are on same device as x
        edge_index = edge_index.to(x.device)
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            edge_weight = torch.ones(src.size(0), device=x.device)
        else:
            edge_weight = edge_weight.to(x.device)
        # Weighted sum of neighbor features per destination
        msg = x[src] * edge_weight.unsqueeze(-1)
        neigh_sum = scatter(msg, dst, dim=0, dim_size=num_nodes, reduce='sum')
        weight_sum = scatter(edge_weight, dst, dim=0, dim_size=num_nodes, reduce='sum')
        denom = (weight_sum.unsqueeze(-1) + 1e-12)
        neigh_mean = neigh_sum / denom
        out = torch.cat([x, neigh_mean], dim=-1)
        return self.lin(out)


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE-based node embedding following WHGDroid Section 3.3.2.
    
    Implements Formulas (4) and (5):
    h^k_N(v) = Aggregate^k({h^{k-1}_u, for all u in N(v)})
    h^k_v = sigma(W^k * concat(h^{k-1}_v, h^k_N(v)))
    """
    def __init__(self, in_dim, hid_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        # First layer (weighted)
        self.convs.append(WeightedSAGEConv(in_dim, hid_dim))
        # Additional layers (weighted)
        for _ in range(num_layers - 1):
            self.convs.append(WeightedSAGEConv(hid_dim, hid_dim))
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: node features (num_nodes, in_dim)
            edge_index: (2, num_edges)
            edge_weight: optional edge weights (num_edges,)
        
        Returns:
            node embeddings (num_nodes, hid_dim)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < self.num_layers - 1:
                x = F.relu(x)
        
        return x


class SemanticAttention(nn.Module):
    """
    Semantic-level attention for metapath fusion following WHGDroid Section 3.4.
    
    Implements Formulas (6), (7), (8):
    omega_MPn = 1/|V| * sum_{i in V} q^T * tanh(W * H^i_MPn + b)
    beta_MPn = exp(omega_MPn) / sum_n exp(omega_MPn)
    """
    def __init__(self, hid_dim):
        super().__init__()
        self.W = nn.Linear(hid_dim, hid_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(hid_dim))
        self.q = nn.Parameter(torch.randn(hid_dim))
        
        # Initialize
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.q.unsqueeze(0))
    
    def forward(self, embeddings_list):
        """
        Args:
            embeddings_list: list of (num_nodes, hid_dim) tensors, one per metapath
        
        Returns:
            fused_embedding: (num_nodes, hid_dim)
            attention_weights: (num_metapaths,)
        """
        num_metapaths = len(embeddings_list)
        
        # Compute importance score for each metapath
        omega_list = []
        for H_mp in embeddings_list:
            # omega_mp = 1/|V| * sum_i q^T * tanh(W * H^i_mp + b)
            transformed = torch.tanh(self.W(H_mp) + self.b)  # (num_nodes, hid_dim)
            scores = torch.matmul(transformed, self.q)  # (num_nodes,)
            omega = scores.mean()  # scalar
            omega_list.append(omega)
        
        # Normalize via softmax to get attention weights (Formula 8)
        omega_tensor = torch.stack(omega_list)  # (num_metapaths,)
        beta = F.softmax(omega_tensor, dim=0)  # (num_metapaths,)
        
        # Fuse embeddings with learned weights (Formula 9)
        # H = sum_n (beta_MPn * H_MPn)
        fused = torch.zeros_like(embeddings_list[0])
        for i, H_mp in enumerate(embeddings_list):
            fused += beta[i] * H_mp
        
        return fused, beta


class TemporalSelfAttention(nn.Module):
    """
    Multi-head Q/K/V self-attention over time (like DyHATR's TemporalAttentionLayer).
    Allows each timestep to attend to all previous timesteps (causal masking).
    """
    def __init__(self, hid_dim, num_heads=4, num_timesteps=8, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.num_timesteps = num_timesteps
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(torch.Tensor(num_timesteps, hid_dim))
        
        # Q, K, V projections
        self.Q_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.K_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.V_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(hid_dim, hid_dim, bias=True)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_proj.weight)
        nn.init.xavier_uniform_(self.K_proj.weight)
        nn.init.xavier_uniform_(self.V_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, seq_embeddings):
        """
        Args:
            seq_embeddings: (T, N, H) tensor of embeddings across time
        Returns:
            output: (T, N, H) attended embeddings
            attn_weights: (num_heads * N, T, T) attention weights
        """
        T, N, H = seq_embeddings.shape
        
        # Permute to (N, T, H)
        x = seq_embeddings.permute(1, 0, 2)  # (N, T, H)
        
        # Add position embeddings
        position_ids = torch.arange(0, min(T, self.num_timesteps), device=x.device).unsqueeze(0)  # (1, T)
        position_ids = position_ids.expand(N, -1)  # (N, T)
        pos_emb = self.position_embeddings[position_ids]  # (N, T, H)
        x = x + pos_emb  # (N, T, H)
        
        # Q, K, V projections
        Q = self.Q_proj(x)  # (N, T, H)
        K = self.K_proj(x)  # (N, T, H)
        V = self.V_proj(x)  # (N, T, H)
        
        # Split into multiple heads: (N, T, H) -> (N, T, num_heads, H/num_heads) -> (N, num_heads, T, H/num_heads)
        head_dim = H // self.num_heads
        Q = Q.view(N, T, self.num_heads, head_dim).transpose(1, 2)  # (N, num_heads, T, head_dim)
        K = K.view(N, T, self.num_heads, head_dim).transpose(1, 2)  # (N, num_heads, T, head_dim)
        V = V.view(N, T, self.num_heads, head_dim).transpose(1, 2)  # (N, num_heads, T, head_dim)
        
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (N, num_heads, T, T)
        scores = scores / math.sqrt(head_dim)
        
        # Causal masking: only attend to past and current timesteps
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))  # (T, T)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (N, num_heads, T, T)
        
        # Dropout
        if self.training:
            attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to V
        attn_output = torch.matmul(attn_weights, V)  # (N, num_heads, T, head_dim)
        
        # Concatenate heads: (N, num_heads, T, head_dim) -> (N, T, H)
        attn_output = attn_output.transpose(1, 2).contiguous().view(N, T, H)
        
        # Output projection
        output = self.out_proj(attn_output)  # (N, T, H)
        
        # Permute back to (T, N, H)
        output = output.permute(1, 0, 2)
        
        return output, attn_weights


class WHGFund(nn.Module):
    """
    WHGFund: Weighted Heterogeneous Graph for Fund Performance Prediction
    
    Architecture following WHGDroid paper:
    1. Entity weight calculation (Section 3.1.2)
    2. Weighted heterogeneous graph construction (Section 3.2.1)
    3. Metapath-based homogeneous graph decomposition (Section 3.2.2, 3.3.1)
    4. GraphSAGE node embedding per metapath (Section 3.3.2)
    5. Semantic attention for metapath fusion (Section 3.4)
    6. MLP for regression (adapted from classification)
    """
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers=2,
        metapaths=None,
        pathsim_threshold=0.1,
        device='cpu'
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.device = device
        self.pathsim_threshold = pathsim_threshold
        
        # Default metapaths: fund-stock-fund, fund-other_asset-fund, fund-manager-fund
        if metapaths is None:
            self.metapaths = ['MP1', 'MP2', 'MP3']
        else:
            self.metapaths = metapaths
        
        # GraphSAGE encoders for each metapath
        self.encoders = nn.ModuleDict({
            mp: GraphSAGEEncoder(in_dim, hid_dim, num_layers)
            for mp in self.metapaths
        })
        
        # Semantic attention for metapath fusion
        self.semantic_attn = SemanticAttention(hid_dim)
        
        # MLP for final regression (following Section 3.4)
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hid_dim, out_dim)
        )
        
        # Storage for relation matrices and metapath graphs
        self.relation_matrices = {}
        self.metapath_graphs = {}
    
    def set_relation_matrices(self, relation_matrices):
        """
        Set relation matrices for metapath construction.
        
        Args:
            relation_matrices: dict with keys 'holds_stock', 'holds_other', 'managed_by'
                               Each is a sparse or dense tensor (num_funds, num_entities)
        """
        self.relation_matrices = relation_matrices
    
    def build_metapath_graphs(self, fund_features):
        """
        Build homogeneous fund-fund graphs for each metapath using PathSim.
        
        Args:
            fund_features: (num_funds, in_dim) initial node features
        
        Returns:
            dict mapping metapath_name -> (edge_index, edge_weight)
        """
        num_funds = fund_features.shape[0]
        # Ensure relation matrices are on same device as features
        device = fund_features.device
        if self.relation_matrices:
            moved = {}
            for k, v in self.relation_matrices.items():
                moved[k] = v.to(device)
            self.relation_matrices = moved
        metapath_builder = MetapathBuilder(self.relation_matrices)
        
        for mp_name in self.metapaths:
            # Compute metapath adjacency (Formula 2)
            phi = metapath_builder.compute_metapath_adjacency(mp_name)
            
            # Compute PathSim similarity (Formula 3)
            sim = metapath_builder.compute_pathsim(phi)
            
            # Threshold to reduce graph size (mentioned in Section 3.3.1)
            mask = sim > self.pathsim_threshold
            
            # Convert to edge_index and edge_weight
            edge_index = mask.nonzero(as_tuple=False).t().to(device)  # (2, num_edges)
            edge_weight = sim[mask].to(device)  # (num_edges,)
            
            self.metapath_graphs[mp_name] = (edge_index, edge_weight)
        
        return self.metapath_graphs
    
    def encode(self, fund_features):
        """
        Encode fund nodes through GraphSAGE on each metapath, then fuse via attention.
        
        Args:
            fund_features: (num_funds, in_dim) node features
        
        Returns:
            fused_embeddings: (num_funds, hid_dim)
            metapath_attention_weights: (num_metapaths,)
        """
        # Ensure tensors on the same device as module parameters
        device = next(self.parameters()).device
        fund_features = fund_features.to(device)
        # Ensure metapath graphs are built
        if not self.metapath_graphs:
            self.build_metapath_graphs(fund_features)
        
        # Encode using GraphSAGE for each metapath
        embeddings_list = []
        for mp_name in self.metapaths:
            edge_index, edge_weight = self.metapath_graphs[mp_name]
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
            
            # Apply GraphSAGE encoder
            h_mp = self.encoders[mp_name](fund_features, edge_index, edge_weight)
            embeddings_list.append(h_mp)
        
        # Fuse embeddings via semantic attention
        fused_emb, attn_weights = self.semantic_attn(embeddings_list)
        
        return fused_emb, attn_weights
    
    def forward(self, fund_features):
        """
        Forward pass for fund performance prediction.
        
        Args:
            fund_features: (num_funds, in_dim) node features
        
        Returns:
            predictions: (num_funds, out_dim) predicted fund performance
        """
        # Encode funds via metapath-guided GraphSAGE + attention
        fused_emb, _ = self.encode(fund_features)
        
        # Final MLP for regression
        preds = self.mlp(fused_emb)
        
        return preds


class WHGFundTemporalWrapper(nn.Module):
    """
    Temporal wrapper for WHGFund to handle time-series fund data.
    
    Note: The original WHGDroid paper doesn't handle time-series data.
    This wrapper extends it with FIVE temporal strategies:
    
    1. 'last': Use only last timestep (simple, but loses temporal patterns)
    2. 'mean': Average features/graphs across time (temporal smoothing)
    3. 'all': Process each timestep and aggregate embeddings (mean)
    4. 'cross_time_attention': Attention over per-timestep embeddings (learned temporal weights)
    5. 'temporal_metapath': Use TEMPORAL METAPATHS across timesteps (captures temporal evolution)
       This creates metapaths like fund(t) --evolves_to--> fund(t+1) to track temporal patterns.
    """
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers=2,
        metapaths=None,
        pathsim_threshold=0.1,
        temporal_mode='last',  # 'last', 'mean', or 'all'
        n_heads=4,  # Multi-head for temporal self-attention (same as DHSpace/DyHATR)
        time_window=8,  # Number of timesteps for position embeddings
        device='cpu'
    ):
        super().__init__()
        
        self.temporal_mode = temporal_mode
        self.n_heads = n_heads  # Store for later use
        self.time_window = time_window  # Store for temporal self-attention
        
        self.whg_fund = WHGFund(
            in_dim=in_dim,
            hid_dim=hid_dim,
            out_dim=hid_dim,  # intermediate embedding
            num_layers=num_layers,
            metapaths=metapaths,
            pathsim_threshold=pathsim_threshold,
            device=device
        )
        
        # Final prediction head
        self.predictor = nn.Linear(hid_dim, out_dim)

    def _ensure_input_dim(self, fund_features):
        """
        Ensure GraphSAGE encoders match actual fund feature dimension.
        Rebuild encoders on-the-fly if there's a mismatch.
        """
        feat_dim = fund_features.size(1)
        device = fund_features.device
        if feat_dim != self.whg_fund.in_dim:
            self.whg_fund.in_dim = feat_dim
            # Rebuild encoders for each metapath
            num_layers = next(iter(self.whg_fund.encoders.values())).num_layers if len(self.whg_fund.encoders) > 0 else 2
            self.whg_fund.encoders = nn.ModuleDict({
                mp: GraphSAGEEncoder(feat_dim, self.whg_fund.hid_dim, num_layers).to(device)
                for mp in self.whg_fund.metapaths
            })
            # Clear cached graphs
            self.whg_fund.metapath_graphs = {}
    
    def forward(self, xs, graphs):
        """
        Handle temporal fund data.
        
        Four strategies implemented (controlled by self.temporal_mode):
        1. 'last': Use only last timestep (original implementation - simple but loses temporal info)
        2. 'mean': Average features and graphs across time (smoothing)
        3. 'all': Process each timestep separately and aggregate embeddings (comprehensive)
        4. 'temporal_metapath': Use TEMPORAL METAPATHS across timesteps (RECOMMENDED - tracks fund evolution!)
        
        Args:
            xs: dict mapping node_type -> (num_time, num_nodes, feat_dim) features
            graphs: list of HeteroData graphs, one per timestep
        
        Returns:
            (num_funds, out_dim) predictions for fund nodes
        """
        # Extract fund features across time
        fund_features_temporal = xs['fund']  # (num_time, num_funds, feat_dim)
        num_time, num_funds, feat_dim = fund_features_temporal.shape
        
        temporal_mode = getattr(self, 'temporal_mode', 'last')
        
        if temporal_mode == 'last':
            # OPTION 1: Use only the last timestep (simplest, but loses temporal info)
            fund_features = fund_features_temporal[-1]  # (num_funds, feat_dim)
            self._ensure_input_dim(fund_features)
            last_graph = graphs[-1]
            relation_matrices = self._extract_relation_matrices(last_graph)
            self.whg_fund.set_relation_matrices(relation_matrices)
            preds = self.whg_fund(fund_features)
        
        elif temporal_mode == 'mean':
            # OPTION 2: Average features across time (temporal smoothing)
            fund_features = fund_features_temporal.mean(dim=0)  # (num_funds, feat_dim)
            self._ensure_input_dim(fund_features)
            
            # Average relation matrices across all timesteps
            relation_matrices_all = [self._extract_relation_matrices(g) for g in graphs]
            relation_matrices = {}
            for key in relation_matrices_all[0].keys():
                # Stack and average across time
                stacked = torch.stack([rm[key] for rm in relation_matrices_all], dim=0)
                relation_matrices[key] = stacked.mean(dim=0)
            
            self.whg_fund.set_relation_matrices(relation_matrices)
            preds = self.whg_fund(fund_features)
        
        elif temporal_mode == 'all':
            # OPTION 3: Process each timestep separately and aggregate embeddings
            embeddings_list = []
            for t in range(num_time):
                fund_features_t = fund_features_temporal[t]  # (num_funds, feat_dim)
                self._ensure_input_dim(fund_features_t)
                graph_t = graphs[t]
                relation_matrices_t = self._extract_relation_matrices(graph_t)
                self.whg_fund.set_relation_matrices(relation_matrices_t)
                # Build per-timestep metapath graphs to avoid stale cache
                self.whg_fund.build_metapath_graphs(fund_features_t)
                # Get embeddings (not final predictions)
                emb_t, _ = self.whg_fund.encode(fund_features_t)  # (num_funds, hid_dim)
                embeddings_list.append(emb_t)
            
            # Aggregate temporal embeddings (simple mean, could use attention)
            temporal_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)  # (num_funds, hid_dim)
            preds = temporal_embeddings  # Will be passed to predictor
        
        elif temporal_mode == 'cross_time_attention':
            # OPTION 4: Multi-head Q/K/V self-attention over timesteps (like DyHATR)
            embeddings_list = []
            for t in range(num_time):
                fund_features_t = fund_features_temporal[t]
                self._ensure_input_dim(fund_features_t)
                graph_t = graphs[t]
                relation_matrices_t = self._extract_relation_matrices(graph_t)
                self.whg_fund.set_relation_matrices(relation_matrices_t)
                # Build per-timestep metapath graphs to avoid stale cache
                self.whg_fund.build_metapath_graphs(fund_features_t)
                emb_t, _ = self.whg_fund.encode(fund_features_t)  # (num_funds, hid_dim)
                embeddings_list.append(emb_t)

            seq_emb = torch.stack(embeddings_list, dim=0)  # (T, N, H)
            # Lazy init on correct device
            if not hasattr(self, 'time_attn'):
                self.time_attn = TemporalSelfAttention(
                    self.whg_fund.hid_dim,
                    num_heads=self.n_heads,  # Use same n_heads as DHSpace/DyHATR
                    num_timesteps=self.time_window,
                    dropout=0.1,
                ).to(seq_emb.device)
            attended_emb, _ = self.time_attn(seq_emb)  # (T, N, H)
            # Take last timestep output (like DyHATR)
            preds = attended_emb[-1]  # (N, H)

        elif temporal_mode == 'temporal_metapath':
            # OPTION 5: Use TEMPORAL METAPATHS across timesteps
            # This captures fund(t) --evolves_to--> fund(t+1) patterns
            
            # Extract relation matrices for all timesteps
            relation_matrices_all = [self._extract_relation_matrices(g) for g in graphs]
            
            # Build temporal metapath builder (class is defined in this file)
            temporal_decay = getattr(self, 'temporal_decay', 0.9)
            temp_mp_builder = TemporalMetapathBuilder(
                relation_matrices_all, 
                temporal_decay=temporal_decay
            )
            
            # Compute temporal metapath similarities
            temp_self_sim = temp_mp_builder.compute_temporal_self_similarity()
            temp_stock_sim = temp_mp_builder.compute_temporal_stock_similarity()
            temp_cross_sim = temp_mp_builder.compute_temporal_cross_similarity()
            
            # Use last timestep features
            fund_features = fund_features_temporal[-1]
            self._ensure_input_dim(fund_features)
            
            # Create extended metapaths: static + temporal
            # Static: MP1 (stock), MP2 (other_asset), MP3 (manager)
            # Temporal: MP_temp_self, MP_temp_stock, MP_temp_cross
            
            # Set static metapaths
            relation_matrices = relation_matrices_all[-1]  # Use last timestep for static
            self.whg_fund.set_relation_matrices(relation_matrices)
            
            # Build metapath graphs for both static and temporal
            static_metapath_graphs = self.whg_fund.build_metapath_graphs(fund_features)
            
            # Add temporal metapath graphs
            temporal_metapath_graphs = {
                'MP_temp_self': self._pathsim_to_graph(temp_self_sim, self.whg_fund.pathsim_threshold),
                'MP_temp_stock': self._pathsim_to_graph(temp_stock_sim, self.whg_fund.pathsim_threshold),
                'MP_temp_cross': self._pathsim_to_graph(temp_cross_sim, self.whg_fund.pathsim_threshold),
            }
            
            # Encode using GraphSAGE for BOTH static and temporal metapaths
            embeddings_list = []
            
            # Process static metapaths
            for mp_name in self.whg_fund.metapaths:
                edge_index, edge_weight = static_metapath_graphs[mp_name]
                edge_index = edge_index.to(fund_features.device)
                edge_weight = edge_weight.to(fund_features.device)
                h_mp = self.whg_fund.encoders[mp_name](fund_features, edge_index, edge_weight)
                embeddings_list.append(h_mp)
            
            # Process temporal metapaths (use a shared encoder for all temporal)
            # Create encoders for temporal metapaths if they don't exist
            if not hasattr(self, 'temporal_encoders'):
                self.temporal_encoders = nn.ModuleDict({
                    'MP_temp_self': GraphSAGEEncoder(fund_features.shape[1], self.whg_fund.hid_dim, 2),
                    'MP_temp_stock': GraphSAGEEncoder(fund_features.shape[1], self.whg_fund.hid_dim, 2),
                    'MP_temp_cross': GraphSAGEEncoder(fund_features.shape[1], self.whg_fund.hid_dim, 2),
                }).to(fund_features.device)
            
            for temp_mp_name, (edge_index, edge_weight) in temporal_metapath_graphs.items():
                edge_index = edge_index.to(fund_features.device)
                edge_weight = edge_weight.to(fund_features.device)
                h_temp_mp = self.temporal_encoders[temp_mp_name](fund_features, edge_index, edge_weight)
                embeddings_list.append(h_temp_mp)
            
            # Fuse all embeddings (static + temporal) via semantic attention
            fused_emb, attn_weights = self.whg_fund.semantic_attn(embeddings_list)
            preds = fused_emb
        
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")
        
        # Final prediction
        output = self.predictor(preds)
        
        return output

    # --- Compatibility with CORE trainer API ---
    def encode(self, support):
        """
        Encode fund nodes into embeddings for node regression.

        Args:
            support: list of HeteroData graphs for the past timesteps (length = twin)

        Returns:
            z: (num_funds, hid_dim) fund embeddings
        """
        # Build temporal fund feature tensor from support graphs
        assert isinstance(support, (list, tuple)), "support must be a list of temporal graphs"
        graphs = support
        num_time = len(graphs)
        # Extract fund features per timestep and stack
        fund_feats = []
        for g in graphs:
            if 'fund' not in g.node_types or not hasattr(g['fund'], 'x'):
                raise RuntimeError("Missing fund.x in temporal graph")
            fund_feats.append(g['fund'].x)
        fund_features_temporal = torch.stack(fund_feats, dim=0)  # (T, N, F)
        device = next(self.parameters()).device

        # Reuse the temporal logic from forward but return embeddings before predictor
        temporal_mode = getattr(self, 'temporal_mode', 'last')

        if temporal_mode == 'last':
            fund_features = fund_features_temporal[-1].to(device)
            last_graph = graphs[-1]
            relation_matrices = self._extract_relation_matrices(last_graph)
            # Move relation matrices to device
            relation_matrices = {k: v.to(device) for k, v in relation_matrices.items()}
            self.whg_fund.set_relation_matrices(relation_matrices)
            # Get fused embedding (not final preds)
            fused_emb, _ = self.whg_fund.encode(fund_features)
            z = fused_emb

        elif temporal_mode == 'mean':
            fund_features = fund_features_temporal.mean(dim=0).to(device)
            relation_matrices_all = [self._extract_relation_matrices(g) for g in graphs]
            relation_matrices = {}
            for key in relation_matrices_all[0].keys():
                stacked = torch.stack([rm[key] for rm in relation_matrices_all], dim=0)
                relation_matrices[key] = stacked.mean(dim=0).to(device)
            self.whg_fund.set_relation_matrices(relation_matrices)
            fused_emb, _ = self.whg_fund.encode(fund_features)
            z = fused_emb

        elif temporal_mode == 'all':
            embeddings_list = []
            for t in range(num_time):
                fund_features_t = fund_features_temporal[t].to(device)
                graph_t = graphs[t]
                relation_matrices_t = self._extract_relation_matrices(graph_t)
                relation_matrices_t = {k: v.to(device) for k, v in relation_matrices_t.items()}
                self.whg_fund.set_relation_matrices(relation_matrices_t)
                self.whg_fund.build_metapath_graphs(fund_features_t)
                emb_t, _ = self.whg_fund.encode(fund_features_t)
                embeddings_list.append(emb_t)
            z = torch.stack(embeddings_list, dim=0).mean(dim=0)

        elif temporal_mode == 'cross_time_attention':
            embeddings_list = []
            for t in range(num_time):
                fund_features_t = fund_features_temporal[t].to(device)
                graph_t = graphs[t]
                relation_matrices_t = self._extract_relation_matrices(graph_t)
                relation_matrices_t = {k: v.to(device) for k, v in relation_matrices_t.items()}
                self.whg_fund.set_relation_matrices(relation_matrices_t)
                self.whg_fund.build_metapath_graphs(fund_features_t)
                emb_t, _ = self.whg_fund.encode(fund_features_t)
                embeddings_list.append(emb_t)
            seq_emb = torch.stack(embeddings_list, dim=0)
            if not hasattr(self, 'time_attn'):
                self.time_attn = TemporalSelfAttention(
                    self.whg_fund.hid_dim,
                    num_heads=self.n_heads,
                    num_timesteps=self.time_window,
                    dropout=0.1,
                ).to(seq_emb.device)
            attended_emb, _ = self.time_attn(seq_emb)
            z = attended_emb[-1]

        elif temporal_mode == 'temporal_metapath':
            relation_matrices_all = [self._extract_relation_matrices(g) for g in graphs]
            temporal_decay = getattr(self, 'temporal_decay', 0.9)
            temp_mp_builder = TemporalMetapathBuilder(
                relation_matrices_all,
                temporal_decay=temporal_decay
            )
            temp_self_sim = temp_mp_builder.compute_temporal_self_similarity()
            temp_stock_sim = temp_mp_builder.compute_temporal_stock_similarity()
            temp_cross_sim = temp_mp_builder.compute_temporal_cross_similarity()
            fund_features = fund_features_temporal[-1].to(device)
            # Ensure temporal similarity matrices are on device
            temp_self_sim = temp_self_sim.to(device)
            temp_stock_sim = temp_stock_sim.to(device)
            temp_cross_sim = temp_cross_sim.to(device)
            relation_matrices = relation_matrices_all[-1]
            relation_matrices = {k: v.to(device) for k, v in relation_matrices.items()}
            self.whg_fund.set_relation_matrices(relation_matrices)
            static_metapath_graphs = self.whg_fund.build_metapath_graphs(fund_features)
            temporal_metapath_graphs = {
                'MP_temp_self': self._pathsim_to_graph(temp_self_sim, self.whg_fund.pathsim_threshold),
                'MP_temp_stock': self._pathsim_to_graph(temp_stock_sim, self.whg_fund.pathsim_threshold),
                'MP_temp_cross': self._pathsim_to_graph(temp_cross_sim, self.whg_fund.pathsim_threshold),
            }
            embeddings_list = []
            for mp_name in self.whg_fund.metapaths:
                edge_index, edge_weight = static_metapath_graphs[mp_name]
                edge_index = edge_index.to(fund_features.device)
                edge_weight = edge_weight.to(fund_features.device)
                h_mp = self.whg_fund.encoders[mp_name](fund_features, edge_index, edge_weight)
                embeddings_list.append(h_mp)
            if not hasattr(self, 'temporal_encoders'):
                self.temporal_encoders = nn.ModuleDict({
                    'MP_temp_self': GraphSAGEEncoder(fund_features.shape[1], self.whg_fund.hid_dim, 2),
                    'MP_temp_stock': GraphSAGEEncoder(fund_features.shape[1], self.whg_fund.hid_dim, 2),
                    'MP_temp_cross': GraphSAGEEncoder(fund_features.shape[1], self.whg_fund.hid_dim, 2),
                }).to(fund_features.device)
            for temp_mp_name, (edge_index, edge_weight) in temporal_metapath_graphs.items():
                edge_index = edge_index.to(fund_features.device)
                edge_weight = edge_weight.to(fund_features.device)
                h_temp_mp = self.temporal_encoders[temp_mp_name](fund_features, edge_index, edge_weight)
                embeddings_list.append(h_temp_mp)
            fused_emb, _ = self.whg_fund.semantic_attn(embeddings_list)
            z = fused_emb

        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        return z

    def decode_nclf(self, z):
        out = self.predictor(z)
        # Ensure (N, 1) for regression and guard against accidental transposes
        if out.dim() == 2:
            if out.size(0) < out.size(1):
                out = out.t()
            if out.size(1) != 1:
                out = out.mean(dim=1, keepdim=True)
        return out
    
    def _pathsim_to_graph(self, sim_matrix, threshold):
        """
        Convert PathSim similarity matrix to edge_index and edge_weight.
        
        Args:
            sim_matrix: (num_funds, num_funds) similarity matrix
            threshold: similarity threshold for edge creation
        
        Returns:
            (edge_index, edge_weight) tuple
        """
        # Threshold to reduce graph size
        mask = sim_matrix > threshold
        
        # Convert to edge_index and edge_weight
        edge_index = mask.nonzero(as_tuple=False).t()  # (2, num_edges)
        edge_weight = sim_matrix[mask]  # (num_edges,)
        
        return (edge_index, edge_weight)
    
    def _extract_relation_matrices(self, graph):
        """
        Extract relation matrices from HeteroData graph.
        
        Uses existing edge weights from the graph:
        - Weighted edges (holds_stock, holds_other): Use holding amounts as edge weights
        - Unweighted edges (managed_by): Set weight=1.0
        
        Args:
            graph: HeteroData with edge_index for different edge types
        
        Returns:
            dict mapping relation_name -> (num_funds, num_entities) matrix
        """
        relation_matrices = {}
        
        # Extract holds_stock relation (WEIGHTED - use holding amounts)
        if ('fund', 'holds_stock', 'stock') in graph.edge_types:
            edge_index = graph[('fund', 'holds_stock', 'stock')].edge_index
            num_funds = graph['fund'].num_nodes
            num_stocks = graph['stock'].num_nodes
            
            fund_stock_matrix = torch.zeros(num_funds, num_stocks, device=edge_index.device)
            
            # Use edge_attr (holding amounts) if available, otherwise default to 1.0
            if hasattr(graph[('fund', 'holds_stock', 'stock')], 'edge_attr'):
                edge_weights = graph[('fund', 'holds_stock', 'stock')].edge_attr
                if edge_weights.dim() > 1:
                    edge_weights = edge_weights.squeeze(-1)
            else:
                edge_weights = torch.ones(edge_index.shape[1], device=edge_index.device)
            
            fund_stock_matrix[edge_index[0], edge_index[1]] = edge_weights
            relation_matrices['holds_stock'] = fund_stock_matrix
        
        # Extract holds_other relation (WEIGHTED - use holding amounts)
        if ('fund', 'holds_other', 'other_asset') in graph.edge_types:
            edge_index = graph[('fund', 'holds_other', 'other_asset')].edge_index
            num_funds = graph['fund'].num_nodes
            num_other = graph['other_asset'].num_nodes
            
            fund_other_matrix = torch.zeros(num_funds, num_other, device=edge_index.device)
            
            # Use edge_attr (holding amounts) if available, otherwise default to 1.0
            if hasattr(graph[('fund', 'holds_other', 'other_asset')], 'edge_attr'):
                edge_weights = graph[('fund', 'holds_other', 'other_asset')].edge_attr
                if edge_weights.dim() > 1:
                    edge_weights = edge_weights.squeeze(-1)
            else:
                edge_weights = torch.ones(edge_index.shape[1], device=edge_index.device)
            
            fund_other_matrix[edge_index[0], edge_index[1]] = edge_weights
            relation_matrices['holds_other'] = fund_other_matrix
        
        # Extract managed_by relation (UNWEIGHTED - set weight=1.0)
        if ('fund', 'managed_by', 'manager') in graph.edge_types:
            edge_index = graph[('fund', 'managed_by', 'manager')].edge_index
            num_funds = graph['fund'].num_nodes
            num_managers = graph['manager'].num_nodes
            
            fund_manager_matrix = torch.zeros(num_funds, num_managers, device=edge_index.device)
            # Set weight=1.0 for all fund-manager edges (reasonable for unweighted edges)
            fund_manager_matrix[edge_index[0], edge_index[1]] = 1.0
            relation_matrices['managed_by'] = fund_manager_matrix
        
        return relation_matrices

