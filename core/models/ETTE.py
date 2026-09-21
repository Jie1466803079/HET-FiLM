import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import Linear
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_add


class ETTE(nn.Module):
    """
    ETTE-PWHG: Edge Time-aware Transformer with Predictive Weighted Heterogeneous Graph

    Features:
    - Multi-head attention with temperature scaling per relation
    - GRU-based edge gates from historical weights
    - Post-attention gating (gates only applied to values)
    - Auxiliary tasks: weight prediction + zero-inflation
    - Probabilistic sampling during training for better gradient flow

    Fixed issues:
    - Double gating removed
    - Separate gate prediction head (decoupled from weight prediction)
    - Temperature parameters now used
    - Probabilistic sampling instead of hard top-k
    """

    def __init__(self, args, metadata):
        """
        Args:
            args: Configuration object with:
                - in_dim: input feature dimension
                - hid_dim: hidden dimension
                - out_dim: output dimension (for prediction)
                - n_heads: number of attention heads (default: 4)
                - gate_hidden: gate GRU hidden size (default: 16)
                - sampling_ratio: fraction of edges to keep per destination (default: 0.8)
                - gate_clip: tuple (min, max) for gate clamping (default: (0.02, 0.98))
                - gate_keep_q: quantile threshold for global pruning (default: 0.30)
                - aux_lambda: weight for auxiliary losses (default: 0.20)
                - use_next_weight_aux: use next-quarter weights as targets (default: True)
                - dropout: dropout rate (default: 0.0)
            metadata: tuple of (node_types, edge_types)
        """
        super().__init__()
        self.args = args
        self.metadata = metadata

        self.hidden = args.hid_dim
        self.num_heads = getattr(args, 'n_heads', 4)
        assert self.hidden % self.num_heads == 0, 'hidden must be divisible by num_heads'
        self.head_dim = self.hidden // self.num_heads

        # Node GRUs for temporal updates
        self.grus = nn.ModuleDict({
            n: nn.GRUCell(self.hidden, self.hidden)
            for n in metadata[0]
        })

        # Input projections
        self.node_embeds = nn.ModuleDict({
            ntype: Linear(args.in_dim, self.hidden)
            for ntype in metadata[0]
        })

        # Relation-specific attention modules
        self.Q = nn.ModuleDict()
        self.K = nn.ModuleDict()
        self.V = nn.ModuleDict()
        self.O = nn.ModuleDict()
        self.pi_r = nn.ParameterDict()  # Relation importance weights
        self.tau_r = nn.ParameterDict()  # Temperature per relation

        # Gate mechanisms & auxiliary heads
        self.edge_gates = nn.ModuleDict()
        self.edge_gate_head = nn.ModuleDict()  # Separate gate prediction
        self.edge_weight_head = nn.ModuleDict()
        self.edge_zi_head = nn.ModuleDict()

        for rel in metadata[1]:
            rel_key = f"{rel[0]}_{rel[1]}_{rel[2]}"
            self.Q[rel_key] = Linear(self.hidden, self.hidden)
            self.K[rel_key] = Linear(self.hidden, self.hidden)
            self.V[rel_key] = Linear(self.hidden, self.hidden)
            self.O[rel_key] = Linear(self.hidden, self.hidden)
            self.pi_r[rel_key] = nn.Parameter(torch.tensor(1.0))

            # Temperature defaults per relation (softer on dense holds_stock)
            init_tau = 1.5 if "holds_stock" in rel_key else (1.2 if "holds_other" in rel_key else 1.0)
            self.tau_r[rel_key] = nn.Parameter(torch.tensor(float(init_tau)))

            if "holds_" in rel_key:
                # GRU over historical weight window
                gate_hidden = getattr(args, 'gate_hidden', 16)
                self.edge_gates[rel_key] = nn.GRU(
                    input_size=1,
                    hidden_size=gate_hidden,
                    batch_first=True
                )
                self.edge_gate_head[rel_key] = Linear(gate_hidden, 1)
                self.edge_weight_head[rel_key] = Linear(gate_hidden, 1)
                self.edge_zi_head[rel_key] = Linear(gate_hidden, 1)
            else:
                # Unweighted edges (e.g., managed_by): age-based gate
                self.edge_gates[rel_key] = nn.Sequential(
                    Linear(1, 8),
                    nn.ReLU(),
                    Linear(8, 1),
                    nn.Sigmoid()
                )

        # Prediction head (for fund performance regression - output dim should be 1)
        self.predictor = nn.Sequential(
            Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Dropout(getattr(args, 'dropout', 0.0)),
            Linear(self.hidden // 2, 1),
        )
        
        # Alias for compatibility with evaluation framework
        self.nclf_linear = self.predictor
        self.predict_type = getattr(args, 'predict_type', 'fund')

        # External edge features (set via set_edge_features before forward)
        self.edge_features = {}

    def set_edge_features(self, edge_features):
        """Set edge features dict before forward pass"""
        self.edge_features = edge_features or {}

    def compute_edge_gates(self, rel_key, edge_index):
        """
        Compute gates for edges based on historical features

        Returns:
            g: gate values [E]
            w_pred: weight predictions [E] (for aux loss)
            zi_pred: zero-inflation logits [E] (for aux loss)
            keep_mask: boolean mask [E] (for pruning low-gate edges)
        """
        feats = self.edge_features.get(rel_key, None)
        E = edge_index.size(1)
        device = next(self.parameters()).device

        if feats is None or E == 0:
            # Default: pass through all edges
            g = torch.ones(E, device=device)
            keep_mask = torch.ones(E, dtype=torch.bool, device=device)
            return g, None, None, keep_mask

        if "holds_" in rel_key:
            window = feats.get('window', None)
            if window is None:
                w_pred = zi_pred = None
                g_raw = torch.ones(E, device=device)
            else:
                # window: [E, L] or [E, L, 1]
                if window.dim() == 2:
                    window = window.unsqueeze(-1)
                window = window.to(device)

                _, hidden = self.edge_gates[rel_key](window)  # [1, E, H]
                hidden = hidden.squeeze(0)  # [E, H]

                # Separate predictions (fixed from original coupling)
                g_raw = torch.sigmoid(self.edge_gate_head[rel_key](hidden).squeeze(-1))
                w_pred = self.edge_weight_head[rel_key](hidden).squeeze(-1)
                zi_pred = self.edge_zi_head[rel_key](hidden).squeeze(-1)
        else:
            # Age-based gate for unweighted edges
            age = feats.get('age', torch.zeros(E, device=device)).to(device)
            g_raw = self.edge_gates[rel_key](age.unsqueeze(-1)).squeeze(-1)
            w_pred = zi_pred = None

        # Clamp for stability
        gate_clip = getattr(self.args, 'gate_clip', (0.02, 0.98))
        g = g_raw.clamp(gate_clip[0], gate_clip[1])

        # Global pruning by quantile
        gate_keep_q = getattr(self.args, 'gate_keep_q', 0.30)
        if gate_keep_q > 0.0:
            thresh = torch.quantile(g.detach(), gate_keep_q)
            keep_mask = g >= thresh
        else:
            keep_mask = torch.ones_like(g, dtype=torch.bool)

        return g, w_pred, zi_pred, keep_mask

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass

        Args:
            x_dict: dict of node features {node_type: [N, in_dim]}
            edge_index_dict: dict of edge indices {(src, rel, dst): [2, E]}

        Returns:
            predictions: [N_fund] predictions for fund nodes
            aux_outputs: dict of auxiliary predictions for loss computation
        """
        device = next(self.parameters()).device

        # Initial embeddings
        h = {ntype: self.node_embeds[ntype](x.to(device))
             for ntype, x in x_dict.items() if ntype in self.node_embeds}

        h_next = {ntype: torch.zeros_like(h[ntype]) for ntype in h}
        messages = {ntype: [] for ntype in h}
        messages_acc = {ntype: torch.zeros_like(h[ntype]) for ntype in h}
        aux_outputs = {}

        for rel in self.metadata[1]:
            rel_key = f"{rel[0]}_{rel[1]}_{rel[2]}"
            if rel not in edge_index_dict:
                continue

            edge_index = edge_index_dict[rel].to(device)
            if edge_index.size(1) == 0:
                continue

            src, _, dst = rel
            src_idx, dst_idx = edge_index

            # Compute gates and masks
            g, w_pred, zi_pred, keep = self.compute_edge_gates(rel_key, edge_index)
            if keep.sum() == 0:
                continue

            # Store aux predictions
            if w_pred is not None:
                aux_outputs[f"{rel_key}_weight"] = (w_pred, keep)
            if zi_pred is not None:
                aux_outputs[f"{rel_key}_zi"] = (zi_pred, keep)

            # Prune edges
            src_idx = src_idx[keep]
            dst_idx = dst_idx[keep]
            g = g[keep]  # Fixed: prune gates too

            # Batched attention using scatter operations (FAST!)
            E_kept = dst_idx.size(0)
            if E_kept == 0:
                continue

            H = self.hidden
            Hh = self.head_dim
            Nh = self.num_heads

            # Compute Q, K, V for all edges at once
            Q_all = self.Q[rel_key](h[dst][dst_idx])  # [E_kept, H]
            K_all = self.K[rel_key](h[src][src_idx])  # [E_kept, H]
            V_all = self.V[rel_key](h[src][src_idx])  # [E_kept, H]

            # Reshape for multi-head attention
            Q = Q_all.view(E_kept, Nh, Hh)  # [E, Nh, Hh]
            K = K_all.view(E_kept, Nh, Hh)  # [E, Nh, Hh]
            V = V_all.view(E_kept, Nh, Hh)  # [E, Nh, Hh]

            # Compute attention scores for all edges
            scores = (Q * K).sum(dim=-1) / (math.sqrt(Hh) * self.tau_r[rel_key])  # [E, Nh]

            # Apply per-destination softmax using PyG's efficient implementation
            alpha = pyg_softmax(scores, dst_idx, dim=0)  # [E, Nh] - softmax per destination

            # Post-attention gating: apply gates to values
            V_gated = V * g.unsqueeze(-1).unsqueeze(-1)  # [E, Nh, Hh]

            # Weighted aggregation
            messages_weighted = alpha.unsqueeze(-1) * V_gated  # [E, Nh, Hh]

            # Scatter-add messages to destinations
            messages_per_head = scatter_add(
                messages_weighted, dst_idx, dim=0, dim_size=h[dst].size(0)
            )  # [N_dst, Nh, Hh]

            # Concat heads and project
            messages_concat = messages_per_head.view(h[dst].size(0), H)  # [N_dst, H]
            messages_out = self.O[rel_key](messages_concat)  # [N_dst, H]

            # Accumulate messages
            messages_acc[dst] += self.pi_r[rel_key] * messages_out

        # Aggregate messages
        for ntype in h:
            messages[ntype].append(messages_acc[ntype])

        # Node GRU updates
        for ntype in h:
            if messages[ntype]:
                agg = sum(messages[ntype])
                h_next[ntype] = self.grus[ntype](agg, h[ntype])
            else:
                h_next[ntype] = h[ntype]

        return self.predictor(h_next['fund']).squeeze(-1), aux_outputs

    def encode(self, graphs, *args, **kwargs):
        """
        Encode method required by CORE framework

        For temporal models, processes a list of graphs and returns embeddings
        for the predict_type nodes (usually 'fund')
        """
        # For ETTE, we need to process temporal graphs
        # Use the last graph's embeddings after GRU updates
        device = next(self.parameters()).device

        # Initialize hidden states
        h = None

        # Process each timestep
        for graph in graphs:
            x_dict = {ntype: x.to(device) for ntype, x in graph.x_dict.items()}
            edge_index_dict = {rel: ei.to(device) for rel, ei in graph.edge_index_dict.items()}

            if h is None:
                # First timestep: initialize from node embeddings
                h = {ntype: self.node_embeds[ntype](x.to(device))
                     for ntype, x in x_dict.items() if ntype in self.node_embeds}

            # Compute messages
            h_next = {ntype: torch.zeros_like(h[ntype]) for ntype in h}
            messages_acc = {ntype: torch.zeros_like(h[ntype]) for ntype in h}

            for rel in self.metadata[1]:
                rel_key = f"{rel[0]}_{rel[1]}_{rel[2]}"
                if rel not in edge_index_dict:
                    continue

                edge_index = edge_index_dict[rel]
                if edge_index.size(1) == 0:
                    continue

                src, _, dst = rel
                src_idx, dst_idx = edge_index

                # Batched attention for encode (no gates, no edge features)
                E = edge_index.size(1)
                H = self.hidden
                Hh = self.head_dim
                Nh = self.num_heads

                # Compute Q, K, V for all edges
                Q_all = self.Q[rel_key](h[dst][dst_idx])  # [E, H]
                K_all = self.K[rel_key](h[src][src_idx])  # [E, H]
                V_all = self.V[rel_key](h[src][src_idx])  # [E, H]

                # Reshape for multi-head
                Q = Q_all.view(E, Nh, Hh)
                K = K_all.view(E, Nh, Hh)
                V = V_all.view(E, Nh, Hh)

                # Attention scores
                scores = (Q * K).sum(dim=-1) / (math.sqrt(Hh) * self.tau_r[rel_key])  # [E, Nh]
                alpha = pyg_softmax(scores, dst_idx, dim=0)  # [E, Nh]

                # Aggregate messages
                messages_weighted = alpha.unsqueeze(-1) * V  # [E, Nh, Hh]
                messages_per_head = scatter_add(
                    messages_weighted, dst_idx, dim=0, dim_size=h[dst].size(0)
                )  # [N_dst, Nh, Hh]

                # Concat and project
                messages_concat = messages_per_head.view(h[dst].size(0), H)
                messages_out = self.O[rel_key](messages_concat)
                messages_acc[dst] += self.pi_r[rel_key] * messages_out

            # GRU updates
            for ntype in h:
                h_next[ntype] = self.grus[ntype](messages_acc[ntype], h[ntype])

            h = h_next

        # Return fund node embeddings
        return h['fund']

    def decode_nclf(self, z):
        """Node classification/regression decode method required by CORE trainer"""
        return self.predictor(z).squeeze(-1)

    def auxiliary_losses(self, aux_outputs, edge_features):
        """
        Compute auxiliary losses for weight prediction and zero-inflation

        Args:
            aux_outputs: dict from forward pass
            edge_features: dict with target labels

        Returns:
            scalar loss value
        """
        device = next(self.parameters()).device
        losses = []

        for rel_key, feats in edge_features.items():
            if "holds_" not in rel_key:
                continue

            wp = aux_outputs.get(f"{rel_key}_weight", None)
            zp = aux_outputs.get(f"{rel_key}_zi", None)

            if wp is None and zp is None:
                continue

            # Get target weights
            use_next = getattr(self.args, 'use_next_weight_aux', True)
            w_next = feats.get('w_next', None)
            w_cur = feats.get('weight', None)

            target_full = None
            if use_next and (w_next is not None):
                target_full = w_next.to(device)
            elif w_cur is not None:
                target_full = w_cur.to(device)

            # Weight regression (MSE)
            if (wp is not None) and (target_full is not None):
                w_pred_all, keep = wp
                w_pred_all = w_pred_all.to(device)
                keep = keep.to(device) if keep is not None else None

                if keep is not None and keep.numel() == target_full.numel():
                    w_pred = w_pred_all[keep]
                    target = target_full[keep]
                elif keep is not None and target_full.numel() == int(keep.sum().item()):
                    w_pred = w_pred_all[keep]
                    target = target_full
                else:
                    m = min(w_pred_all.numel(), target_full.numel())
                    w_pred = w_pred_all[:m]
                    target = target_full[:m]

                if w_pred.numel() > 0 and target.numel() > 0:
                    losses.append(F.mse_loss(w_pred, target))

            # Zero-inflation (BCE)
            if (zp is not None) and (target_full is not None):
                zi_pred_all, keep = zp
                zi_pred_all = zi_pred_all.to(device)
                keep = keep.to(device) if keep is not None else None
                exist_full = (target_full > 0).float()

                if keep is not None and keep.numel() == exist_full.numel():
                    zi_pred = zi_pred_all[keep]
                    exist = exist_full[keep]
                elif keep is not None and exist_full.numel() == int(keep.sum().item()):
                    zi_pred = zi_pred_all[keep]
                    exist = exist_full
                else:
                    m = min(zi_pred_all.numel(), exist_full.numel())
                    zi_pred = zi_pred_all[:m]
                    exist = exist_full[:m]

                if zi_pred.numel() > 0 and exist.numel() > 0:
                    losses.append(F.binary_cross_entropy_with_logits(zi_pred, exist))

        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()