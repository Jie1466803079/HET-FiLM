"""
DHSpaceEW: DHSpace variant that removes all GRUs and uses raw edge weights
for weighted relations during message passing.

Behavior:
- No NodeGRU (no temporal smoothing of node features)
- No EdgeGRU (no gating from weight history)
- For weighted edges (relations whose name contains 'holds_'), scale relation
  messages by the current edge weight (taken as the last entry of the provided
  edge weight window from support.edge_features['<rel_key>']['window']).
- For unweighted edges, use pass-through (gate=1.0).

This class builds on DHSpaceGR (which integrates the attention pipelines and
edge-level gating hooks) by overriding compute_edge_gates to eliminate GRUs
and consume the current edge weights directly.
"""

from torch import nn
import torch

from .DHSpaceGR import DHSpaceGR


class DHSpaceEW(DHSpaceGR):
    """DHSpace with raw edge-weight scaling, no GRUs (node or edge)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure no node GRUs are present (already empty in DHSpaceGR)
        self.node_grus = nn.ModuleDict()
        # Remove any edge GRUs/head MLPs inherited from DHSpaceGRU lineage
        # (DHSpaceGR sets up edge_grus via its parent; clear them here)
        if hasattr(self, 'edge_grus'):
            self.edge_grus = nn.ModuleDict()
        if hasattr(self, 'edge_gate_heads'):
            self.edge_gate_heads = nn.ModuleDict()
        if hasattr(self, 'edge_weight_heads'):
            self.edge_weight_heads = nn.ModuleDict()
        if hasattr(self, 'edge_zi_heads'):
            self.edge_zi_heads = nn.ModuleDict()

        # Storage for externally provided edge features
        self.edge_features = {}

    def set_edge_features(self, edge_features):
        """Accept external per-edge features from the trainer/dataset."""
        self.edge_features = edge_features or {}

    def compute_edge_gates(self, rel_key, edge_index, t_src=None):
        """
        Return per-edge gates. For weighted edges ("holds_*"), use the latest
        edge weight from the provided window. For unweighted edges, return ones.

        Args:
            rel_key: relation identifier string
            edge_index: tensor [2, E] with the indices used this step
            t_src: unused (kept for signature compatibility)
        """
        if edge_index is None:
            return None, None, None
        E = edge_index.size(1)
        device = edge_index.device

        # Weighted relations: use current weight (last column of window)
        if 'holds_' in rel_key:
            feats = self.edge_features.get(rel_key, None)
            if feats is None or 'window' not in feats:
                return torch.ones(E, device=device), None, None
            window = feats['window'].to(device)
            # window shape: [E, L] or [E, L, 1]; take the most recent value
            if window.dim() == 3:
                window = window.squeeze(-1)
            current = window[:, -1]
            # Ensure non-negative scaling and clamp to reasonable band
            # (holdings can be 0-heavy; add small epsilon to avoid all zeros)
            current = torch.relu(current)
            # Optional normalization: map to [0.02, 0.98] band
            if current.numel() > 0:
                # Prevent div by zero
                denom = (current.max() - current.min()).clamp_min(1e-6)
                scaled = (current - current.min()) / denom
                gates = 0.02 + 0.96 * scaled
            else:
                gates = torch.ones(E, device=device)
            return gates, None, None

        # Unweighted relations: pass-through
        return torch.ones(E, device=device), None, None
