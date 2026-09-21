"""
DHSpaceNodeGRU: DHSpace backbone with temporal node-level GRU pre-processing.

This variant keeps the NodeGRU enhancement from DHSpaceGRU but drops the
EdgeGRU machinery so edge weights are handled identically to the vanilla
DHSpace model.
"""

from torch import nn
import torch

from .DHSpace import DHSpace


class DHSpaceNodeGRU(DHSpace):
    """Apply a GRU cell per node type before running the DHSpace attention."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_grus = nn.ModuleDict(
            {ntype: nn.GRUCell(self.hid_dim, self.hid_dim) for ntype in self.node_types}
        )

    def reset_parameters(self):
        super().reset_parameters()
        for gru in self.node_grus.values():
            gru.reset_parameters()

    def forward(self, xs, graphs):
        """
        Args:
            xs: list of dictionaries {node_type: features} for each time step.
            graphs: list of PyG HeteroData objects per time step.
        """
        h_states = {ntype: None for ntype in self.node_types}
        processed = []

        for t in range(self.twin):
            x_t = xs[t]
            processed_t = {}
            for ntype, gru in self.node_grus.items():
                x_node = x_t[ntype]
                if h_states[ntype] is None:
                    h_states[ntype] = torch.zeros_like(x_node)
                h_states[ntype] = gru(x_node, h_states[ntype])
                processed_t[ntype] = h_states[ntype]
            processed.append(processed_t)

        return super().forward(processed, graphs)
