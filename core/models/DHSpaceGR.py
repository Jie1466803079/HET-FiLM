"""
DHSpaceGR: DHSpace with EdgeGRU gating only (no NodeGRU).

This variant mirrors DHSpaceGRU's edge-history gating for weighted edges
but removes the per-node temporal GRU preprocessing. Feature sequences are
fed directly to the DHSpace attention as in the base model.
"""

from torch import nn
import torch

from .DHSpaceGRU import DHSpaceGRU


class DHSpaceGR(DHSpaceGRU):
    """Disable NodeGRU while keeping EdgeGRU gating."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove node GRUs to avoid extra params; keep edge GRUs intact
        self.node_grus = nn.ModuleDict()

    def forward(self, xs, graphs):
        """
        Forward pass without NodeGRU preprocessing.

        Args:
            xs: list of x_dict, one per time step [time, ntype] -> [N, F]
            graphs: list of graph objects, one per time step

        Returns:
            x_res: list of updated node representations [time, ntype] -> [N, F]
            aux_outputs_all: dict of auxiliary predictions for loss computation
        """
        twin = self.twin
        device = xs[0][self.id2ntype[0]].device

        # No NodeGRU: pass features through as-is
        x_win = xs
        x_res = []
        aux_outputs_all = {}

        ATo, _, _ = self.A
        ATo = ATo.to(device)
        for t_tar in range(twin):
            ATo_tar = ATo[t_tar]
            if ATo_tar.sum() == 0:
                x_dict = x_win[t_tar]
            else:
                topos = []
                x_tar = x_win[t_tar]
                for t_src in range(twin):
                    if ATo_tar[t_src].sum() == 0:
                        continue
                    graph_src = graphs[t_src]
                    x_src = x_win[t_src]
                    for rel in ATo_tar[t_src].nonzero():
                        nsrc, rel, ntar = self.id2etype[rel]
                        ei_rel = graph_src[rel].edge_index
                        x_tar_rel = x_tar[ntar].index_select(self.node_dim, ei_rel[1, :])
                        x_src_rel = x_src[nsrc].index_select(self.node_dim, ei_rel[0, :])
                        ei_rel_tar = ei_rel[1, :].T
                        topo = (x_tar_rel, x_src_rel, t_tar, t_src, rel, ei_rel_tar, ei_rel)
                        topos.append(topo)
                x_dict, aux_outputs = self.DHAttnOne2Multi(x_tar, topos)

                for rel_key in aux_outputs:
                    if rel_key not in aux_outputs_all:
                        aux_outputs_all[rel_key] = {'w_pred': [], 'zi_pred': []}
                    aux_outputs_all[rel_key]['w_pred'].append(aux_outputs[rel_key]['w_pred'])
                    aux_outputs_all[rel_key]['zi_pred'].append(aux_outputs[rel_key]['zi_pred'])

                if self.norm:
                    x_dict = self.update_norm(x_dict)
            x_res.append(x_dict)

        for rel_key in aux_outputs_all:
            if aux_outputs_all[rel_key]['w_pred']:
                aux_outputs_all[rel_key]['w_pred'] = torch.cat(aux_outputs_all[rel_key]['w_pred'])
                aux_outputs_all[rel_key]['zi_pred'] = torch.cat(aux_outputs_all[rel_key]['zi_pred'])

        return x_res, aux_outputs_all

