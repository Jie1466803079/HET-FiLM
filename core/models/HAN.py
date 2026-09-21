import torch
import torch.nn as nn
from torch_geometric.nn import HANConv

from core.models.ew_utils import use_ew_message, build_edge_weight_dict


class EdgeWeightHANConv(HANConv):
    """HANConv that scales per-metapath messages by per-edge weights.

    HANConv.forward loops edge types in edge_index_dict order, calling
    propagate once per type; the propagate hook pops the matching weight
    tensor so message() can scale without copying the forward loop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ew_queue = None
        self._current_ew = None

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None, *args, **kwargs):
        if edge_weight_dict is not None:
            self._ew_queue = [edge_weight_dict.get(et) for et in edge_index_dict.keys()]
        else:
            self._ew_queue = None
        self._current_ew = None
        try:
            return super().forward(x_dict, edge_index_dict, *args, **kwargs)
        finally:
            self._ew_queue = None
            self._current_ew = None

    def propagate(self, edge_index, **kwargs):
        if self._ew_queue is not None:
            self._current_ew = self._ew_queue.pop(0)
        return super().propagate(edge_index, **kwargs)

    def message(self, x_j, alpha_i, alpha_j, index, ptr, size_i):
        out = super().message(x_j, alpha_i, alpha_j, index, ptr, size_i)
        w = self._current_ew
        if w is not None:
            out = out * w.view(-1, 1)
        return out


class HAN(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        num_layers,
        metadata,
        predict_type,
        heads=8,
        dropout=0.6,
        featemb=None,
        nclf_linear=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self._use_ew_message = use_ew_message()
        conv_cls = EdgeWeightHANConv if self._use_ew_message else HANConv
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = conv_cls(
                -1, hidden_channels, heads=heads, dropout=dropout, metadata=metadata
            )
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)
        self.predict_type = predict_type
        self.featemb = featemb if featemb else lambda x: x

        self.nclf = nclf_linear

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        out = x_dict
        for i in range(self.num_layers):
            if edge_weight_dict is not None:
                out = self.convs[i](out, edge_index_dict, edge_weight_dict=edge_weight_dict)
            else:
                out = self.convs[i](out, edge_index_dict)

        predict_type = self.predict_type
        if isinstance(predict_type, list):
            out = [self.lin(out[predict_type[0]]), self.lin(out[predict_type[1]])]
        else:
            out = self.lin(out[predict_type])

        return out

    def encode(self, data, *args, **kwargs):
        x = self.featemb(data.x_dict)
        e = data.edge_index_dict
        if self._use_ew_message:
            return self.forward(x, e, edge_weight_dict=build_edge_weight_dict(data, e))
        return self.forward(x, e)

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf(z)
        return out
