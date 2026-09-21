import torch
from torch_geometric.nn import GraphSAGE
from core.data.utils import make_hodata


class SAGE(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_layers,
        metadata,
        predict_type,
        dropout=0.5,
        featemb=None,
        nclf_linear=None,
    ):
        super().__init__()
        self.gnn = GraphSAGE(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=num_layers,
            aggr='mean',
            dropout=dropout,
        )
        self.predict_type = predict_type
        self.featemb = featemb if featemb else lambda x: x
        self.nclf = nclf_linear

    def encode(self, data, *args, **kwargs):
        x_dict = self.featemb(data.x_dict)
        x, e, mask, _ = make_hodata(x_dict, data.edge_index_dict, self.predict_type)
        x = self.gnn(x, e)
        if isinstance(mask, list):
            return [x[mask[0]], x[mask[1]]]
        return x[mask]

    def decode(self, z, edge_label_index, *args, **kwargs):
        if isinstance(z, (list, tuple)):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        return self.nclf(z)
