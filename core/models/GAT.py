import torch
from torch_geometric.nn import GAT as PyGGAT
from torch_geometric.nn.conv import GATConv
from core.data.utils import make_hodata
from core.models.ew_utils import use_ew_message, build_edge_weight_vector


class EdgeWeightGATConv(GATConv):
    """GATConv that scales attended messages by per-edge weights.

    Attention logits are untouched (same choice as HGT+EW: values are scaled,
    not scores). GATConv appends self-loops after the original edges, so the
    weight vector is right-padded with 1.0 to match.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._edge_weight = None

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        if (edge_weight is not None and self.add_self_loops
                and isinstance(edge_index, torch.Tensor)):
            # GATConv's add_self_loops path first REMOVES numerically
            # coincident (i,i) pairs — real edges when src/dst are different
            # node types of a hetero relation — then appends loops. Mirror the
            # removal on the weight vector so it stays edge-aligned.
            mask = edge_index[0] != edge_index[1]
            if not bool(mask.all()):
                edge_weight = edge_weight[mask]
        self._edge_weight = edge_weight
        try:
            return super().forward(x, edge_index, **kwargs)
        finally:
            self._edge_weight = None

    def message(self, x_j, alpha):
        out = alpha.unsqueeze(-1) * x_j
        w = self._edge_weight
        if w is not None:
            pad = x_j.size(0) - w.numel()
            if pad > 0:
                w = torch.cat([w, w.new_ones(pad)])
            out = out * w.view(-1, 1, 1)
        return out


class EdgeWeightGAT(PyGGAT):
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(self, in_channels, out_channels, **kwargs):
        conv = super().init_conv(in_channels, out_channels, **kwargs)
        assert type(conv) is GATConv, f"expected GATConv, got {type(conv)}"
        conv.__class__ = EdgeWeightGATConv
        conv._edge_weight = None
        return conv


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        num_layers,
        metadata,
        predict_type,
        heads=8,
        dropout=0.6,
        featemb=None,
        nclf_linear=None,
    ):
        super().__init__()
        self._use_ew_message = use_ew_message()
        gnn_cls = EdgeWeightGAT if self._use_ew_message else PyGGAT
        self.gnn = gnn_cls(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            concat=False,
        )
        self.predict_type = predict_type
        self.featemb = featemb if featemb else lambda x: x
        self.nclf = nclf_linear

    def encode(self, data, *args, **kwargs):
        x_dict = self.featemb(data.x_dict)
        x, e, mask, _ = make_hodata(x_dict, data.edge_index_dict, self.predict_type)
        if self._use_ew_message:
            w = build_edge_weight_vector(data, data.edge_index_dict)
            x = self.gnn(x, e, edge_weight=w)
        else:
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
