"""HGT+EW variant with log1p-transformed edge weights (Strategy C of the
A+C edge-weight design experiment).

Subclasses :class:`HGTEW` and only overrides :meth:`_encode_snapshot` to
apply ``torch.log1p`` to every per-edge weight tensor before it is passed
to the EW conv layers. Compresses the dynamic range of holding-percent
weights so that a few large positions no longer dominate the message
sums.

Registered as model name ``"HGT+EWlog"`` in load_model.py.
"""

import torch

from .HGT_EW import HGTEW


class HGTEWLog(HGTEW):
    """HGT+EW with ``log1p`` applied to edge weights before message passing."""

    def _encode_snapshot(self, data):
        x_dict = self.featemb(data.x_dict)
        edge_index_dict = data.edge_index_dict

        edge_time_dict = None
        if self.convs[0].use_RTE:
            et = {}
            for etype in edge_index_dict:
                if hasattr(data[etype], 'edge_time'):
                    et[etype] = data[etype].edge_time.squeeze(-1)
            if et:
                edge_time_dict = et

        edge_attr_dict = {}
        for etype in edge_index_dict:
            store = data[etype]
            if (hasattr(store, 'edge_attr')
                    and store.edge_attr is not None
                    and store.edge_attr.numel() > 0):
                ea = store.edge_attr.float()
                if ea.dim() > 1:
                    ea = ea.squeeze(-1)
                edge_attr_dict[etype] = torch.log1p(torch.clamp(ea, min=0.0))

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(
                x_dict, edge_index_dict, edge_time_dict,
                edge_attr_dict=edge_attr_dict,
            )
            if self.drop is not None:
                for k in x_dict:
                    if x_dict[k] is not None:
                        x_dict[k] = self.drop(x_dict[k])

        return x_dict
