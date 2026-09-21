import torch as th
from torch import nn
# from torch.nn import init
from dgl import function as fn
from dgl.utils import expand_as_pair


class GraphConv(nn.Module):

    def __init__(
        self,
        norm="both",
        activation=None,
    ):
        super(GraphConv, self).__init__()

        self._norm = norm
        self._activation = activation

    def forward(self, graph, feat, edge_weight=None):

        with graph.local_scope():

            if edge_weight is not None:
                graph.edata["_ew"] = edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight
                aggregate_fn = fn.u_mul_e("h", "_ew", "m")
            else:
                aggregate_fn = fn.copy_u("h", "m")
            feat_src, feat_dst = expand_as_pair(feat, graph)

            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)+1
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg="m", out="h_neigh"))
            rst = feat_dst+graph.dstdata["h_neigh"]
            # rst = graph.dstdata["h_neigh"]

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1) +1
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm


            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)

if __name__ == "__main__":
    import torch,dgl

    g = dgl.heterograph({
        ('author', 'writes', 'paper'): [(0, 0), (0,1),(1, 1), (2, 2)],  # author -> paper
        # ('paper', 'cites', 'paper'): [(0, 1), (1, 2)],  # paper -> paper
    })

    # 为节点分配特征
    g.nodes['author'].data['feat'] = torch.FloatTensor([[1],[2],[3]])  # author 节点 16 维特征
    g.nodes['paper'].data['feat'] = torch.FloatTensor([[1],[2],[3]])  # paper 节点 32 维特征

    gcn_layer = GraphConv(norm="right", activation=None, )
    paper = gcn_layer(g, (g.nodes['author'].data['feat'], g.nodes['paper'].data['feat']))

    print("Updated paper node features:\n", paper)
    # print("Updated paper node features:\n", paper_feats)