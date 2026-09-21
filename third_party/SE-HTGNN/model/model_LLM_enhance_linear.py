import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch.nn import GRU
from torch_geometric.nn.norm import LayerNorm
from collections import defaultdict
from model.LightGCN import GraphConv


class Dyanmicatt(nn.Module):
    def __init__(self,n_inp, n_hid, layer=1):
        super(Dyanmicatt, self).__init__()
        self.gru = GRU(n_inp, 1, layer, batch_first=True)   # batch_first=True

    def forward(self, x, mask):
        # Ensure input and hidden state live on the same device and dtype to avoid
        # runtime errors when the model is moved to GPU from outside.
        if mask is not None:
            if hasattr(mask, "device"):
                x = x.to(mask.device)
            if hasattr(mask, "dtype"):
                x = x.to(mask.dtype)
        out, _ = self.gru(x, mask)
        mask = torch.split(out.mean(0), 1, dim=0)
        dict_mask = {f"t{i}":mask[i].squeeze() for i in range(len(mask))} #(1)[0]

        return dict_mask


class linearproj(nn.Module):
    def __init__(self, n_hid, timeframe):
        super(linearproj, self).__init__()
        self.project = nn.Linear(len(timeframe), 1)

    def forward(self, h):
        out = self.project(h).mean(-1)
        return out


class HTGNNLayer(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_heads: int, timeframe: list, norm: bool, device: torch.device, dropout: float,LLM_feature):
        """

        :param graph    : dgl.DGLGraph, a heterogeneous graph
        :param n_inp    : int         , input dimension
        :param n_hid    : int         , hidden dimension
        :param n_heads  : int         , number of attention heads
        :param timeframe: list        , list of time slice
        :param norm     : bool        , use LayerNorm or not
        :param device   : torch.device, gpu
        :param dropout  : float       , dropout rate
        """
        super(HTGNNLayer, self).__init__()
        
        self.n_inp     = n_inp
        self.n_hid     = n_hid
        self.n_heads   = n_heads
        self.timeframe = timeframe
        self.norm      = norm
        self.dropout   = dropout
        self.GRUlayer = 1
        # intra reltion aggregation modules
        self.LLM_features = LLM_feature
        self.intra_rel_agg = GraphConv(norm='right',activation= nn.ELU())

        # dynamic-attention
        self.predict = nn.ModuleDict({
            etype[0:-(len(etype.split('_')[-1]) + 1)]: Dyanmicatt(n_inp, n_hid, self.GRUlayer)
            for srctype, etype, dsttype in graph.canonical_etypes
        })

        if norm:
            self.norm_layer = nn.ModuleDict({ntype: LayerNorm(n_hid,) for ntype in graph.ntypes})


    def forward(self, graph: dgl.DGLGraph, node_features: dict, init_parameter):
        """

        :param graph: dgl.DGLGraph
        :param node_features: dict, {'ntype': {'ttype': features}}
        :return: output_features: dict, {'ntype': {'ttype': features}}
        """
        init_attention= init_parameter

        # same type neighbors aggregation
        # intra_features, dict, {'ttype': {(stype, etype, dtype): features}}
        intra_features = dict({ttype:{} for ttype in self.timeframe})
        h_mask = {etype[0:-(len(etype.split('_')[-1])+1)]: [] for stype, etype, dtype in graph.canonical_etypes}

        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            ttype = etype.split('_')[-1]
            reltype = etype[0:-(len(ttype)+1)]
            ew = rel_graph.edata.get("_ew", None)
            dst_feat = self.intra_rel_agg(rel_graph, (node_features[stype][ttype], node_features[dtype][ttype]), edge_weight=ew)
            intra_features[ttype][(stype, etype, dtype)] = dst_feat
            h_mask[reltype].append(dst_feat)

        # attention_coefficient predict
        # Align the initial GRU hidden state device/dtype with the stacked input
        # features and ensure contiguity (GRU requires hx to be contiguous).
        for key in list(h_mask.keys()):
            x_seq = torch.stack(h_mask[key], dim=1)
            h0 = init_attention[key].expand(self.GRUlayer, x_seq.size(0), 1)
            h0 = h0.to(x_seq.device).to(x_seq.dtype).contiguous()
            h_mask[key] = self.predict[key](x_seq, h0)

        # dynamic-attention-based fusion
        # inter_features, dict, {'ntype': {ttype: features}}
        inter_features = dict({ntype:{} for ntype in graph.ntypes})
        for ttype in intra_features.keys():
            for ntype in graph.ntypes:
                types_features = []
                types_weight = []
                for stype, etype, dtype in intra_features[ttype]:
                    if ntype == dtype:
                        reltype = etype[0:-(len(ttype) + 1)]
                        weight = h_mask[reltype][ttype]
                        types_weight.append(weight)
                        types_features.append(intra_features[ttype][(stype, etype, dtype)])
                out_feat = []
                #weight softmax
                types_weight = F.softmax(torch.stack(types_weight, dim=0), dim=0)
                for i in range(len(types_features)):
                    out_feat.append(types_features[i]*types_weight[i])
                out_feat = sum(out_feat)
                inter_features[ntype][ttype] = self.norm_layer[ntype](out_feat) if self.norm else (out_feat+node_features[ntype][ttype])


        return inter_features


class LLM4init(nn.Module):

    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, LLM_feature, device):
        super(LLM4init, self).__init__()
        self.device = device
        self.nid = n_hid
        self.consistent = nn.Linear(n_inp, n_hid)
        # self.k_w = nn.Linear(n_hid, n_hid, bias=False)
        # self.q_w = nn.Linear(n_hid, n_hid, bias=False)
        self.LLM_feature = LLM_feature
        self.reltype = []
        self.Relation_feat_init(graph)

    def Relation_feat_init(self, graph):
        for stype, etype, dtype in graph.canonical_etypes:
            ttype = etype.split('_')[-1]
            reltype = etype[0:-(len(ttype) + 1)]
            if (stype, reltype, dtype) not in self.reltype:
                self.reltype.append((stype, reltype, dtype))

    def forward(self):
        feature_dict = {}
        for key, feature in self.LLM_feature.items():
            feature = feature.to(self.device)
            feature_dict[key] = feature
            # feature_dict[key] = self.consistent(feature)

        grouped_edges = defaultdict(list)
        for stype, etype, dtype in  self.reltype:
            # stype_feature = self.k_w(feature_dict[stype])
            # dtype_feature = self.q_w(feature_dict[dtype])
            stype_feature = feature_dict[stype]
            dtype_feature = feature_dict[dtype]
            inner_product = torch.dot(stype_feature.squeeze(), dtype_feature.squeeze())
            grouped_edges[dtype].append((inner_product, stype, etype, dtype))

        normalized_inner_products = {}
        for dtype, edges in grouped_edges.items():
            # Ensure the small inner-product tensor is float32 on the target
            # device so log/softmax are always implemented, even when the
            # underlying LLM features are stored in float16.
            inner_products = torch.stack([edge[0] for edge in edges])
            inner_products = inner_products.to(self.device).float()
            softmax_weights = F.softmax(torch.log(inner_products), dim=0)
            for i, (inner_product, stype, etype, dtype) in enumerate(edges):
                normalized_inner_products[etype] = softmax_weights[i]
                # normalized_inner_products[etype] = 0.5

        return normalized_inner_products,  feature_dict


class SEHTGNN(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_layers: int, n_heads: int, time_window: int, norm: bool, device: torch.device, dropout: float = 0.2, LLM_feature = None, inp_list = None):
        """

        :param graph      : dgl.DGLGraph, a dgl heterogeneous graph
        :param n_inp      : int         , input dimension
        :param n_hid      : int         , hidden dimension
        :param n_layers   : int         , number of stacked layers
        :param n_heads    : int         , number of attention heads

        :param time_window: int         , number of timestamps
        :param norm       : bool        , use Norm or not
        :param device     : torch.device, gpu
        :param dropout    : float       , dropout rate
        """
        super(SEHTGNN, self).__init__()

        self.n_inp     = n_inp
        self.n_hid     = n_hid
        self.n_layers  = n_layers
        self.n_heads   = n_heads
        self.timeframe = [f't{_}' for _ in range(time_window)]
        self.drop = nn.Dropout(dropout)
        self.norm = norm

        # feature projector
        if inp_list is None:
            self.adaption_layer = nn.ModuleDict({ntype: nn.Linear(n_inp, n_hid) for ntype in graph.ntypes})
        else:
            self.adaption_layer = nn.ModuleDict({ntype: nn.Linear(inp_list[ntype], n_hid) for ntype in graph.ntypes})

        # LLM-enhanced prompt
        self.LLM_init = LLM4init(graph, 4096, n_hid, LLM_feature,device)

        # Dynamic-attention-based graph learning
        self.gnn_layers = nn.ModuleDict({str(i): HTGNNLayer(graph, n_hid, n_hid, n_heads, self.timeframe, norm, device, dropout, LLM_feature) for i in range(self.n_layers)})

        # Linear projection
        self.LinearProj = linearproj(n_hid, self.timeframe)


    def forward(self, graph: dgl.DGLGraph, predict_type: str):
        """
        :param graph       : dgl.DGLGraph, a dgl heterogeneous graph
        :param predict_type: str         , predicted node type
        """
        # LLM-enhanced initialization
        init_attention, init_temporal = self.LLM_init()

        # Heterogeneous feature projection & drop
        # inp_feat: dict, {'ntype': {'ttype': features}}
        inp_feat = {}
        for ntype in graph.ntypes:
            inp_feat[ntype] = {}
            for ttype in self.timeframe:
                inp_feat[ntype][ttype] = self.adaption_layer[ntype](self.drop(graph.nodes[ntype].data[ttype]),)
        spatial_feat = inp_feat

        # Dynamic attention-based spatial learning
        for step in range(self.n_layers):
            spatial_feat = self.gnn_layers[str(step)](graph, spatial_feat, init_attention)

        # Linear project
        if predict_type == "all":
            out_dict = {}
            for ntype in graph.ntypes:
                # Reuse the same LinearProj (which is type-agnostic: hidden -> hidden)
                # input to LinearProj: (N, hid, T) -> it performs weighted sum over T
                z_ntype = self.LinearProj(torch.stack([spatial_feat[ntype][ttype] for ttype in self.timeframe], dim=2))
                out_dict[ntype] = z_ntype
            return out_dict
        else:
            out_feat = self.LinearProj(torch.stack([spatial_feat[predict_type][ttype] for ttype in self.timeframe],dim=2) )
            return out_feat



class NodePredictor(nn.Module):
    def __init__(self, n_inp: int, n_classes: int):
        """
    
        :param n_inp      : int, input dimension
        :param n_classes  : int, number of classes
        """
        super().__init__()

        self.fc1 = nn.Linear(n_inp, n_inp)
        self.fc2 =  nn.Linear(n_inp, n_classes)

    def forward(self, node_feat: torch.tensor):
        """
        
        :param node_feat: torch.tensor
        """

        node_feat = F.relu(self.fc1(node_feat))
        # pred = F.relu(self.fc2(node_feat))
        pred = self.fc2(node_feat)

        return pred
