import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import Linear
from torch.nn import LayerNorm


class HLinear(nn.Module):
    def __init__(self, out_dim, metadata, act="tanh"):
        super(HLinear, self).__init__()
        self.out_dim = out_dim
        node_types = metadata[0]
        self.adapt_ws = nn.ModuleDict()
        for nt in node_types:
            self.adapt_ws[nt] = Linear(-1, out_dim)
        if act == "tanh":
            self.act = torch.tanh
        elif act == "relu":
            self.act = F.relu
        elif act == "None":
            self.act = lambda x: x
        else:
            raise NotImplementedError(f"Unknown HLinear activation {act}")

    def __getitem__(self, index):
        return self.adapt_ws[index]

    def reset_parameters(self):
        for k, lin in self.adapt_ws.items():
            lin.reset_parameters()

    def forward(self, x_dict, *args, **kwargs):
        y_dict = {}
        for nt in x_dict:
            y_dict[nt] = self.act(self.adapt_ws[nt](x_dict[nt]))
        return y_dict


class FeatEmbed(nn.Module):
    def __init__(self, dataset, emb_types, embed_dim):
        super(FeatEmbed, self).__init__()
        embeds = nn.ModuleDict()
        # Allow dataset to be either a HeteroData or a dataset wrapper (e.g., FundsUniDataset)
        def _global_num_nodes(tp):
            # Prefer unified sizes if available (FundsUniDataset)
            try:
                if hasattr(dataset, 'datas') and len(dataset.datas) > 0 and (tp in dataset.datas[0].node_types):
                    return int(dataset.datas[0][tp].num_nodes)
            except Exception:
                pass
            # Fallback to HeteroData store
            try:
                return int(dataset[tp].x.shape[0])
            except Exception:
                return 0

        for tp in emb_types:
            num_nodes = _global_num_nodes(tp)
            if num_nodes <= 0:
                num_nodes = 1
            embeds[tp] = torch.nn.Embedding(num_nodes, embed_dim)
        self.embeds = embeds
        # Per-type fallback vectors (not used unless we add bounds checks)
        self.type_vecs = nn.ParameterDict({tp: nn.Parameter(torch.zeros(embed_dim)) for tp in emb_types})

    def reset_parameters(self):
        for tp, emb in self.embeds.items():
            emb.reset_parameters()

    def forward(self, x_dict):
        y_dict = {}
        for tp in x_dict:
            if tp in self.embeds:
                x = x_dict[tp]
                # For specified embed types, always use a learned embedding per node
                N = x.shape[0]
                idx = torch.arange(N, device=x.device, dtype=torch.long)
                # If N exceeds vocabulary due to runtime unification, expand embedding on-the-fly
                if idx.max().item() >= self.embeds[tp].num_embeddings:
                    with torch.no_grad():
                        old = self.embeds[tp]
                        new_num = int(idx.max().item()) + 1
                        new_emb = torch.nn.Embedding(new_num, old.embedding_dim).to(old.weight.device)
                        # copy existing weights
                        new_emb.weight[: old.num_embeddings].copy_(old.weight)
                        # kaiming init for new rows
                        if new_num > old.num_embeddings:
                            torch.nn.init.normal_(new_emb.weight[old.num_embeddings :], mean=0.0, std=0.02)
                        self.embeds[tp] = new_emb
                y = self.embeds[tp](idx)
                y_dict[tp] = y
            else:
                y_dict[tp] = x_dict[tp]
        return y_dict


class HLayerNorm(nn.Module):
    def __init__(self, out_dim, metadata):
        super(HLayerNorm, self).__init__()
        self.out_dim = out_dim
        node_types = metadata[0]
        self.hfuncs = nn.ModuleDict()
        for nt in node_types:
            self.hfuncs[nt] = LayerNorm(out_dim)

    def __getitem__(self, index):
        return self.hfuncs[index]

    def reset_parameters(self):
        for k, func in self.hfuncs.items():
            func.reset_parameters()

    def forward(self, x_dict, *args, **kwargs):
        y_dict = {}
        for nt in x_dict:
            y_dict[nt] = self.hfuncs[nt](x_dict[nt])
        return y_dict
