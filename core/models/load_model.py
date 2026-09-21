from .GCN import GCN
from .GAT import GAT
from .RGCN import RGCN
from .HGT import HGT
from .DyHATR import DyHATR
from .HTGNN import HTGNN
from .DHSpace import DHSpace, DHNet
from .DHSpaceGRU import DHSpaceGRU, DHNetGRU
from .DHSpaceGR import DHSpaceGR
from .DHSpaceEW import DHSpaceEW
from .DHSpaceMP import DHSpaceMP
from .DHSpaceMPNAS import DHSpaceMPNAS
from .DHSpaceMGNAS import DHSpaceMGNAS
from .DHSpaceNodeGRU import DHSpaceNodeGRU
from .DHSpaceMeta import DHSpaceMeta
from .DHSpaceMGCell import DHSpaceMGCell
from .DHSpaceMGGlobal import DHSpaceMGGlobal
from .WHGFund import WHGFundTemporalWrapper
from .HLinear import FeatEmbed
from torch import nn
import torch
import json
import os
import sys
import dgl
from ..utils import count_parameters, cnt2str

from ..utils import setup_seed


class SEHTGNNFundsWrapper(nn.Module):
    """
    Thin adapter to plug the SE-HTGNN LLM-enhanced backbone into the existing
    CORE Funds training pipeline.

    - Exposes encode()/decode_nclf() so the nreg trainer can call it.
    - On each encode() call, it builds a DGL heterograph from the temporal
      PyG HeteroData snapshots in `support` and forwards through SEHTGNN.
    """

    def __init__(self, backbone, nclf_linear, predict_type, time_window):
        super().__init__()
        self.backbone = backbone
        self.nclf_linear = nclf_linear
        self.predict_type = predict_type
        self.time_window = time_window

    def _build_dgl_graph(self, support):
        """
        Construct a DGL heterograph with time-sliced edge types and per-time
        node features from a list of PyG HeteroData snapshots.
        """
        # Dynamic case: list/tuple of HeteroData; static: single HeteroData
        if isinstance(support, (list, tuple)):
            pyg_list = list(support)
        else:
            pyg_list = [support]

        hetero_dict = {}
        for t, hg in enumerate(pyg_list):
            for (src_type, rel, dst_type) in hg.edge_types:
                edge_index = hg[(src_type, rel, dst_type)].edge_index
                src = edge_index[0].cpu()
                dst = edge_index[1].cpu()
                etype_name = f"{rel}_t{t}"
                hetero_dict[(src_type, etype_name, dst_type)] = (src, dst)

        # Use node counts from the first snapshot
        num_nodes_dict = {
            ntype: int(pyg_list[0][ntype].num_nodes)
            for ntype in pyg_list[0].node_types
        }
        g = dgl.heterograph(hetero_dict, num_nodes_dict=num_nodes_dict)

        # Attach edge weights from PyG edge_attr (skip if DISABLE_EDGE_WEIGHT_GCN=1)
        if os.environ.get("DISABLE_EDGE_WEIGHT_GCN", "0") != "1":
            for t, hg in enumerate(pyg_list):
                for (src_type, rel, dst_type) in hg.edge_types:
                    store = hg[(src_type, rel, dst_type)]
                    if hasattr(store, "edge_attr") and store.edge_attr is not None and store.edge_attr.numel() > 0:
                        ew = store.edge_attr.float().cpu()
                        if ew.dim() > 1:
                            ew = ew.squeeze(-1)
                        etype_name = f"{rel}_t{t}"
                        g.edges[src_type, etype_name, dst_type].data["_ew"] = ew

        # Attach per-time features t0..t{T-1}
        import torch as _torch

        for t, hg in enumerate(pyg_list):
            for ntype in g.ntypes:
                if ntype in hg.node_types and hasattr(hg[ntype], "x"):
                    # Move features to CPU to match the graph's device (graph will be moved to GPU later)
                    feat = hg[ntype].x
                    if hasattr(feat, "device") and feat.device.type != "cpu":
                        feat = feat.cpu()
                    g.nodes[ntype].data[f"t{t}"] = feat
                else:
                    # Fallback to zeros if this type is absent at this time
                    # Use t0 dimension where available; otherwise 1-dim stub.
                    existing = g.nodes[ntype].data.get("t0", None)
                    feat_dim = existing.shape[1] if existing is not None else 1
                    g.nodes[ntype].data[f"t{t}"] = _torch.zeros(
                        g.num_nodes(ntype), feat_dim
                    )
        return g

    def encode(self, support):
        # Infer device from backbone parameters
        device = next(self.backbone.parameters()).device
        g = self._build_dgl_graph(support).to(device)
        # When predict_type is a list (joint training), forward with "all"
        # and return a list of embeddings in the same order.
        if isinstance(self.predict_type, list):
            z_dict = self.backbone(g, "all")
            return [z_dict[pt] for pt in self.predict_type]
        else:
            return self.backbone(g, self.predict_type)

    def decode_nclf(self, z):
        return self.nclf_linear(z)


def load_pre_post(args, dataset):
    feat_hid_dim = args.hid_dim
    if args.dataset == "Aminer":
        feat_hid_dim = 32 if args.homo else args.hid_dim
        featemb = FeatEmbed(dataset.dataset, "author venue".split(), feat_hid_dim)
        nclf_linear = None
    elif args.dataset == "Ecomm":
        featemb = FeatEmbed(dataset.dataset, "user item".split(), feat_hid_dim)
        nclf_linear = None
    elif args.dataset == "Yelp-nc":
        featemb = None
        nclf_linear = nn.Linear(args.hid_dim, args.num_classes)
    elif args.dataset == "covid":
        from core.trainer.nreg import NodePredictor

        featemb = None
        nclf_linear = NodePredictor(n_inp=8, n_classes=1)
    elif args.dataset == "Funds":
        from core.trainer.nreg import NodePredictor

        # Optionally add learnable embeddings for selected node types.
        # Default: ON for Funds so that sparse/zero-feature types (other_asset, manager)
        # get a trainable embedding even when raw features are missing.
        env_force = os.environ.get("FUNDS_LEARNABLE_FEATS", "1").strip().lower()
        use_learnable = getattr(args, 'use_learnable_feats', False) or env_force in ("1", "true", "yes", "y")
        featemb = None
        if use_learnable:
            # Embed only 'other_asset' and 'manager' if present; keep 'fund' and 'stock' numeric
            embed_types = [t for t in ['other_asset', 'manager'] if t in dataset.metadata[0]]
            if len(embed_types) > 0:
                featemb = FeatEmbed(dataset.dataset, embed_types, feat_hid_dim)
        fund_feat_hid_dim = 16 if feat_hid_dim == -1 else feat_hid_dim
        nclf_linear = NodePredictor(n_inp=fund_feat_hid_dim, n_classes=1)
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")
    return featemb, nclf_linear


def load_backbone(args, dataset, featemb, nclf_linear):
    in_dim, hid_dim, out_dim = args.in_dim, args.hid_dim, args.out_dim
    n_layers, metadata, predict_type, n_heads, time_window, device, norm = (
        args.n_layers,
        dataset.metadata,
        args.predict_type,
        args.n_heads,
        args.twin,
        args.device,
        args.norm,
    )
    dhconfig = args.dhconfig
    model = args.model
    if model == "GCN":
        from .GCN import GCN as Net

        model = Net(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
    elif model == "GAT":
        from .GAT import GAT as Net

        model = Net(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            heads=n_heads,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
    elif model == "SAGE":
        from .SAGE import SAGE as Net

        model = Net(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
    elif model == "RGCN":
        from .RGCN import RGCN as Net

        model = Net(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
    elif model == "HGT":
        from .HGT import HGT as Net

        model = Net(
            hidden_channels=hid_dim,
            out_channels=out_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
        if hasattr(args, 'dropout'):
            try:
                model.drop.p = args.dropout
            except Exception:
                pass
    elif model == "HGT+":
        # When --use_tcmp is set, swap the standard HGT for the TCMP variant
        # (per-layer FiLM modulation on fund features). Default off — existing
        # PBSes are unaffected.
        if getattr(args, 'use_tcmp', False):
            from .HGT_TCMP import HGT_TCMP as Net
            model = Net(
                hidden_channels=hid_dim,
                out_channels=out_dim,
                num_heads=n_heads,
                num_layers=n_layers,
                metadata=metadata,
                predict_type=predict_type,
                use_RTE=True,
                featemb=featemb,
                nclf_linear=nclf_linear,
                tcmp_text_dim=int(getattr(args, 'tcmp_text_dim', 128)),
            )
        else:
            from .HGT import HGT as Net
            model = Net(
                hidden_channels=hid_dim,
                out_channels=out_dim,
                num_heads=n_heads,
                num_layers=n_layers,
                metadata=metadata,
                predict_type=predict_type,
                use_RTE=True,
                featemb=featemb,
                nclf_linear=nclf_linear,
            )
        if hasattr(args, 'dropout'):
            try:
                model.drop.p = args.dropout
            except Exception:
                pass
    elif model == "HGT+EW":
        from .HGT_EW import HGTEW as Net

        model = Net(
            hidden_channels=hid_dim,
            out_channels=out_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            use_RTE=True,
            featemb=featemb,
            nclf_linear=nclf_linear,
            dropout=getattr(args, 'dropout', 0.0),
            time_window=time_window,
        )
    elif model == "HGT+EWlog":
        from .HGT_EW_log import HGTEWLog as Net

        model = Net(
            hidden_channels=hid_dim,
            out_channels=out_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            use_RTE=True,
            featemb=featemb,
            nclf_linear=nclf_linear,
            dropout=getattr(args, 'dropout', 0.0),
            time_window=time_window,
        )
    elif model == "HGT+EWattn":
        from .HGT_EW_attn import HGTEWAttn as Net

        model = Net(
            hidden_channels=hid_dim,
            out_channels=out_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            use_RTE=True,
            featemb=featemb,
            nclf_linear=nclf_linear,
            dropout=getattr(args, 'dropout', 0.0),
            time_window=time_window,
        )
    elif model == "HGT+EWlogPlus":
        from .HGT_EW_log_plus import HGTEWLogPlus as Net

        model = Net(
            hidden_channels=hid_dim,
            out_channels=out_dim,
            num_heads=n_heads,
            num_layers=n_layers,
            metadata=metadata,
            predict_type=predict_type,
            use_RTE=True,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
        if hasattr(args, 'dropout'):
            try:
                model.drop.p = args.dropout
            except Exception:
                pass
    elif model == "DyHATR":
        from .DyHATR import DyHATR as Net

        model = Net(
            in_dim,
            hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            time_window=time_window,
            metadata=metadata,
            predict_type=predict_type,
            dropout=getattr(args, 'dropout', 0.2),
            edge_layers=n_layers,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=getattr(args, 'hlinear_act', 'tanh'),
        )  # use edge_layers=2,n_heads=4,dropout=0.2 in original code.
        if hasattr(args, 'dropout'):
            for layer in model.gnn_layers:
                if hasattr(layer, 'post_rel_dropout'):
                    layer.post_rel_dropout.p = args.dropout
                if hasattr(layer, 'post_time_dropout'):
                    layer.post_time_dropout.p = args.dropout
    elif model == "HTGNN":
        from .HTGNN import HTGNN as Net

        model = Net(
            hid_dim,
            hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            time_window=time_window,
            norm=False,
            metadata=metadata,
            device=device,
            predict_type=predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
        )
        if hasattr(args, 'dropout'):
            for layer in model.gnn_layers:
                if hasattr(layer, 'post_rel_dropout'):
                    layer.post_rel_dropout.p = args.dropout
        # use n_heads=1,n_layers=2,norm=True in original code. but layernorm can cause very low performance. This may be due to embedding layer.
    elif model == "SEHTGNN_LLM":
        # LLM-enhanced SE-HTGNN model using precomputed node-type embeddings.
        # This path is intended for the Funds dataset and plugs into the
        # standard nreg trainer via the SEHTGNNFundsWrapper above.
        # The original SE-HTGNN repo lives under mutual_fund_prediction/SE-HTGNN
        # (sibling of CRSP_code_v4), and exposes the model as
        # model.model_LLM_enhance_linear.SEHTGNN.
        # Walk up: .../CRSP_code_v4/CORE/core/models ->
        #          .../CRSP_code_v4/CORE/dhgas ->
        #          .../CRSP_code_v4/CORE ->
        #          .../CRSP_code_v4 ->
        #          .../mutual_fund_prediction
        _models_dir = os.path.dirname(__file__)
        _dhgas_dir = os.path.dirname(_models_dir)
        _dhgas_root = os.path.dirname(_dhgas_dir)
        _crsp_root = os.path.dirname(_dhgas_root)
        _mf_root = os.path.dirname(_crsp_root)
        se_root = os.path.join(_mf_root, "SE-HTGNN")
        if os.path.isdir(se_root) and se_root not in sys.path:
            sys.path.insert(0, se_root)
        try:
            from model.model_LLM_enhance_linear import SEHTGNN  # type: ignore
        except Exception as e:
            raise ImportError(
                f"Failed to import SEHTGNN from SE-HTGNN repo at {se_root}. "
                f"Please ensure the repository is cloned there."
            ) from e

        llm_path = getattr(args, 'llm_feature_path', None)
        if llm_path is None:
            raise ValueError(
                "SEHTGNN_LLM requires --llm_feature_path pointing to funds_llm_features_llama3_8b.pt"
            )
        llm_feature = torch.load(llm_path, map_location=device)
        # Align LLM_feature keys with Funds node types.
        # The Funds graph uses 'other_asset' as the node type, while the
        # prompt/embedding job may have stored this under the shorter key
        # 'other'. Create an alias so LLM4init can look up by dtype name.
        try:
            ntypes, _ = dataset.metadata
        except Exception:
            ntypes = []
        if "other_asset" in ntypes and "other_asset" not in llm_feature:
            if "other" in llm_feature:
                llm_feature["other_asset"] = llm_feature["other"]

        # Build SEHTGNN on top of a representative DGL heterograph template.
        # FundsUniDataset attaches base_graph_dgl during construction with
        # time-sliced node features t0..t{T-1}. We derive per-node-type input
        # dimensions from these features so that the SEHTGNN adaption_layer
        # has the correct in_features for each type.
        try:
            graph_dgl = dataset.base_graph_dgl
        except Exception as e:
            raise ValueError(
                "SEHTGNN_LLM expects dataset.base_graph_dgl to be set "
                "to a DGL heterograph for Funds."
            ) from e

        # Infer per-node-type feature dimensions from t0 features
        inp_list = {}
        for ntype in graph_dgl.ntypes:
            feat = None
            # Prefer t0; fall back to any t*-key if needed
            if "t0" in graph_dgl.nodes[ntype].data:
                feat = graph_dgl.nodes[ntype].data["t0"]
            else:
                for k, v in graph_dgl.nodes[ntype].data.items():
                    if k.startswith("t"):
                        feat = v
                        break
            if feat is not None and hasattr(feat, "shape"):
                if feat.dim() == 2:
                    inp_list[ntype] = int(feat.size(1))
                else:
                    # Flatten non-2D just in case
                    inp_list[ntype] = int(feat.view(feat.size(0), -1).size(1))
            else:
                # Fallback: use fund embedding dim if available, else 1
                inp_list[ntype] = int(llm_feature.get("fund", torch.zeros(4096)).shape[0])

        # n_inp is only used when inp_list is None; keep a sensible scalar
        # (e.g., max over per-type dims) but rely on inp_list for each type.
        scalar_n_inp = max(inp_list.values()) if len(inp_list) > 0 else llm_feature["fund"].shape[0]

        backbone = SEHTGNN(
            graph=graph_dgl,
            n_inp=scalar_n_inp,
            n_hid=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            time_window=time_window,
            norm=False,
            device=device,
            dropout=getattr(args, "dropout", 0.2),
            LLM_feature=llm_feature,
            inp_list=inp_list,
        )
        # Wrap SEHTGNN so it exposes encode()/decode_nclf() for the trainer.
        model = SEHTGNNFundsWrapper(
            backbone=backbone,
            nclf_linear=nclf_linear,
            predict_type=predict_type,
            time_window=time_window,
        )
    elif model == "ETTE":
        from .ETTE import ETTE as Net

        model = Net(
            args=args,
            metadata=metadata,
        )
        # ETTE already uses args.dropout internally
    elif model == "DHSpaceMPNAS":
        from .DHSpace import DHNet as Net

        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                dhspaces.append(
                    DHSpaceMPNAS(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                        mp_budget=getattr(args, 'mp_budget', 2),
                        hard_topk=bool(getattr(args, 'mp_hard_topk', False)),
                        tau=float(getattr(args, 'mp_tau', 1.0)),
                    ).assign_basic_arch(["causal", "node_hetero"])
                )

            dhspaces.append(
                DHSpaceMPNAS(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=10,
                    K_N=K_N,
                    K_R=K_R,
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                    mp_budget=getattr(args, 'mp_budget', 2),
                    hard_topk=bool(getattr(args, 'mp_hard_topk', False)),
                    tau=float(getattr(args, 'mp_tau', 1.0)),
                ).assign_basic_arch(["last", "node_hetero"])
            )
        else:
            cfg = torch.load(os.path.join(dhconfig, "config"))
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for a in cfg:
                dhspace = DHSpaceMPNAS(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info["KTO"],
                    K_N=info["KN"],
                    K_R=info["KR"],
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                    mp_budget=getattr(args, 'mp_budget', 2),
                    hard_topk=bool(getattr(args, 'mp_hard_topk', False)),
                    tau=float(getattr(args, 'mp_tau', 1.0)),
                ).assign_arch(a)
                dhspaces.append(dhspace)

        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=getattr(args, 'hlinear_act', 'tanh'),
        )
    elif model == "DHSpaceMGNAS":
        from .DHSpace import DHNet as Net

        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                dhspaces.append(
                    DHSpaceMGNAS(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                        mg_budget=getattr(args, 'mg_budget', 2),
                    ).assign_basic_arch(["causal", "node_hetero"])
                )
            dhspaces.append(
                DHSpaceMGNAS(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=10,
                    K_N=K_N,
                    K_R=K_R,
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                    mg_budget=getattr(args, 'mg_budget', 2),
                    hard_topk=bool(getattr(args, 'mg_hard_topk', False)),
                    tau=float(getattr(args, 'mg_tau', 1.0)),
                ).assign_basic_arch(["last", "node_hetero"])
            )
        else:
            cfg = torch.load(os.path.join(dhconfig, "config"))
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for a in cfg:
                dhspace = DHSpaceMGNAS(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info["KTO"],
                    K_N=info["KN"],
                    K_R=info["KR"],
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                    mg_budget=getattr(args, 'mg_budget', 2),
                ).assign_arch(a)
                dhspaces.append(dhspace)

        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=getattr(args, 'hlinear_act', 'tanh'),
        )
    elif model == "XGBoost":
        from .XGBoost import XGBoostWrapper as Net

        # Get XGBoost-specific parameters from args or use defaults
        xgb_params = {}
        if hasattr(args, 'xgb_max_depth'):
            xgb_params['max_depth'] = args.xgb_max_depth
        if hasattr(args, 'xgb_learning_rate'):
            xgb_params['learning_rate'] = args.xgb_learning_rate
        if hasattr(args, 'xgb_n_estimators'):
            xgb_params['n_estimators'] = args.xgb_n_estimators
        if hasattr(args, 'xgb_subsample'):
            xgb_params['subsample'] = args.xgb_subsample
        if hasattr(args, 'xgb_colsample_bytree'):
            xgb_params['colsample_bytree'] = args.xgb_colsample_bytree
        if hasattr(args, 'xgb_min_child_weight'):
            xgb_params['min_child_weight'] = args.xgb_min_child_weight
        if hasattr(args, 'xgb_gamma'):
            xgb_params['gamma'] = args.xgb_gamma
        if hasattr(args, 'xgb_reg_alpha'):
            xgb_params['reg_alpha'] = args.xgb_reg_alpha
        if hasattr(args, 'xgb_reg_lambda'):
            xgb_params['reg_lambda'] = args.xgb_reg_lambda

        model = Net(
            n_inp=in_dim,
            n_hid=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            time_window=time_window,
            norm=norm,
            metadata=metadata,
            device=device,
            predict_type=predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            **xgb_params
        )
    elif model == "DHSpaceNodeGRU":
        from .DHSpace import DHNet as Net

        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                dhspaces.append(
                    DHSpaceNodeGRU(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["causal", "node_hetero"])
                )

            dhspaces.append(
                DHSpaceNodeGRU(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=10,
                    K_N=K_N,
                    K_R=K_R,
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                ).assign_basic_arch(["last", "node_hetero"])
            )
        else:
            cfg = torch.load(os.path.join(dhconfig, "config"))
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for a in cfg:
                dhspace = DHSpaceNodeGRU(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info["KTO"],
                    K_N=info["KN"],
                    K_R=info["KR"],
                    rel_time_type=info["rel_time_type"],
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                    args=args,
                )
                dhspace.assign_arch(a)
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
        # Optional diversity regularizer for MGCell/MG-like models
        try:
            model.mg_div_lambda = float(getattr(args, 'mg_div_lambda', 0.0))
        except Exception:
            pass
    elif "DHSpace_GRU" in model:
        from .DHSpace_GRU import DHNet_GRU as Net
        from .DHSpace_GRU import DHSpace_GRU

        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                if model == "DHSpace_GRU":
                    dhspaces.append(
                        DHSpace_GRU(
                            hid_dim,
                            metadata,
                            time_window,
                            K_To=10,
                            K_N=K_N,
                            K_R=K_R,
                            n_heads=n_heads,
                            norm=norm,
                            args=args,
                            use_gru=True,
                        ).assign_basic_arch(["causal", "node_hetero"])
                    )
                elif model == "DHSpace_GRU_Bi":
                    dhspaces.append(
                        DHSpace_GRU(
                            hid_dim,
                            metadata,
                            time_window,
                            K_To=10,
                            K_N=K_N,
                            K_R=K_R,
                            n_heads=n_heads,
                            norm=norm,
                            args=args,
                            use_gru=True,
                            gru_bidirectional=True,
                        ).assign_basic_arch(["causal", "node_hetero"])
                    )

            if model == "DHSpace_GRU":
                dhspaces.append(
                    DHSpace_GRU(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                        use_gru=True,
                    ).assign_basic_arch(["last", "node_hetero"])
                )
            elif model == "DHSpace_GRU_Bi":
                dhspaces.append(
                    DHSpace_GRU(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                        use_gru=True,
                        gru_bidirectional=True,
                    ).assign_basic_arch(["last", "node_hetero"])
                )
        else:
            cfg = torch.load(
                os.path.join(dhconfig, "config")
            )  # config only determines Ato,AN,AR
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for a in cfg:
                dhspace = DHSpace_GRU(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info["KTO"],
                    K_N=info["KN"],
                    K_R=info["KR"],
                    rel_time_type=info["rel_time_type"],
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                    use_gru=True,
                )
                dhspace.assign_arch(a)
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
    elif model == "WHGFund":
        # WHGFund: Weighted Heterogeneous Graph for Fund Performance Prediction
        # Adapted from WHGDroid paper (Huang et al., 2023)
        # Extended with TEMPORAL METAPATHS for time-series fund data
        # Infer fund input feature dimension from validation sample
        fund_in_dim = in_dim
        try:
            sample = dataset.val_dataset[0]
            # For link task, val_dataset returns (graph, eval_data) tuple
            if isinstance(sample, tuple) and len(sample) == 2:
                g = sample[0]  # Extract graph from tuple
            elif isinstance(sample, (list, tuple)) and len(sample) > 0:
                g = sample[-1]
            else:
                g = sample
            # Check if graph has fund node type and extract dimension
            if hasattr(g, 'node_types') and 'fund' in g.node_types and hasattr(g['fund'], 'x'):
                fund_in_dim = g['fund'].x.size(1)
        except Exception:
            pass
        # Force regression head to output 1-dim per node
        model = WHGFundTemporalWrapper(
            in_dim=fund_in_dim,  # use actual fund feature dimension
            hid_dim=hid_dim,
            out_dim=1,
            num_layers=n_layers,
            metapaths=['MP1', 'MP2', 'MP3'],  # fund-stock-fund, fund-other-fund, fund-manager-fund
            pathsim_threshold=getattr(args, 'pathsim_threshold', 0.1),
            temporal_mode=getattr(args, 'temporal_mode', 'last'),  # 'last', 'mean', 'all', 'cross_time_attention', or 'temporal_metapath'
            n_heads=n_heads,  # Same n_heads as DHSpace/DyHATR for temporal self-attention
            time_window=time_window,  # Same time_window as DHSpace/DyHATR for position embeddings
            device=device
        )
        # Set temporal decay if using temporal_metapath mode
        if hasattr(args, 'temporal_decay'):
            model.temporal_decay = args.temporal_decay
    elif model == "DHSpaceGRU":
        from .DHSpaceGRU import DHNetGRU as Net
        from .DHSpaceGRU import DHSpaceGRU

        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                dhspaces.append(
                    DHSpaceGRU(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["causal", "node_hetero"])
                )

            dhspaces.append(
                DHSpaceGRU(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=10,
                    K_N=K_N,
                    K_R=K_R,
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                ).assign_basic_arch(["last", "node_hetero"])
            )
        else:
            cfg = torch.load(
                os.path.join(dhconfig, "config")
            )  # config only determines Ato,AN,AR
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for a in cfg:
                dhspace = DHSpaceGRU(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info["KTO"],
                    K_N=info["KN"],
                    K_R=info["KR"],
                    rel_time_type=info["rel_time_type"],
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                    args=args,
                )
                dhspace.assign_arch(a)
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
    elif model == "DHSpaceMGGlobal":
        from .DHSpaceSearch import DHNet as Net
        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            # Only add meta relations for predict_type targets when possible
            meta_targets = []
            pt = predict_type
            if isinstance(pt, (list, tuple)):
                meta_targets = [t for t in pt if t in metadata[0]]
            elif isinstance(pt, str) and pt in metadata[0]:
                meta_targets = [pt]
            if not meta_targets:
                meta_targets = list(metadata[0])
            K_R = num_relations + len(meta_targets)
            for i in range(n_layers - 1):
                dhspaces.append(
                    DHSpaceMGGlobal(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["causal", "node_hetero"])
                )
            dhspaces.append(
                DHSpaceMGGlobal(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=10,
                    K_N=K_N,
                    K_R=K_R,
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                ).assign_basic_arch(["last", "node_hetero"])
            )
        else:
            cfg = torch.load(os.path.join(dhconfig, "config"))
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for idx, a in enumerate(cfg):
                dhspace = DHSpaceMGGlobal(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info.get("KTO", 10),
                    K_N=info.get("KN", len(metadata[0])),
                    K_R=info.get("KR", len(metadata[1])),
                    rel_time_type=info.get("rel_time_type", "relative"),
                    time_patch_num=(info.get("patch_num", 1) if idx < n_layers - 1 else 1),
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                    args=args,
                )
                dhspace.assign_arch(a)
                # Sanitize AN to fit current time_patch_num*num_types rows to avoid out-of-range n_alpha indexing
                try:
                    AN = dhspace.A[1]
                    max_rows = dhspace.time_patch_num * dhspace.num_types
                    if AN.max().item() >= max_rows:
                        AN.clamp_(min=0, max=max_rows - 1)
                        dhspace.A[1] = AN
                except Exception:
                    pass
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
    elif model == "DHSpaceMGCell":
        from .DHSpaceSearch import DHNet as Net
        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                dhspaces.append(
                    DHSpaceMGCell(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["causal", "node_hetero"])
                )
            dhspaces.append(
                DHSpaceMGCell(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=10,
                    K_N=K_N,
                    K_R=K_R,
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                ).assign_basic_arch(["last", "node_hetero"])
            )
        else:
            cfg = torch.load(os.path.join(dhconfig, "config"))
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for a in cfg:
                dhspace = DHSpaceMGCell(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info.get("KTO", 10),
                    K_N=info.get("KN", len(metadata[0])),
                    K_R=info.get("KR", len(metadata[1])),
                    rel_time_type=info.get("rel_time_type", "relative"),
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                    args=args,
                )
                dhspace.assign_arch(a)
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
    elif model == "DHSpaceMeta":
        from .DHSpaceSearch import DHNet as Net
        # Parse meta-paths from args (format: "rel1+rel2;rel3+rel4")
        mp_str = getattr(args, 'meta_paths', '')
        meta_paths = []
        if isinstance(mp_str, str) and mp_str.strip():
            for seg in mp_str.split(';'):
                hops = [h.strip() for h in seg.split('+') if h.strip()]
                if hops:
                    meta_paths.append(hops)
        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                dhspaces.append(
                    DHSpaceMeta(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                        rel_time_type=args.rel_time_type if hasattr(args, 'rel_time_type') else 'relative',
                        time_patch_num=getattr(args, 'patch_num', 1),
                        meta_paths=meta_paths,
                    ).assign_basic_arch(["causal", "node_hetero"])
                )
            dhspaces.append(
                DHSpaceMeta(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=10,
                    K_N=K_N,
                    K_R=K_R,
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                    rel_time_type="relative",
                    time_patch_num=1,
                    meta_paths=meta_paths,
                ).assign_basic_arch(["last", "node_hetero"])
            )
        else:
            cfg = torch.load(
                os.path.join(dhconfig, "config")
            )  # config only determines ATo,AN,AR
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            # Rebuild each space with the same patching as in search: first (n_layers-1) use info["patch_num"], last uses 1
            for idx, a in enumerate(cfg):
                dhspace = DHSpaceMeta(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info.get("KTO", 10),
                    K_N=info.get("KN", len(metadata[0])),
                    K_R=info.get("KR", len(metadata[1])),
                    rel_time_type=info.get("rel_time_type", "relative"),
                    time_patch_num=(info.get("patch_num", 1) if idx < n_layers - 1 else 1),
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                    args=args,
                    meta_paths=meta_paths,
                )
                dhspace.assign_arch(a)
                # Sanitize AN to fit current time_patch_num*num_types rows to avoid out-of-range n_alpha indexing
                try:
                    AN = dhspace.A[1]
                    max_rows = dhspace.time_patch_num * dhspace.num_types
                    if AN.max().item() >= max_rows:
                        AN.clamp_(min=0, max=max_rows - 1)
                        dhspace.A[1] = AN
                except Exception:
                    pass
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
    elif model == "DHSpaceGR":
        from .DHSpaceGRU import DHNetGRU as Net

        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                dhspaces.append(
                    DHSpaceGR(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["causal", "node_hetero"])
                )

            dhspaces.append(
                DHSpaceGR(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=10,
                    K_N=K_N,
                    K_R=K_R,
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                ).assign_basic_arch(["last", "node_hetero"])
            )
        else:
            cfg = torch.load(os.path.join(dhconfig, "config"))
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for a in cfg:
                dhspace = DHSpaceGR(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info["KTO"],
                    K_N=info["KN"],
                    K_R=info["KR"],
                    rel_time_type=info["rel_time_type"],
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                    args=args,
                )
                dhspace.assign_arch(a)
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
    elif model == "DHSpaceEW":
        from .DHSpaceGRU import DHNetGRU as Net

        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                dhspaces.append(
                    DHSpaceEW(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["causal", "node_hetero"])
                )

            dhspaces.append(
                DHSpaceEW(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=10,
                    K_N=K_N,
                    K_R=K_R,
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                ).assign_basic_arch(["last", "node_hetero"])
            )
        else:
            cfg = torch.load(os.path.join(dhconfig, "config"))
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for a in cfg:
                dhspace = DHSpaceEW(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info["KTO"],
                    K_N=info["KN"],
                    K_R=info["KR"],
                    rel_time_type=info["rel_time_type"],
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                    args=args,
                )
                dhspace.assign_arch(a)
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
    elif model == "DHSpaceMP":
        # Use the standard DHNet wrapper; internally DHSpaceMP contains a DHSpace backbone
        from .DHSpace import DHNet as Net
        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                dhspaces.append(
                    DHSpaceMP(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                        mp_channels=getattr(args, 'mp_channels', 4),
                    ).assign_basic_arch(["causal", "node_hetero"])
                )
            dhspaces.append(
                DHSpaceMP(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=10,
                    K_N=K_N,
                    K_R=K_R,
                    n_heads=n_heads,
                    norm=norm,
                    args=args,
                    mp_channels=getattr(args, 'mp_channels', 4),
                ).assign_basic_arch(["last", "node_hetero"])
            )
        else:
            cfg = torch.load(os.path.join(dhconfig, "config"))
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            for a in cfg:
                dhspace = DHSpaceMP(
                    hid_dim,
                    metadata,
                    time_window,
                    K_To=info["KTO"],
                    K_N=info["KN"],
                    K_R=info["KR"],
                    rel_time_type=info["rel_time_type"],
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                    args=args,
                    mp_channels=getattr(args, 'mp_channels', 4),
                )
                dhspace.assign_arch(a)
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
    elif "DHSpace" in model:
        from .DHSpace import DHNet as Net
        from .DHSpace import DHSpace

        dhspaces = []
        if not dhconfig:
            num_types, num_relations = len(metadata[0]), len(metadata[1])
            K_N = num_types
            K_R = num_relations
            for i in range(n_layers - 1):
                if model == "DHSpace":
                    dhspaces.append(
                        DHSpace(
                            hid_dim,
                            metadata,
                            time_window,
                            K_To=10,
                            K_N=K_N,
                            K_R=K_R,
                            n_heads=n_heads,
                            norm=norm,
                            args=args,
                        ).assign_basic_arch(["causal", "node_hetero"])
                    )
                elif model == "DHSpaceS":
                    dhspaces.append(
                        DHSpace(
                            hid_dim,
                            metadata,
                            time_window,
                            K_To=10,
                            K_N=K_N,
                            K_R=K_R,
                            n_heads=n_heads,
                            norm=norm,
                            args=args,
                        ).assign_basic_arch(["causal"])
                    )
                elif model == "DHSpaceF":
                    dhspaces.append(
                        DHSpace(
                            hid_dim,
                            metadata,
                            time_window,
                            K_To=10,
                            K_N=K_N,
                            K_R=K_R,
                            n_heads=n_heads,
                            norm=norm,
                            args=args,
                        ).assign_basic_arch(["full", "node_hetero"])
                    )
                elif model == "DHSpaceH":
                    dhspaces.append(
                        DHSpace(
                            hid_dim,
                            metadata,
                            time_window,
                            K_To=10,
                            K_N=K_N,
                            K_R=K_R,
                            n_heads=n_heads,
                            norm=norm,
                            args=args,
                        ).assign_basic_arch(["full", "node_hetero", "rel_hetero"])
                    )

            if model == "DHSpace":
                dhspaces.append(
                    DHSpace(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["last", "node_hetero"])
                )
            elif model == "DHSpaceS":
                dhspaces.append(
                    DHSpace(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["last"])
                )
            elif model == "DHSpaceF":
                dhspaces.append(
                    DHSpace(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["last", "node_hetero"])
                )
            elif model == "DHSpaceH":
                dhspaces.append(
                    DHSpace(
                        hid_dim,
                        metadata,
                        time_window,
                        K_To=10,
                        K_N=K_N,
                        K_R=K_R,
                        n_heads=n_heads,
                        norm=norm,
                        args=args,
                    ).assign_basic_arch(["last", "node_hetero", "rel_hetero"])
                )
        else:
            cfg = torch.load(
                os.path.join(dhconfig, "config")
            )  # config only determines Ato,AN,AR
            info = json.load(open(os.path.join(dhconfig, "supernet.json")))
            # Use the search config's twin so architecture tensors match
            search_twin = info.get("twin", time_window)
            for a in cfg:
                dhspace = DHSpace(
                    hid_dim,
                    metadata,
                    search_twin,
                    K_To=info["KTO"],
                    K_N=info["KN"],
                    K_R=info["KR"],
                    rel_time_type=info["rel_time_type"],
                    n_heads=n_heads,
                    norm=norm,
                    hupdate=True,
                )
                dhspace.assign_arch(a)
                dhspaces.append(dhspace)
        model = Net(
            hid_dim,
            search_twin if dhconfig else time_window,
            metadata,
            dhspaces,
            predict_type,
            featemb=featemb,
            nclf_linear=nclf_linear,
            hlinear_act=args.hlinear_act,
        )
    else:
        raise NotImplementedError(f"Unexpected model {model}")
    return model


def load_lazy_hetero_weights(args, dataset, model):
    # Skip initialization for XGBoost (non-neural network model)
    if args.model == "XGBoost":
        return

    with torch.no_grad():  # Initialize lazy modules.
        # For DHSpaceMeta/MGGlobal, guard against n_alpha index issues during one-pass init by forcing fix_N
        is_meta = getattr(args, 'model', '') in ('DHSpaceMeta', 'DHSpaceMGGlobal')
        saved_fixN = []
        if is_meta and hasattr(model, 'spaces'):
            for sp in model.spaces:
                saved_fixN.append(getattr(sp, 'fix_N', False))
                try:
                    sp.fix_N = True
                except Exception:
                    pass
        try:
            if args.dataset in "Aminer Ecomm".split():
                out = model.encode(dataset.val_dataset[0])
            elif args.dataset in "Yelp-nc".split():
                out = model.encode(dataset.val_dataset[0][0])
            elif args.dataset in "covid".split():
                # For covid dataset, get the full support data (list of temporal graphs)
                support_data = dataset.val_dataset[0]  # This is the list of temporal graphs
                if hasattr(model, 'timeframe') and isinstance(support_data, list):
                    # For temporal models like HTGNN, convert list to dict indexed by time
                    support_dict = {i: support_data[i] for i in range(len(support_data))}
                    out = model.encode(support_dict)
                elif isinstance(support_data, list) and len(support_data) > 0:
                    # For non-temporal models, just use the last snapshot
                    out = model.encode(support_data[-1])
                else:
                    out = model.encode(support_data)
            elif args.dataset in "Funds".split():
                # Handle Funds dataset - for link task, val_dataset returns list of tuples
                val_sample = dataset.val_dataset[0]
                if isinstance(val_sample, tuple) and len(val_sample) == 2:
                    # Link task: (graph, eval_data) -> extract graph
                    support_data = val_sample[0]
                else:
                    # Regression task: just the graph
                    support_data = val_sample
                out = model.encode(support_data)
        finally:
            if is_meta and hasattr(model, 'spaces') and saved_fixN:
                for sp, old in zip(model.spaces, saved_fixN):
                    try:
                        sp.fix_N = old
                    except Exception:
                        pass


def load_model(args, dataset):
    setup_seed(args.seed)
    featemb, nclf_linear = load_pre_post(args, dataset)
    model = load_backbone(args, dataset, featemb, nclf_linear)
    load_lazy_hetero_weights(args, dataset, model)
    return model
