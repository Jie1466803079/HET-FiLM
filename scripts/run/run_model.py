import sys
import os
# Remove v3 path and add v4 path at the beginning
if '/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v3/CORE' in sys.path:
    sys.path.remove('/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v3/CORE')

import torch
import torch.nn as nn
from core.data import load_data
from core.models import load_model
from core.args_model import get_args
from core.trainer import load_trainer
from core.evaluation_metrics import evaluate_dhgas_model, print_evaluation_results, save_metrics_to_csv
from core.evaluate_link import evaluate_link_dhgas
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add, scatter_logsumexp
from types import SimpleNamespace
import time

# args
args = get_args()
# If a specific time window was provided, prefer it over twin
if getattr(args, "twin", -1) == -1 and getattr(args, "time_window", -1) != -1:
    args.twin = args.time_window
# Propagate edge-delta thresholds to env for dataset preprocessing
os.environ.setdefault("EDGE_DELTA_PCT", str(getattr(args, "edge_delta_pct", 80.0)))
os.environ.setdefault("EDGE_DELTA_ABS", str(getattr(args, "edge_delta_abs", -1.0)))

# Override max_epochs for faster testing (legacy behavior for link_weight)
if getattr(args, 'task', '') == 'link_weight':
    args.max_epochs = min(args.max_epochs, 30)  # Run 30 epochs to see convergence
    print(f"[TEST MODE] Limited max_epochs to {args.max_epochs}")

# Specialized path: joint link + edge-weight multitask on Funds (mirrors run_hgt_plus_edge_weight.py)
if getattr(args, "task", "") in (
    "link_weight_multitask",
    "link_weight_multitask_new",
    "link_weight_multitask_new_twostage",
    "link_weight_multitask_new_joint",
):
    from core.data.funds_edge_weight import FundsEdgeWeightDataset
    from core.data.new_edge_query import build_new_edge_query
    from core.models.multitask_edge import MultiTaskEdgePredictor
    from core.trainer.edge_multitask import train_till_end, evaluate
    from core.utils import setup_seed
    import json

    def to_device(pairs, device):
        result = []
        for s, q in pairs:
            if isinstance(s, (list, tuple)):
                s = [g.to(device) for g in s]
            else:
                s = s.to(device)
            result.append((s, q.to(device)))
        return result

    def _filter_new_edges(pairs, neg_sampling_ratio):
        """Build a CasMLN-style new-edge split with freshly sampled negatives."""
        filtered = []
        for support, query in pairs:
            new_query = build_new_edge_query(query, neg_sampling_ratio=neg_sampling_ratio)
            if new_query is None:
                continue
            if new_query is query:
                filtered.append((support, query))
                continue
            filtered.append((support, new_query))
        return filtered

    def _filter_positive_edges(pairs):
        """Keep only positive edges (edge_label==1)."""
        filtered = []
        for support, query in pairs:
            mask = query.edge_label == 1
            if not mask.any():
                continue
            q2 = query.__class__()
            q2.edge_label_index = query.edge_label_index[:, mask]
            q2.edge_label = query.edge_label[mask]
            q2.edge_weight = query.edge_weight[mask]
            if hasattr(query, "edge_continue"):
                q2.edge_continue = query.edge_continue[mask]
            if hasattr(query, "edge_prev_weight"):
                q2.edge_prev_weight = query.edge_prev_weight[mask]
            if hasattr(query, "edge_fund_baseline") and query.edge_fund_baseline is not None:
                q2.edge_fund_baseline = query.edge_fund_baseline[mask]
            filtered.append((support, q2))
        return filtered

    def _filter_unseen_nodes(pairs, train_seen_nodes):
        """Remove edges where at least one endpoint node was unseen during training."""
        fund_seen = train_seen_nodes['fund']
        stock_seen = train_seen_nodes['stock']
        filtered = []
        for support, query in pairs:
            ei = query.edge_label_index
            mask = torch.tensor([
                int(ei[0, i]) in fund_seen and int(ei[1, i]) in stock_seen
                for i in range(ei.size(1))
            ], dtype=torch.bool)
            if not mask.any():
                continue
            q2 = query.__class__()
            q2.edge_label_index = query.edge_label_index[:, mask]
            q2.edge_label = query.edge_label[mask]
            q2.edge_weight = query.edge_weight[mask]
            if hasattr(query, "edge_continue"):
                q2.edge_continue = query.edge_continue[mask]
            if hasattr(query, "edge_prev_weight"):
                q2.edge_prev_weight = query.edge_prev_weight[mask]
            if hasattr(query, "edge_fund_baseline") and query.edge_fund_baseline is not None:
                q2.edge_fund_baseline = query.edge_fund_baseline[mask]
            filtered.append((support, q2))
        return filtered

    def _time_merge_hetero(graphs):
        """Merge list of HeteroData snapshots into one graph (DHGAS-style for static models).

        Node features from last snapshot; edge indices concatenated across all snapshots.
        With EDGE_WEIGHT_MESSAGE=1, edge_attr is concatenated in the same order
        (1.0 fallback for snapshots without it) so EW baselines see aligned weights.
        Default path is unchanged.
        """
        merge_ew = os.environ.get("EDGE_WEIGHT_MESSAGE", "0") != "0"
        if merge_ew:
            from core.models.ew_utils import snapshot_edge_attr
        merged = graphs[-1].clone()
        for etype in merged.edge_types:
            eids = []
            eattrs = []
            for g in graphs:
                if hasattr(g[etype], 'edge_index'):
                    eids.append(g[etype].edge_index)
                    if merge_ew:
                        eattrs.append(snapshot_edge_attr(g, etype))
            if eids:
                merged[etype].edge_index = torch.cat(eids, dim=1)
                if merge_ew:
                    merged[etype].edge_attr = torch.cat(eattrs)
        return merged

    class _TimeMergeWrapper(nn.Module):
        """Wraps a static model to accept list-of-snapshots by time-merging first."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def encode(self, data, *args, **kwargs):
            if isinstance(data, (list, tuple)):
                data = _time_merge_hetero(data)
            return self.model.encode(data, *args, **kwargs)

        def decode(self, *args, **kwargs):
            return self.model.decode(*args, **kwargs)

    class _LiveWrapper:
        """Per-access derivation of filtered+device-placed views over a dataset.

        Used when the underlying dataset's `train_dataset` is a `@property`
        (CasMLN-literal per-epoch resampling — see
        `core/data/funds_edge_weight_casmln.py:FundsEdgeWeightDatasetCasMLN`).
        Each access to `.train_dataset` re-runs the filter chain on a fresh
        `dataset.train_dataset` access, which fires the property and produces
        freshly sampled negatives (matching CasMLN's
        `EcommUniDataset.train_dataset` + `shift_negetive_sample` pattern).
        `val_dataset` and `test_dataset` are cached once at construction since
        CasMLN's protocol fixes them across epochs.

        For static (non-property) datasets, the existing snapshot path below
        is used instead — original baseline behavior is preserved exactly.
        """

        def __init__(self, dataset_, mode_, neg_ratio_, device_, is_new_edge_task_,
                     train_seen_nodes_):
            self._dataset = dataset_
            self._mode = mode_  # 'all' / 'new' / 'pos' / 'new_pos'
            self._neg_ratio = neg_ratio_
            self._device = device_
            self._is_new_edge_task = is_new_edge_task_
            self.train_seen_nodes = train_seen_nodes_
            # Cache val/test (CasMLN: fixed across epochs; underlying dataset's
            # val_dataset/test_dataset properties are themselves idempotent.)
            self._val_cache = self._derive('val')
            self._test_cache = self._derive('test')

        def _derive(self, split_):
            if split_ == 'train':
                pairs = self._dataset.train_dataset    # property fires → resample
            elif split_ == 'val':
                pairs = self._dataset.val_dataset
            else:
                pairs = self._dataset.test_dataset

            if self._is_new_edge_task and self._mode in ('new', 'new_pos'):
                pairs = _filter_new_edges(pairs, neg_sampling_ratio=self._neg_ratio)

            if split_ in ('val', 'test'):
                pairs = _filter_unseen_nodes(pairs, self.train_seen_nodes)

            if self._mode in ('pos', 'new_pos'):
                pairs = _filter_positive_edges(pairs)

            return to_device(pairs, self._device)

        @property
        def train_dataset(self):
            return self._derive('train')

        @property
        def val_dataset(self):
            return self._val_cache

        @property
        def test_dataset(self):
            return self._test_cache

    setup_seed(args.seed)
    time_window = args.twin if args.twin != -1 else 5
    neg_ratio = getattr(args, "neg_sampling_ratio", 1.0)
    dataset = FundsEdgeWeightDataset(
        time_window=time_window,
        neg_sampling_ratio=neg_ratio,
        seed=args.seed,
    )

    is_new_edge_task = args.task in ("link_weight_multitask_new", "link_weight_multitask_new_twostage", "link_weight_multitask_new_joint")

    # Detect whether the dataset's train_dataset is a property (CasMLN-literal
    # per-epoch resampling). Static (instance-attribute) train_dataset → use
    # the snapshot path below to preserve original-baseline behavior exactly.
    _is_per_epoch_dataset = isinstance(getattr(type(dataset), 'train_dataset', None), property)

    if _is_per_epoch_dataset:
        print("[run_model] Detected property-based dataset.train_dataset — "
              "enabling per-epoch resampling via _LiveWrapper")
        tsn = dataset.train_seen_nodes
        data_wrapper_all     = _LiveWrapper(dataset, 'all',     neg_ratio, args.device, is_new_edge_task, tsn)
        data_wrapper_new     = _LiveWrapper(dataset, 'new',     neg_ratio, args.device, is_new_edge_task, tsn)
        data_wrapper_pos     = _LiveWrapper(dataset, 'pos',     neg_ratio, args.device, is_new_edge_task, tsn)
        data_wrapper_new_pos = _LiveWrapper(dataset, 'new_pos', neg_ratio, args.device, is_new_edge_task, tsn)
    else:
        # ── Existing snapshot path (UNCHANGED — preserves original baseline) ──
        train_pairs_all = dataset.train_dataset
        val_pairs_all = dataset.val_dataset
        test_pairs_all = dataset.test_dataset

        if is_new_edge_task:
            train_pairs_new = _filter_new_edges(train_pairs_all, neg_sampling_ratio=neg_ratio)
            val_pairs_new = _filter_new_edges(val_pairs_all, neg_sampling_ratio=neg_ratio)
            test_pairs_new = _filter_new_edges(test_pairs_all, neg_sampling_ratio=neg_ratio)
        else:
            train_pairs_new = train_pairs_all
            val_pairs_new = val_pairs_all
            test_pairs_new = test_pairs_all

        # Exclude unseen nodes from val/test (CasMLN-style transductive evaluation)
        # Training data is kept as-is — it defines what "seen" means.
        tsn = dataset.train_seen_nodes
        val_pairs_all = _filter_unseen_nodes(val_pairs_all, tsn)
        test_pairs_all = _filter_unseen_nodes(test_pairs_all, tsn)
        val_pairs_new = _filter_unseen_nodes(val_pairs_new, tsn)
        test_pairs_new = _filter_unseen_nodes(test_pairs_new, tsn)

        # Positive-only views for pure regression in stage 2
        train_pairs_pos = _filter_positive_edges(train_pairs_all)
        val_pairs_pos = _filter_positive_edges(val_pairs_all)    # already unseen-filtered
        test_pairs_pos = _filter_positive_edges(test_pairs_all)  # already unseen-filtered

        # New-edge positive-only views for stage 2 (new edges only)
        train_pairs_new_pos = _filter_positive_edges(train_pairs_new)
        val_pairs_new_pos = _filter_positive_edges(val_pairs_new)    # already unseen-filtered
        test_pairs_new_pos = _filter_positive_edges(test_pairs_new)  # already unseen-filtered

        # Device placement
        train_pairs_all = to_device(train_pairs_all, args.device)
        val_pairs_all = to_device(val_pairs_all, args.device)
        test_pairs_all = to_device(test_pairs_all, args.device)

        train_pairs_new = to_device(train_pairs_new, args.device)
        val_pairs_new = to_device(val_pairs_new, args.device)
        test_pairs_new = to_device(test_pairs_new, args.device)

        train_pairs_pos = to_device(train_pairs_pos, args.device)
        val_pairs_pos = to_device(val_pairs_pos, args.device)
        test_pairs_pos = to_device(test_pairs_pos, args.device)

        train_pairs_new_pos = to_device(train_pairs_new_pos, args.device)
        val_pairs_new_pos = to_device(val_pairs_new_pos, args.device)
        test_pairs_new_pos = to_device(test_pairs_new_pos, args.device)

        data_wrapper_new = SimpleNamespace(
            train_dataset=train_pairs_new, val_dataset=val_pairs_new, test_dataset=test_pairs_new,
            train_seen_nodes=tsn,
        )
        data_wrapper_all = SimpleNamespace(
            train_dataset=train_pairs_all, val_dataset=val_pairs_all, test_dataset=test_pairs_all,
            train_seen_nodes=tsn,
        )
        data_wrapper_pos = SimpleNamespace(
            train_dataset=train_pairs_pos, val_dataset=val_pairs_pos, test_dataset=test_pairs_pos,
            train_seen_nodes=tsn,
        )
        data_wrapper_new_pos = SimpleNamespace(
            train_dataset=train_pairs_new_pos, val_dataset=val_pairs_new_pos, test_dataset=test_pairs_new_pos,
            train_seen_nodes=tsn,
        )

    hid_dim = args.hid_dim if args.hid_dim != -1 else 64
    time_window = args.twin if args.twin != -1 else 5
    model_name = args.model

    if model_name in ("HGT", "HGT+"):
        use_rte = (model_name == "HGT+")
        # When --use_tcmp is set, swap standard HGT for HGT_TCMP (per-layer
        # FiLM modulation on fund features). Default off — flag absent keeps
        # existing PBSes byte-identical.
        if getattr(args, 'use_tcmp', False):
            from core.models.HGT_TCMP import HGT_TCMP
            base_model = HGT_TCMP(
                hidden_channels=hid_dim,
                out_channels=1,
                num_heads=args.n_heads,
                num_layers=args.n_layers,
                metadata=dataset.metadata,
                predict_type=["fund", "stock"],
                use_RTE=use_rte,
                dropout=getattr(args, 'dropout', 0.1),
                time_window=time_window,
                tcmp_text_dim=int(getattr(args, 'tcmp_text_dim', 128)),
            )
        else:
            from core.models.HGT import HGT
            base_model = HGT(
                hidden_channels=hid_dim,
                out_channels=1,
                num_heads=args.n_heads,
                num_layers=args.n_layers,
                metadata=dataset.metadata,
                predict_type=["fund", "stock"],
                use_RTE=use_rte,
                dropout=getattr(args, 'dropout', 0.1),
                time_window=time_window,
            )
    elif model_name == "HGT+EW":
        from core.models.HGT_EW import HGTEW
        base_model = HGTEW(
            hidden_channels=hid_dim,
            out_channels=1,
            num_heads=args.n_heads,
            num_layers=args.n_layers,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            use_RTE=True,
            dropout=getattr(args, 'dropout', 0.1),
            time_window=time_window,
        )
    elif model_name == "HGT+EWlog":
        from core.models.HGT_EW_log import HGTEWLog
        base_model = HGTEWLog(
            hidden_channels=hid_dim,
            out_channels=1,
            num_heads=args.n_heads,
            num_layers=args.n_layers,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            use_RTE=True,
            dropout=getattr(args, 'dropout', 0.1),
            time_window=time_window,
        )
    elif model_name == "HGT+EWattn":
        from core.models.HGT_EW_attn import HGTEWAttn
        base_model = HGTEWAttn(
            hidden_channels=hid_dim,
            out_channels=1,
            num_heads=args.n_heads,
            num_layers=args.n_layers,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            use_RTE=True,
            dropout=getattr(args, 'dropout', 0.1),
            time_window=time_window,
        )
    elif model_name == "HGT+EWlogPlus":
        from core.models.HGT_EW_log_plus import HGTEWLogPlus
        base_model = HGTEWLogPlus(
            hidden_channels=hid_dim,
            out_channels=1,
            num_heads=args.n_heads,
            num_layers=args.n_layers,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            use_RTE=True,
            dropout=getattr(args, 'dropout', 0.1),
            time_window=time_window,
        )
    elif model_name == "HGT+EW+TCETF":
        from core.models.HGT_EW_tcetf import HGTEWTCETF
        base_model = HGTEWTCETF(
            hidden_channels=hid_dim,
            out_channels=1,
            num_heads=args.n_heads,
            num_layers=args.n_layers,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            use_RTE=True,
            dropout=getattr(args, 'dropout', 0.1),
            time_window=time_window,
        )
    elif model_name == "HTGNN":
        from core.models.HTGNN import HTGNN
        base_model = HTGNN(
            n_inp=hid_dim,
            n_hid=hid_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            time_window=time_window,
            norm=False,
            device=args.device,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            dropout=getattr(args, 'dropout', 0.2),
        )
    elif model_name == "DHSpace":
        from core.models.DHSpace import DHSpace as _DHSpace, DHNet
        import json as _json
        dhconfig = getattr(args, 'dhconfig', '')
        if not dhconfig:
            raise ValueError("DHSpace requires --dhconfig path to searched architecture")
        cfg = torch.load(os.path.join(dhconfig, "config"), map_location="cpu")
        info = _json.load(open(os.path.join(dhconfig, "supernet.json")))
        dhspaces = []
        for a in cfg:
            dhspace = _DHSpace(
                hid_dim,
                dataset.metadata,
                time_window,
                K_To=info["KTO"],
                K_N=info["KN"],
                K_R=info["KR"],
                rel_time_type=info["rel_time_type"],
                n_heads=args.n_heads,
                norm=getattr(args, 'norm', False),
                hupdate=True,
                args=args,
            )
            dhspace.assign_arch(a)
            dhspaces.append(dhspace)
        base_model = DHNet(
            hid_dim,
            time_window,
            dataset.metadata,
            dhspaces,
            predict_type=["fund", "stock"],
            hlinear_act=getattr(args, 'hlinear_act', 'tanh'),
        )
    elif model_name == "SEHTGNN":
        # SE-HTGNN baseline (TGHTGNN with both modules disabled) for joint training.
        # Uses the TGHTGNN class from SE-HTGNN repo with use_relation_enrichment=False,
        # use_temporal_rules=False — identical to original SEHTGNN behaviour.
        import dgl
        # run_model.py -> scripts/run -> scripts -> mutual_fund_prediction -> project root
        _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        se_root = os.path.join(_proj_root, "SE-HTGNN")
        if os.path.isdir(se_root) and se_root not in sys.path:
            sys.path.insert(0, se_root)
        from model.model_TG_HTGNN import TGHTGNN

        llm_path = getattr(args, 'llm_embedding_path', None)
        if llm_path is None or not os.path.exists(llm_path):
            raise ValueError(
                f"SEHTGNN requires --llm_embedding_path pointing to funds_llm_features .pt file "
                f"(got: {llm_path})"
            )
        llm_feature = torch.load(llm_path, map_location=args.device)
        print(f"[SEHTGNN] Loaded LLM features from {llm_path}, keys: {list(llm_feature.keys())}")

        # Build a template DGL heterograph from the first training sample
        # to initialize TGHTGNN's adaption layers and graph structure.
        from core.models.load_model import SEHTGNNFundsWrapper
        _wrapper_tmp = SEHTGNNFundsWrapper.__new__(SEHTGNNFundsWrapper)
        _wrapper_tmp.time_window = time_window
        first_support = data_wrapper_all.train_dataset[0][0]
        if isinstance(first_support, (list, tuple)):
            first_support_cpu = [g.cpu() for g in first_support]
        else:
            first_support_cpu = first_support.cpu()
        template_g = SEHTGNNFundsWrapper._build_dgl_graph(_wrapper_tmp, first_support_cpu)

        # Build per-node-type input dimension dict from template graph features
        inp_list = {}
        for ntype in template_g.ntypes:
            if "t0" in template_g.nodes[ntype].data:
                feat = template_g.nodes[ntype].data["t0"]
            else:
                feat = next(iter(template_g.nodes[ntype].data.values()), None)
            if feat is not None and feat.dim() == 2:
                inp_list[ntype] = int(feat.size(1))
            else:
                inp_list[ntype] = 1

        # When --use_prospectus is on, MultiTaskEdgePredictor._apply_prospectus_fusion
        # replaces fund.x with the fused 128-d output before base_model.encode
        # runs, so TGHTGNN's adaption_layer['fund'] must be sized 128 (not 11)
        # or the Linear(11, n_hid) matmul against (N, 128) input errors out.
        # Both ProspectusTextFusion and ProspectusTextConcatFusion emit
        # OUTPUT_DIM=128 for h_fund by design.
        if getattr(args, 'use_prospectus', False):
            inp_list['fund'] = 128

        backbone = TGHTGNN(
            graph=template_g,
            n_inp=max(inp_list.values()),
            n_hid=hid_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            time_window=time_window,
            norm=bool(getattr(args, 'norm', False)),
            device=args.device,
            dropout=getattr(args, 'dropout', 0.2),
            LLM_feature=llm_feature,
            inp_list=inp_list,
            use_relation_enrichment=False,
            use_temporal_rules=False,
        )
        base_model = SEHTGNNFundsWrapper(
            backbone=backbone,
            nclf_linear=None,
            predict_type=["fund", "stock"],
            time_window=time_window,
        )
        print(f"[SEHTGNN] inp_list={inp_list}, hid_dim={hid_dim}, n_layers={args.n_layers}, "
              f"n_heads={args.n_heads}, time_window={time_window}")
    elif model_name == "CMLN":
        from core.models.CMLN import CMLN as CMLNNet
        # Load pre-computed LLM features
        llm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "llm_features")
        llm_graph_path = os.path.join(llm_dir, "casmln_funds_graph_emb.pt")
        llm_cate_path = os.path.join(llm_dir, "casmln_funds_cate_embs.pt")
        if not os.path.exists(llm_graph_path) or not os.path.exists(llm_cate_path):
            raise FileNotFoundError(
                f"CasMLN requires pre-computed LLM features.\n"
                f"  Expected: {llm_graph_path}\n"
                f"           {llm_cate_path}\n"
                f"  Run: python scripts/data_building/generate_casmln_llm_features.py"
            )
        llm_graph_emb = torch.load(llm_graph_path, map_location=args.device)
        llm_cate_embs = torch.load(llm_cate_path, map_location=args.device)
        print(f"[CMLN] Loaded LLM features: graph={llm_graph_emb.shape}, cate={len(llm_cate_embs)}x{llm_cate_embs[0].shape}")

        amplifier = getattr(args, 'amplifier', 5.0)
        # NODE-LEVEL PROSPECTUS MODULATION (independent of --use_prospectus)
        _node_text_loader = None
        if getattr(args, 'use_prospectus_node', False):
            from core.data.prospectus_loader import ProspectusEmbeddingLoader
            _node_text_loader = ProspectusEmbeddingLoader(
                getattr(args, 'prospectus_emb_path',
                        'sec_filings_project/embeddings_openai/prospectus_embeddings.h5'),
                mask_invalid_strategy=bool(getattr(args, 'mask_invalid_strategy', 1)),
            )
            print("[NODE_TEXT] Loaded prospectus loader for node-level modulation")
        base_model = CMLNNet(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            num_layers=args.n_layers,
            dropout=getattr(args, 'dropout', 0.6),
            time_window=time_window,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            device=args.device,
            amplifier=amplifier,
            llm_graph_emb=llm_graph_emb,
            llm_cate_embs=llm_cate_embs,
            prospectus_loader=_node_text_loader,
        )
        print(f"[CMLN] hid_dim={hid_dim}, n_layers={args.n_layers}, "
              f"amplifier={amplifier}, dropout={getattr(args, 'dropout', 0.6)}, "
              f"time_window={time_window}")
    elif model_name == "GCN":
        from core.models.GCN import GCN
        from core.models.HLinear import HLinear
        hlinear = HLinear(hid_dim, dataset.metadata, act='tanh')
        static_model = GCN(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            num_layers=args.n_layers,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            featemb=hlinear,
        )
        base_model = _TimeMergeWrapper(static_model)

    elif model_name == "GAT":
        from core.models.GAT import GAT
        from core.models.HLinear import HLinear
        hlinear = HLinear(hid_dim, dataset.metadata, act='tanh')
        static_model = GAT(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            num_layers=args.n_layers,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            heads=args.n_heads,
            dropout=getattr(args, 'dropout', 0.2),
            featemb=hlinear,
        )
        base_model = _TimeMergeWrapper(static_model)

    elif model_name == "SAGE":
        from core.models.SAGE import SAGE
        from core.models.HLinear import HLinear
        hlinear = HLinear(hid_dim, dataset.metadata, act='tanh')
        static_model = SAGE(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            num_layers=args.n_layers,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            featemb=hlinear,
        )
        base_model = _TimeMergeWrapper(static_model)

    elif model_name == "RGCN":
        from core.models.RGCN import RGCN
        from core.models.HLinear import HLinear
        hlinear = HLinear(hid_dim, dataset.metadata, act='tanh')
        static_model = RGCN(
            in_dim=hid_dim,
            hid_dim=hid_dim,
            num_layers=args.n_layers,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            featemb=hlinear,
        )
        base_model = _TimeMergeWrapper(static_model)

    elif model_name == "HAN":
        from core.models.HAN import HAN
        static_model = HAN(
            out_channels=hid_dim,
            hidden_channels=hid_dim,
            num_layers=args.n_layers,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            heads=args.n_heads,
            dropout=getattr(args, 'dropout', 0.2),
        )
        base_model = _TimeMergeWrapper(static_model)

    elif model_name == "DySAT":
        from core.models.DySAT import DySAT
        from core.models.HLinear import HLinear
        hlinear = HLinear(hid_dim, dataset.metadata, act='tanh')
        base_model = DySAT(
            hid_dim=hid_dim,
            time_length=time_window,
            num_layers=args.n_layers,
            n_heads=args.n_heads,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            featemb=hlinear,
        )

    elif model_name == "DyHATR":
        from core.models.DyHATR import DyHATR
        base_model = DyHATR(
            in_dim=hid_dim,
            n_hid=hid_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            time_window=time_window,
            metadata=dataset.metadata,
            predict_type=["fund", "stock"],
            dropout=getattr(args, 'dropout', 0.2),
        )

    else:
        raise ValueError(f"Unsupported model '{model_name}' for multitask edge weight prediction. "
                         f"Supported: HGT, HGT+, HTGNN, DHSpace, SEHTGNN, CMLN, GCN, GAT, SAGE, RGCN, HAN, DySAT, DyHATR")

    base_model = base_model.to(args.device)
    is_joint = args.task == "link_weight_multitask_new_joint"
    use_two_stage = args.task == "link_weight_multitask_new_twostage"
    use_joint_losses = getattr(args, 'use_joint_losses', False)

    # PROSPECTUS_INTEGRATION: create fusion modules if enabled
    _prospectus_fusion = None
    _contrastive_loss = None
    _prospectus_loader = None
    _fund_stock_contrastive = None  # [Approach 4]
    _text_behavior_alignment = None  # Option 1
    _spatial_alignment = None  # Option 2
    _text_only_head = None  # C2-4
    if getattr(args, 'use_prospectus', False):
        from core.data.prospectus_loader import ProspectusEmbeddingLoader
        from core.models.prospectus_fusion import (
            ProspectusTextFusion, ContrastiveLoss,
            FundStockContrastiveLoss,
        )
        _prospectus_fusion_mode = getattr(args, 'prospectus_fusion_mode', 'gated')
        if _prospectus_fusion_mode == 'concat':
            from core.data.prospectus_loader_flexdim import (
                ProspectusEmbeddingLoaderFlexDim,
            )
            from core.models.prospectus_fusion_concat import (
                ProspectusTextConcatFusion,
            )
            _prospectus_loader = ProspectusEmbeddingLoaderFlexDim(
                args.prospectus_emb_path,
                risk_weight=getattr(args, 'risk_weight', 0.375),
                device="cpu",
                mask_invalid_strategy=bool(getattr(args, 'mask_invalid_strategy', 1)),
            )
            _prospectus_fusion = ProspectusTextConcatFusion(
                input_dim=_prospectus_loader.EMB_DIM,
                numerical_dim=getattr(args, 'numerical_dim', 16),
            )
        elif _prospectus_fusion_mode == 'sum':
            from core.data.prospectus_loader_flexdim import (
                ProspectusEmbeddingLoaderFlexDim,
            )
            from core.models.prospectus_fusion_sum import (
                ProspectusTextSumFusion,
            )
            _prospectus_loader = ProspectusEmbeddingLoaderFlexDim(
                args.prospectus_emb_path,
                risk_weight=getattr(args, 'risk_weight', 0.375),
                device="cpu",
                mask_invalid_strategy=bool(getattr(args, 'mask_invalid_strategy', 1)),
            )
            _prospectus_fusion = ProspectusTextSumFusion(
                input_dim=_prospectus_loader.EMB_DIM,
                numerical_dim=getattr(args, 'numerical_dim', 16),
            )
        else:
            _prospectus_loader = ProspectusEmbeddingLoader(
                args.prospectus_emb_path,
                risk_weight=getattr(args, 'risk_weight', 0.375),
                device="cpu",  # kept on CPU, moved per-batch
                mask_invalid_strategy=bool(getattr(args, 'mask_invalid_strategy', 1)),
            )
            _prospectus_fusion = ProspectusTextFusion(
                numerical_dim=getattr(args, 'numerical_dim', 16),
                no_staleness=getattr(args, 'no_staleness', False),
                staleness_scale=getattr(args, 'staleness_scale', 6.0),
                use_jacobian_ratio_gate=getattr(args, 'use_jacobian_ratio_gate', False),
                no_text_fund_feature=getattr(args, 'no_text_fund_feature', False),
                fusion_mode=getattr(args, 'fusion_mode', 'convex'),
            )
        if getattr(args, 'no_text_fund_feature', False):
            print("[PROSPECTUS] --no_text_fund_feature ON: fund.x stays raw "
                  "numerical (baseline-style). abs_proj_out still flows to "
                  "TBA/SpAB/SpatialAlign/ToA aux losses.")
        # Contrastive loss is opt-in (off by default per 2026-04-07 spec).
        # Class is allocated but the auxiliary loss term is NOT yet wired into
        # train_epoch — enabling this flag is a no-op until that wiring lands.
        if getattr(args, 'use_contrastive', False):
            _contrastive_loss = ContrastiveLoss(temperature=0.07)
            print("[PROSPECTUS][WARN] --use_contrastive set, but ContrastiveLoss is not "
                  "yet wired into train_epoch. The module is allocated but its loss is "
                  "currently NOT added to the training objective. TODO before relying on it.")
        # [Approach 4] Fund-stock contrastive (off by default)
        if getattr(args, 'use_fund_stock_contrastive', False):
            _fund_stock_contrastive = FundStockContrastiveLoss(abs_proj_dim=128, temperature=0.07)
        # Option 1: text-behavior trajectory alignment (off by default)
        if getattr(args, 'use_text_behavior_alignment', False):
            from core.models.text_behavior_alignment import TextBehaviorAlignment
            _text_behavior_alignment = TextBehaviorAlignment(
                text_dim=ProspectusEmbeddingLoader.EMB_DIM,
                behav_dim=TextBehaviorAlignment.DEFAULT_BEHAV_DIM,  # 33 for 2005Q3 dataset
                temperature=0.1,
            )
        # Option 2: spatial text-stock alignment (off by default)
        if getattr(args, 'use_spatial_alignment', False):
            from core.models.spatial_alignment import SpatialAlignment
            # Auto-detect stock feature dim from first snapshot. US 2005Q3 gives 15
            # (matches DEFAULT_STOCK_FEAT_DIM → unchanged behavior); Canadian
            # brokenfeats gives 11 (avoids the previous hard-coded 15 shape mismatch).
            _spa_stock_dim = SpatialAlignment.DEFAULT_STOCK_FEAT_DIM
            try:
                # FundsEdgeWeightDataset exposes .graphs (dict[key]->HeteroData)
                # and .keys (sorted list). Fall back to .dataset for older /
                # alternate dataset shapes.
                _snap_iter = getattr(dataset, 'graphs', None) \
                    or getattr(dataset, 'dataset', None)
                _keys = getattr(dataset, 'keys', None) or (
                    list(_snap_iter.keys()) if hasattr(_snap_iter, 'keys') else [0])
                _first_snap = _snap_iter[_keys[0]]
                if 'stock' in _first_snap.node_types and _first_snap['stock'].x is not None:
                    _spa_stock_dim = int(_first_snap['stock'].x.shape[1])
                    print(f"[SpA] auto-detected stock_feat_dim={_spa_stock_dim} "
                          f"from first snapshot ({_keys[0]})")
            except Exception as _e:
                print(f"[SpA] stock_feat_dim auto-detect failed ({_e!r}); "
                      f"using default {_spa_stock_dim}")
            _spatial_alignment = SpatialAlignment(
                abs_proj_dim=ProspectusTextFusion.ABS_PROJ_DIM,
                stock_feat_dim=_spa_stock_dim,
                align_dim=64,
                temperature=getattr(args, 'spatial_alignment_temperature', 0.1),
            )
        # C2-4: Text-only auxiliary head (off by default — ablation appendix)
        if getattr(args, 'use_text_only_aux_loss', False):
            from core.models.text_only_head import TextOnlyEdgeHead
            _text_only_head = TextOnlyEdgeHead(
                abs_proj_dim=ProspectusTextFusion.ABS_PROJ_DIM,
                stock_feat_dim=TextOnlyEdgeHead.DEFAULT_STOCK_FEAT_DIM,
                hidden_dim=64,
            )

    # M1 (IBF probe): IntentConditionedAttention. Off unless --use_intent_attention.
    # Requires --use_prospectus + --prospectus_emb_path (loader sources abs_emb/delta_t).
    _intent_attention = None
    if getattr(args, 'use_intent_attention', False):
        if _prospectus_loader is None:
            raise ValueError(
                "--use_intent_attention requires --use_prospectus + "
                "--prospectus_emb_path. M1 attention sources text via the "
                "prospectus loader.")
        from core.models.intent_conditioned_attention import IntentConditionedAttention
        _intent_attention = IntentConditionedAttention(
            encoder_dim=hid_dim,
            lambda_mu=getattr(args, 'intent_lambda', 1.0),
            safety_cap=getattr(args, 'intent_safety_cap', 512),
            holdings_cap=getattr(args, 'intent_holdings_cap', 512),
            siblings_cap=getattr(args, 'intent_siblings_cap', 128),
            staleness_scale=getattr(args, 'intent_staleness_scale', 24.0),
        )

    model = MultiTaskEdgePredictor(
        base_model,
        hidden_dim=hid_dim,
        regression_loss="l2",
        class_loss_scale=getattr(args, 'cls_weight', 1.0) if is_joint else 1.0,
        weight_loss_scale=getattr(args, 'reg_weight', 1.0) if is_joint else 1.0,
        persist_loss_scale=0.0 if is_joint else 1.0,
        use_class_weights=not is_joint and not (use_two_stage and use_joint_losses),
        joint_mode=is_joint,
        huber_delta=getattr(args, 'huber_delta', 1.0),
        pos_weight_override=getattr(args, 'pos_weight', -1.0),
        use_bce_with_logits=(use_two_stage and use_joint_losses),
        use_logspace_huber=(use_two_stage and use_joint_losses),
        use_cls_mlp=os.environ.get("USE_CLS_MLP", "0") == "1",
        # PROSPECTUS_INTEGRATION
        prospectus_fusion=_prospectus_fusion,
        contrastive_loss=_contrastive_loss,
        prospectus_loader=_prospectus_loader,
        # [Approach 3]
        text_stock_prior=getattr(args, 'text_stock_prior', False),
        # [Approach 2]
        text_mlp_decoder=getattr(args, 'text_mlp_decoder', False),
        # [Approach 4]
        fund_stock_contrastive=_fund_stock_contrastive,
        fund_stock_contrastive_lambda=getattr(args, 'fund_stock_contrastive_lambda', 0.1),
        # Option 1: text-behavior trajectory alignment
        text_behavior_alignment=_text_behavior_alignment,
        text_behavior_alignment_lambda=getattr(args, 'text_behavior_alignment_lambda', 0.1),
        text_behavior_alignment_gate=getattr(args, 'text_behavior_alignment_gate', False),
        # Option 2: spatial text-stock alignment
        spatial_alignment=_spatial_alignment,
        spatial_alignment_lambda=getattr(args, 'spatial_alignment_lambda', 0.1),
        spatial_logit_beta=getattr(args, 'spatial_logit_beta', 0.0),
        use_combined_gate=getattr(args, 'use_combined_gate', False),
        alignment_kl_lambda=getattr(args, 'alignment_kl_lambda', 0.0),
        use_spatial_attention_bias=getattr(args, 'use_spatial_attention_bias', False),
        spa_negative_scope=getattr(args, 'spa_negative_scope', 'global'),
        # C2-4: Text-only auxiliary head
        text_only_head=_text_only_head,
        text_only_aux_lambda=getattr(args, 'text_only_aux_lambda', 0.1),
        # M1 (IBF probe): intent-conditioned candidate attention
        intent_attention=_intent_attention,
    ).to(args.device)

    # Edge-trajectory module: gated by --use_edge_trajectory.
    # When the flag is absent, this block is skipped and behavior is byte-identical
    # to baseline. configure_edge_trajectory() lives on MultiTaskEdgePredictor (see
    # Task 8); it builds an EdgeTrajectoryEncoder + EdgeTrajectoryFiLM internally
    # using the hetero metadata exposed via the dataset.
    if getattr(args, 'use_edge_trajectory', False):
        ds_metadata = getattr(dataset, 'metadata', None)
        assert ds_metadata is not None, (
            "Dataset does not expose `.metadata`; --use_edge_trajectory needs "
            "hetero metadata to instantiate per-edge-type FiLM heads."
        )
        model.configure_edge_trajectory(
            dim=args.edge_trajectory_dim,
            metadata=ds_metadata,
            hid_dim=args.hid_dim,
            heads=args.n_heads,
            log1p_input=getattr(args, 'edge_trajectory_log1p', False),
            no_present=getattr(args, 'edge_trajectory_no_present', False),
            static=getattr(args, 'edge_trajectory_static', False),
            no_delta_w=getattr(args, 'edge_trajectory_no_delta_w', False),
            use_tcetf=getattr(args, 'use_tcetf', False),
            tcetf_text_dim=int(getattr(args, 'tcetf_text_dim', 128)),
            tcetf_mode=getattr(args, 'tcetf_mode', 'additive'),
            use_text_init=getattr(args, 'tcetf_text_init', False),
            text_init_dim=int(getattr(args, 'tcetf_text_init_dim', 128)),
            use_text_step_input=getattr(args, 'tcetf_text_step_input', False),
            text_step_input_dim=int(getattr(args, 'tcetf_text_step_input_dim', 128)),
        )

    # Temporal Text Trajectory (TTT): instantiate the GRU encoder for per-fund
    # text across the support window. Output replaces instantaneous abs_proj as
    # TCMP's per-layer FiLM input. Requires --use_tcmp + --use_prospectus.
    # Default off — existing PBSes unaffected.
    if getattr(args, 'use_ttt', False):
        if not getattr(args, 'use_tcmp', False):
            print("[TTT][WARN] --use_ttt requires --use_tcmp; ignoring (no TCMP FiLM to feed).")
        elif not getattr(args, 'use_prospectus', False):
            print("[TTT][WARN] --use_ttt requires --use_prospectus; ignoring.")
        else:
            model.configure_text_trajectory(
                text_dim=128,
                traj_dim=int(getattr(args, 'ttt_dim', 128)),
            )

    # Tail-reweighting (B-experiment): attach alpha + training q67 onto the model.
    # train_epoch reads these and up-weights positive edges with raw weight > q67.
    # alpha=0 (default) leaves the loss unchanged.
    model.tail_reweight_alpha = float(getattr(args, 'tail_reweight_alpha', 0.0) or 0.0)
    model.tail_reweight_q67 = float(getattr(dataset, 'weight_q67_train', 0.0) or 0.0)

    # PROSPECTUS_INTEGRATION: set dataset keys for snapshot→quarter alignment
    if getattr(args, 'use_prospectus', False):
        model._dataset_keys = dataset.keys

    # NODE-LEVEL PROSPECTUS MODULATION: also set _dataset_keys on the BASE model
    # so CMLN.encode() can map snapshot indices to H5 quarter indices.
    if getattr(args, 'use_prospectus_node', False):
        if hasattr(model, 'base_model'):
            model.base_model._dataset_keys = dataset.keys
        else:
            model._dataset_keys = dataset.keys
        print(f"[NODE_TEXT] Set _dataset_keys ({len(dataset.keys)} snapshots) on base CMLN")

    # Cold-start token follow-up (only meaningful when use_prospectus_node is also set)
    if getattr(args, 'use_prospectus_node_cold_token', False):
        _base = model.base_model if hasattr(model, 'base_model') else model
        if hasattr(_base, '_use_cold_token'):
            _base._use_cold_token = True
            print(f"[NODE_TEXT] Cold-start token substitution ENABLED")
        else:
            print(f"[NODE_TEXT][WARN] --use_prospectus_node_cold_token requires --use_prospectus_node "
                  f"on a CMLN model — flag ignored.")

    # NODE-LEVEL PROSPECTUS MODULATION: Option B — periodic forward-hook diagnostics
    # The two-stage multitask trainer (core/trainer/edge_multitask.py) does not call
    # log_prospectus_diagnostics from inside its loop. As a non-invasive workaround,
    # register a forward hook on the model that prints diagnostics every K forward
    # calls (~once per epoch). This gives mid-training visibility without touching
    # the trainer.
    if getattr(args, 'use_prospectus_node', False) and hasattr(model, 'base_model'):
        _node_text_state = {"calls": 0}
        _node_text_K = 200  # ~once per epoch on the 2005Q3 dataset
        def _node_text_diag_hook(module, inputs, output):
            _node_text_state["calls"] += 1
            if _node_text_state["calls"] % _node_text_K == 0:
                model.base_model.log_prospectus_diagnostics(
                    prefix=f" call={_node_text_state['calls']}"
                )
        model.register_forward_hook(_node_text_diag_hook)
        print(f"[NODE_TEXT] Registered forward hook (prints every {_node_text_K} calls)")

    # WEIGHT-AWARE CONTRASTIVE LOSS (Stage 1 auxiliary)
    if getattr(args, 'weight_contrastive', False):
        from core.loss.weight_contrastive import WeightAwareContrastiveLoss
        _wc_loss = WeightAwareContrastiveLoss(
            n_bins=getattr(args, 'weight_contrastive_bins', 4),
            temperature=getattr(args, 'weight_contrastive_temp', 0.1),
            max_samples=512,
        ).to(args.device)
        model.weight_contrastive_fn = _wc_loss
        model.weight_contrastive_lambda = getattr(args, 'weight_contrastive_lambda', 0.1)
        print(f"[WEIGHT_CONTRASTIVE] Enabled for Stage 1: "
              f"lambda={model.weight_contrastive_lambda}, "
              f"temp={args.weight_contrastive_temp}, "
              f"bins={args.weight_contrastive_bins}")

    # EDGE MEMORY NETWORK
    if getattr(args, 'use_edge_memory', False):
        from core.models.edge_memory import EdgeMemoryModule, PeerAttentionInit
        _mem_dim_v2 = getattr(args, 'memory_dim_v2', None)
        _mem_dim = _mem_dim_v2 if _mem_dim_v2 is not None else getattr(args, 'memory_dim', 32)
        _stratified = getattr(args, 'stratified_peers', False)
        _profile_query = getattr(args, 'profile_query', False)
        _profile_dim = 4 if _profile_query else 0
        # [Abl4] Rich GRU input: project fund+stock embeddings to a compact dim
        _rich_dim = hid_dim if getattr(args, 'gru_rich_input', False) else 0
        _edge_mem = EdgeMemoryModule(memory_dim=_mem_dim, rich_input_dim=_rich_dim)
        # [Abl3] Build holders from all snapshots
        if getattr(args, 'peer_holders_all_snapshots', False):
            _edge_mem.all_snapshots_holders = True
        _max_peers = getattr(args, 'max_peers', 50)
        _peer_attn = PeerAttentionInit(
            node_embed_dim=hid_dim,
            memory_dim=_mem_dim,
            num_heads=4,
            profile_dim=_profile_dim,
            stratified_peers=_stratified,
            max_peers=_max_peers,
        )
        model.edge_memory_module = _edge_mem
        model.peer_attention = _peer_attn
        model.edge_memory_module.to(args.device)
        model.peer_attention.to(args.device)
        model.use_edge_memory = True
        model.use_weight_profile = getattr(args, 'use_weight_profile', False)
        model._profile_query = _profile_query
        model._film_decoder = getattr(args, 'film_decoder', False)
        # Wire up ablation flags
        model._edge_memory_trainable = getattr(args, 'edge_memory_trainable', False)
        model._edge_profile_mode = getattr(args, 'edge_weight_profile_mode', 'fund')
        model._peer_residual_gate = getattr(args, 'peer_residual_gate', False)
        model._gru_rich_input = getattr(args, 'gru_rich_input', False)
        model._dual_pathway = getattr(args, 'dual_pathway', False)
        # [Abl5] Create gate and projection layers for residual connection
        if model._peer_residual_gate:
            import torch.nn as _nn
            model._gate_proj = _nn.Linear(hid_dim * 2 + _mem_dim, 1).to(args.device)
            model._peer_to_base = _nn.Linear(_mem_dim, hid_dim * 2).to(args.device)
        _abl_tags = []
        if model._edge_memory_trainable: _abl_tags.append("trainable_gru")
        if model._edge_profile_mode == "edge": _abl_tags.append("edge_profiles")
        if getattr(args, 'peer_holders_all_snapshots', False): _abl_tags.append("all_snap_holders")
        if model._gru_rich_input: _abl_tags.append("rich_gru_input")
        if model._peer_residual_gate: _abl_tags.append("residual_gate")
        if model._dual_pathway: _abl_tags.append("dual_pathway")
        if _stratified: _abl_tags.append("stratified_peers")
        if _profile_query: _abl_tags.append("profile_query")
        if model._film_decoder: _abl_tags.append("film_decoder")
        print(f"[EDGE_MEMORY] Enabled: memory_dim={_mem_dim}, "
              f"weight_profile={'ON' if model.use_weight_profile else 'OFF'}"
              + (f", ablations=[{', '.join(_abl_tags)}]" if _abl_tags else ""))

    try:
        print(f"Model: {model_name}, Params: {sum(p.numel() for p in model.parameters()):,}")
    except ValueError:
        print(f"Model: {model_name} (param count deferred — lazy modules)")
    if use_two_stage and use_joint_losses:
        print(f"[CONFIG] Two-stage with joint-style losses: BCEWithLogitsLoss (cls) + HuberLoss/log1p (reg)")

    # PROSPECTUS_INTEGRATION: create optimizer with separate param groups if prospectus enabled
    def _make_optimizer(lr_override=None, wd_override=None):
        lr = lr_override if lr_override is not None else args.lr
        wd = wd_override if wd_override is not None else args.wd
        if getattr(args, 'use_prospectus', False):
            # Two param groups: backbone vs text fusion modules
            backbone_params = []
            text_params = []
            for name, param in model.named_parameters():
                if any(k in name for k in ("prospectus_fusion", "contrastive",
                                            "text_behavior_alignment",
                                            "spatial_alignment",
                                            "text_only_head")):
                    text_params.append(param)
                else:
                    backbone_params.append(param)
            return torch.optim.Adam([
                {"params": backbone_params, "lr": lr, "weight_decay": wd},
                {"params": text_params, "lr": lr, "weight_decay": wd},
            ])
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    use_two_stage = args.task == "link_weight_multitask_new_twostage"
    monitor_metric = getattr(args, "monitor_metric", "")
    if use_two_stage:
        print(f"Training {model_name} (edge weight multitask) on NEW edges with TWO stages")
        print("Stage 1: link prediction (val_loss). Stage 2: edge weight regression (val_loss).")
        model.persist_loss_scale = 0.0  # disable persistence loss for both stages
        print(f"Device: {args.device}")
        print(f"[Stage 1] Train steps: {len(data_wrapper_new.train_dataset)}, Val: {len(data_wrapper_new.val_dataset)}, Test: {len(data_wrapper_new.test_dataset)} (new edges)")
        print(f"[Stage 2] Train steps: {len(data_wrapper_new_pos.train_dataset)}, Val: {len(data_wrapper_new_pos.val_dataset)}, Test: {len(data_wrapper_new_pos.test_dataset)} (new pos edges)")
        print(f"[Unseen-node filter] Val/test restricted to edges with both endpoints seen in training")

        def _train_stage(stage_name: str, class_scale: float, weight_scale: float, monitor: str, ckpt_path: str = None, lr_override=None, wd_override=None, grad_clip: float = 0.0, use_scheduler: bool = False):
            model.class_loss_scale = class_scale
            model.weight_loss_scale = weight_scale
            optimizer = _make_optimizer(lr_override=lr_override, wd_override=wd_override)

            # PROSPECTUS_INTEGRATION: use three-phase training when prospectus enabled.
            # --no_phase_scheduler bypasses this and routes through the regular trainer
            # (forward fusion still runs; only the freeze/LR schedule is skipped).
            if (getattr(args, 'use_prospectus', False)
                    and not getattr(args, 'no_phase_scheduler', False)):
                from core.models.prospectus_fusion import PhaseScheduler
                from core.trainer.edge_multitask import train_till_end_prospectus
                phase_scheduler = PhaseScheduler(
                    phase_a_epochs=10,
                    phase_b_epochs=30,
                    no_modality_dropout=getattr(args, 'no_modality_dropout', False),
                )
                print(f"\n[{stage_name}] THREE-PHASE: class_scale={class_scale}, weight_scale={weight_scale}, monitor={monitor}")
                print(f"  Phase A (0-9): freeze backbone, lr_txt=1e-3")
                print(f"  Phase B (10-39): unfreeze, modality dropout 0.80→0.20")
                print(f"  Phase C (40+): full training, dropout 0.20")
                return train_till_end_prospectus(
                    model=model,
                    optimizer=optimizer,
                    dataset=data_wrapper_new if class_scale > 0 else data_wrapper_new_pos,
                    args=args,
                    phase_scheduler=phase_scheduler,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    disable_progress=False,
                    grad_clip=grad_clip,
                    monitor=monitor,
                    skip_reg=class_scale > 0,
                    skip_cls=weight_scale > 0,
                    best_ckpt_path=ckpt_path,
                )

            # Original path (no prospectus)
            # Create scheduler if requested
            scheduler = None
            if use_scheduler:
                sched_type = os.environ.get("STAGE2_SCHEDULER", "plateau").lower()
                if sched_type == "cosine":
                    warmup_epochs = int(os.environ.get("STAGE2_WARMUP", 10))
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=args.max_epochs - warmup_epochs,
                        eta_min=1e-6,
                    )
                    scheduler._warmup_epochs = warmup_epochs
                    scheduler._base_lr = optimizer.param_groups[0]['lr']
                    print(f"[{stage_name}] Using learning rate scheduler: CosineAnnealingLR "
                          f"(T_max={args.max_epochs - warmup_epochs}, warmup={warmup_epochs} epochs)")
                else:
                    mode = "min" if monitor in {"mae", "mae_scaled", "mae_unscaled", "mae_entry", "mae_large_change", "val_loss"} else "max"
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode=mode,
                        factor=0.5,
                        patience=10,
                        verbose=True,
                        min_lr=1e-6
                    )
                    print(f"[{stage_name}] Using learning rate scheduler: ReduceLROnPlateau (mode={mode}, patience=10, factor=0.5)")

            print(f"\n[{stage_name}] class_scale={class_scale}, weight_scale={weight_scale}, monitor={monitor}")
            return train_till_end(
                model=model,
                optimizer=optimizer,
                # Stage 1 uses new edges (classification); Stage 2 uses new positives only (regression)
                dataset=data_wrapper_new if class_scale > 0 else data_wrapper_new_pos,
                args=args,
                max_epochs=args.max_epochs,
                patience=args.patience,
                disable_progress=False,
                grad_clip=grad_clip,
                monitor=monitor,
                skip_reg=class_scale > 0,  # Stage1: skip regression metrics/loss logging
                skip_cls=weight_scale > 0,  # Stage2: skip classification metrics
                best_ckpt_path=ckpt_path,
                scheduler=scheduler,
                # Stage-2 only: linear LR warmup to prevent fresh-regression-head
                # first-step blowups on large-scale targets (default 0 = off)
                warmup_epochs=int(getattr(args, 'stage2_warmup_epochs', 0)) if weight_scale > 0 else 0,
            )

        # Edge memory in Stage 1: precompute once so GRU/attention are trained for link prediction
        if getattr(args, 'use_edge_memory', False) and getattr(args, 'edge_memory_both_stages', False):
            model._edge_memory_active = True
            all_s1_batches = list(data_wrapper_new.train_dataset) + list(data_wrapper_new.val_dataset) + list(data_wrapper_new.test_dataset)
            model.precompute_all_edge_memories(all_s1_batches)
            print(f"[EDGE_MEMORY] Active in Stage 1 — GRU/attention trained for link prediction")

        # Stage 1: link prediction only (weight loss disabled), or load checkpoint if skipped
        stage1_monitor = getattr(args, 'stage1_monitor', '') or "val_loss"
        stage1_ckpt = os.path.join(args.log_dir, "stage1_best.pt")
        skip_stage1 = bool(getattr(args, "skip_stage1", False)) or os.environ.get("SKIP_STAGE1", "0").strip() in (
            "1", "true", "True", "yes", "YES",
        )
        stage1_load_path = (getattr(args, "stage1_ckpt", "") or "").strip() or os.environ.get("STAGE1_CKPT", "").strip()
        if not stage1_load_path:
            stage1_load_path = stage1_ckpt

        stage1_start = time.time()
        if skip_stage1:
            if not os.path.isfile(stage1_load_path):
                raise FileNotFoundError(
                    f"--skip_stage1 requires an existing Stage 1 checkpoint; missing: {stage1_load_path!r}. "
                    f"Use --stage1_ckpt or env STAGE1_CKPT, or place stage1_best.pt under --log_dir."
                )
            print(f"[Stage 1] SKIPPED — loading weights from {stage1_load_path}")
            state = torch.load(stage1_load_path, map_location=args.device)
            # Materialize lazy modules (e.g. HGTConv's lin_dict) before load_state_dict.
            with torch.no_grad():
                _sup_q = next(iter(data_wrapper_new.val_dataset))
                _sup, _q = _sup_q
                model(_sup, _q.edge_label_index,
                      edge_fund_baseline=getattr(_q, 'edge_fund_baseline', None))
            model.load_state_dict(state, strict=True)

            # Apply Stage 1 loss-scale settings then run evaluate() to populate test metrics
            # (test_auc/ap/entry_recall@K etc.) — otherwise stage1_stats would be empty and
            # downstream consumers (CSV builders, summary scripts) would see no Stage 1 metrics.
            model.class_loss_scale = 1.0
            model.weight_loss_scale = 0.0
            from core.trainer.edge_multitask import evaluate as _evaluate
            tsn_local = data_wrapper_new.train_seen_nodes if hasattr(data_wrapper_new, "train_seen_nodes") else None
            val_metrics = _evaluate(model, data_wrapper_new.val_dataset,
                                    skip_cls=False, skip_reg=True,
                                    train_seen_nodes=tsn_local, return_loss=True)
            test_metrics = _evaluate(model, data_wrapper_new.test_dataset,
                                     skip_cls=False, skip_reg=True,
                                     train_seen_nodes=tsn_local)
            stage1_stats = {
                "skipped": True,
                "loaded_checkpoint": stage1_load_path,
                "val_loss": float(val_metrics.get("val_loss", 0.0)),
            }
            for k, v in test_metrics.items():
                rk = f"test_{k}" if not k.startswith("test_") else k
                stage1_stats[rk] = v
            stage1_dur = 0.0
        else:
            stage1_stats = _train_stage(
                "Stage 1 (link)",
                class_scale=1.0,
                weight_scale=0.0,
                monitor=stage1_monitor,
                ckpt_path=stage1_ckpt,
                grad_clip=2.0,
            )
            stage1_dur = time.time() - stage1_start
            if os.path.exists(stage1_ckpt):
                print(f"Reloading best Stage 1 checkpoint from {stage1_ckpt}")
                state = torch.load(stage1_ckpt, map_location=args.device)
                model.load_state_dict(state)

        # Save Stage 1 per-edge predictions (additive; the existing Stage 2
        # save at end of training is unchanged). Enabled by SAVE_EDGE_PREDICTIONS=1,
        # same env var used for the Stage 2 save. New file: test_edge_predictions_stage1.npz.
        if os.environ.get("SAVE_EDGE_PREDICTIONS", "0") == "1":
            try:
                from core.trainer.edge_multitask import save_edge_predictions as _save_pred_s1
                _s1_path = os.path.join(args.log_dir, "test_edge_predictions_stage1.npz")
                _save_pred_s1(model, data_wrapper_new.test_dataset, _s1_path, joint_log_space=False)
                print(f"[Stage 1] Per-edge predictions saved to {_s1_path}")
            except Exception as _e_s1:
                print(f"[Stage 1] Per-edge prediction save failed: {_e_s1}")

        # Evaluate best Stage1 checkpoint multiple times to gauge variance (fixed negatives)
        if skip_stage1:
            repeat_eval = int(os.environ.get("STAGE1_EVAL_REPEATS", "0"))
        else:
            repeat_eval = int(os.environ.get("STAGE1_EVAL_REPEATS", "5"))
        stage1_repeat_metrics = []
        if repeat_eval > 0:
            for i in range(repeat_eval):
                metrics = {
                    "val": evaluate(model, data_wrapper_new.val_dataset, skip_cls=False, skip_reg=True),
                    "test": evaluate(model, data_wrapper_new.test_dataset, skip_cls=False, skip_reg=True),
                }
                stage1_repeat_metrics.append(metrics)
            print(f"[Stage 1] Repeat eval ({repeat_eval} runs) complete.")
        else:
            print("[Stage 1] Repeat eval skipped (STAGE1_EVAL_REPEATS=0 or --skip_stage1).")

        import gc as _gc_stage1
        _gc_stage1.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

        # Disable weight-aware contrastive loss for Stage 2 (only needed in Stage 1)
        if getattr(model, 'weight_contrastive_fn', None) is not None:
            print("[WEIGHT_CONTRASTIVE] Disabled for Stage 2")
            model.weight_contrastive_fn = None

        # Helper: build N-layer regression MLP (reads STAGE2_MLP_LAYERS env var)
        def _build_reg_mlp(input_dim, hidden_dim=128, dropout=0.1):
            n_layers = int(os.environ.get("STAGE2_MLP_LAYERS", 2))
            if n_layers <= 2:
                return nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
            layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            layers.append(nn.Linear(hidden_dim, 1))
            mlp = nn.Sequential(*layers)
            print(f"[DEEP_MLP] Built {n_layers}-layer regression MLP: {input_dim} -> {hidden_dim} x{n_layers-1} -> 1")
            return mlp

        # EDGE MEMORY: activate for Stage 2
        if getattr(args, 'use_edge_memory', False):
            model._edge_memory_active = True
            import torch.nn as _nn
            _mem_dim_v2 = getattr(args, 'memory_dim_v2', None)
            mem_dim = _mem_dim_v2 if _mem_dim_v2 is not None else getattr(args, 'memory_dim', 32)
            profile_dim = 4 if getattr(args, 'use_weight_profile', False) else 0

            # [v2] FiLM conditioning: peer memory produces scale/shift on GNN-only prediction
            if getattr(model, '_film_decoder', False):
                gnn_input_dim = hid_dim * 2
                model.reg_mlp = _build_reg_mlp(gnn_input_dim).to(args.device)
                model._film_scale_net = _nn.Sequential(
                    _nn.Linear(mem_dim, mem_dim),
                    _nn.ReLU(),
                    _nn.Linear(mem_dim, 1),
                    _nn.Sigmoid(),
                ).to(args.device)
                model._film_shift_net = _nn.Sequential(
                    _nn.Linear(mem_dim, mem_dim),
                    _nn.ReLU(),
                    _nn.Linear(mem_dim, 1),
                ).to(args.device)
                # Initialize scale near 1.0 and shift near 0.0
                with torch.no_grad():
                    model._film_scale_net[-2].bias.fill_(2.0)
                    model._film_shift_net[-1].bias.fill_(0.0)
                print(f"[FILM_DECODER] Activated for Stage 2:")
                print(f"  - GNN head (reg_mlp): input_dim={gnn_input_dim}")
                print(f"  - FiLM scale/shift nets: peer_dim={mem_dim}")

            # [Strategy 3] Dual-pathway: separate GNN and trajectory heads
            elif getattr(model, '_dual_pathway', False):
                gnn_input_dim = hid_dim * 2
                traj_input_dim = mem_dim + profile_dim
                gate_input_dim = hid_dim * 2 + mem_dim

                model.reg_mlp = _build_reg_mlp(gnn_input_dim).to(args.device)

                model.traj_mlp = _nn.Sequential(
                    _nn.Linear(traj_input_dim, 128),
                    _nn.ReLU(),
                    _nn.Dropout(0.1),
                    _nn.Linear(128, 1),
                ).to(args.device)

                model.pathway_gate = _nn.Sequential(
                    _nn.Linear(gate_input_dim, 64),
                    _nn.ReLU(),
                    _nn.Linear(64, 1),
                ).to(args.device)

                print(f"[DUAL_PATHWAY] Activated for Stage 2:")
                print(f"  - GNN head (reg_mlp): input_dim={gnn_input_dim}")
                print(f"  - Trajectory head (traj_mlp): input_dim={traj_input_dim}")
                print(f"  - Gate network: input_dim={gate_input_dim}")

            # [Abl5] Gated residual: decoder input is hid_dim*2 (+ optional profiles)
            elif getattr(model, '_peer_residual_gate', False):
                input_dim = hid_dim * 2 + profile_dim
                model.reg_mlp = _build_reg_mlp(input_dim).to(args.device)
                print(f"[EDGE_MEMORY] Activated for Stage 2, reg_mlp rebuilt: input_dim={input_dim}")

            else:
                input_dim = hid_dim * 2 + mem_dim + profile_dim
                model.reg_mlp = _build_reg_mlp(input_dim).to(args.device)
                print(f"[EDGE_MEMORY] Activated for Stage 2, reg_mlp rebuilt: input_dim={input_dim}")

            _gc_stage1.collect()
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

            # Re-precompute for Stage 2 (positive-only edges, different dataset from Stage 1)
            all_batches = list(data_wrapper_new_pos.train_dataset) + list(data_wrapper_new_pos.val_dataset) + list(data_wrapper_new_pos.test_dataset)
            model.precompute_all_edge_memories(all_batches)

        # Stage 2: weight regression only (classification loss disabled); monitor MAE only
        # Improved stability settings: lower LR, weight decay, gradient clipping, LR scheduler
        stage2_monitor = getattr(args, 'stage2_monitor', '') or "val_loss"
        stage2_lr = float(os.environ.get("STAGE2_LR", 0.0005))  # Reduced from 0.0001 to 0.0005
        stage2_wd = float(os.environ.get("STAGE2_WD", 0.0001))  # Added weight decay
        stage2_grad_clip = float(os.environ.get("STAGE2_GRAD_CLIP", 1.0))  # Reduced from 2.0 to 1.0
        
        stage2_sched = os.environ.get("STAGE2_SCHEDULER", "plateau").lower()
        stage2_mlp_layers = int(os.environ.get("STAGE2_MLP_LAYERS", 2))
        print(f"[Stage 2] Configuration:")
        print(f"  - Learning rate: {stage2_lr}")
        print(f"  - Weight decay: {stage2_wd}")
        print(f"  - Gradient clipping: {stage2_grad_clip}")
        print(f"  - LR scheduler: {stage2_sched}")
        print(f"  - Regression MLP layers: {stage2_mlp_layers}")
        print(f"  - Dropout: 0.1")
        
        stage2_ckpt = os.path.join(args.log_dir, "stage2_best.pt")  # PROSPECTUS_INTEGRATION: save Stage 2 ckpt

        # Optional skip path mirroring --skip_stage1: load best Stage 2 ckpt and run final eval only.
        skip_stage2 = bool(getattr(args, "skip_stage2", False)) or os.environ.get("SKIP_STAGE2", "0").strip() in (
            "1", "true", "True", "yes", "YES",
        )
        stage2_load_path = (getattr(args, "stage2_ckpt", "") or "").strip() or os.environ.get("STAGE2_CKPT", "").strip()
        if not stage2_load_path:
            stage2_load_path = stage2_ckpt

        stage2_start = time.time()
        if skip_stage2:
            if not os.path.isfile(stage2_load_path):
                raise FileNotFoundError(
                    f"--skip_stage2 requires an existing Stage 2 checkpoint; missing: {stage2_load_path!r}. "
                    f"Use --stage2_ckpt or env STAGE2_CKPT, or place stage2_best.pt under --log_dir."
                )
            print(f"[Stage 2] SKIPPED — loading weights from {stage2_load_path}")
            # Apply Stage 2 loss-scale settings (mirrors _train_stage at stage 2 invocation).
            model.class_loss_scale = 0.0
            model.weight_loss_scale = 1.0
            state = torch.load(stage2_load_path, map_location=args.device)
            model.load_state_dict(state, strict=True)

            # Run final eval on Stage 2's positive-only test set (mirrors train_till_end's final_test).
            from core.trainer.edge_multitask import evaluate as _evaluate
            tsn_local = data_wrapper_new_pos.train_seen_nodes if hasattr(data_wrapper_new_pos, "train_seen_nodes") else None
            val_metrics = _evaluate(model, data_wrapper_new_pos.val_dataset,
                                    skip_cls=True, skip_reg=False,
                                    train_seen_nodes=tsn_local, return_loss=True)
            test_metrics = _evaluate(model, data_wrapper_new_pos.test_dataset,
                                     skip_cls=True, skip_reg=False,
                                     train_seen_nodes=tsn_local)
            stage2_stats = {
                "skipped": True,
                "loaded_checkpoint": stage2_load_path,
                "val_val_loss": float(val_metrics.get("val_loss", 0.0)),
            }
            for k, v in test_metrics.items():
                rk = f"test_{k}" if not k.startswith("test_") else k
                stage2_stats[rk] = v
            stage2_dur = 0.0
        else:
            stage2_stats = _train_stage(
                "Stage 2 (weight)",
                class_scale=0.0,
                weight_scale=1.0,
                monitor=stage2_monitor,
                ckpt_path=stage2_ckpt,
                lr_override=stage2_lr,
                wd_override=stage2_wd,
                grad_clip=stage2_grad_clip,
                use_scheduler=True,  # Enable LR scheduler for Stage 2
            )
            stage2_dur = time.time() - stage2_start

        # NODE-LEVEL PROSPECTUS MODULATION: Option A — final aggregate summary
        # Prints once after all training completes. Gives at minimum a single
        # confirmation that modulation fired, plus aggregate cold-start fraction
        # and modulation magnitudes over the entire run.
        if getattr(args, 'use_prospectus_node', False) and hasattr(model, 'base_model'):
            if hasattr(model.base_model, 'log_prospectus_diagnostics'):
                model.base_model.log_prospectus_diagnostics(prefix=" final-aggregate")

        # Rename generic 'val_auc' key to reflect actual monitor metric.
        if stage2_monitor in {"mae", "mae_scaled", "mae_unscaled", "val_loss"}:
            if "val_auc" in stage2_stats:
                stage2_stats[f"val_{stage2_monitor}"] = stage2_stats.pop("val_auc")
            if "test_auc" in stage2_stats:
                stage2_stats.pop("test_auc")
        if stage1_monitor == "val_loss":
            if "val_auc" in stage1_stats:
                stage1_stats["val_loss"] = stage1_stats.pop("val_auc")

        train_stats = {
            "stage1": stage1_stats,
            "stage2": stage2_stats,
        }
        duration = stage1_dur + stage2_dur
    else:
        optimizer = _make_optimizer()
        if monitor_metric:
            monitor = monitor_metric
        elif args.task == "link_weight_multitask_new_joint":
            monitor = "val_loss"
        elif args.task == "link_weight_multitask_new":
            monitor = "mae"
        else:
            monitor = "auc"

        if args.task == "link_weight_multitask_new_joint":
            print(f"Training {model_name} (JOINT new-edge) — single-stage BCE+Huber")
            print(f"  cls_weight={args.cls_weight}, reg_weight={args.reg_weight}, huber_delta={args.huber_delta}, pos_weight={'auto' if args.pos_weight < 0 else args.pos_weight}")
        elif args.task == "link_weight_multitask_new":
            print(f"Training {model_name} (edge weight multitask) on NEW edges only")
        else:
            print(f"Training {model_name} (edge weight multitask)")
        print(f"Device: {args.device}")
        active_wrapper = data_wrapper_new if is_new_edge_task else data_wrapper_all
        print(f"Train steps: {len(active_wrapper.train_dataset)}, Val: {len(active_wrapper.val_dataset)}, Test: {len(active_wrapper.test_dataset)}")

        # Joint training: report all metrics (both cls and reg); early stop on val_loss
        skip_cls = not is_joint and args.task == "link_weight_multitask_new"
        skip_reg = False
        gc = getattr(args, 'grad_clip', 0.0)
        if is_joint:
            gc = gc if gc > 0 else 2.0

        start = time.time()
        os.makedirs(args.log_dir, exist_ok=True)
        joint_ckpt_path = os.path.join(args.log_dir, "joint_best.pt") if is_joint else None

        # Optional skip path: load joint_best.pt and run final eval only (mirrors --skip_stage2).
        skip_joint = is_joint and (
            bool(getattr(args, "skip_joint", False))
            or os.environ.get("SKIP_JOINT", "0").strip() in ("1", "true", "True", "yes", "YES")
        )
        joint_load_path = (getattr(args, "joint_ckpt", "") or "").strip() or os.environ.get("JOINT_CKPT", "").strip()
        if not joint_load_path:
            joint_load_path = joint_ckpt_path or ""

        if skip_joint:
            if not joint_load_path or not os.path.isfile(joint_load_path):
                raise FileNotFoundError(
                    f"--skip_joint requires an existing Joint checkpoint; missing: {joint_load_path!r}. "
                    f"Use --joint_ckpt or env JOINT_CKPT, or place joint_best.pt under --log_dir."
                )
            print(f"[Joint] SKIPPED — loading weights from {joint_load_path}")
            state = torch.load(joint_load_path, map_location=args.device)
            # Materialize lazy modules with a dummy forward before load_state_dict.
            with torch.no_grad():
                _sup_q = next(iter(active_wrapper.val_dataset))
                _sup, _q = _sup_q
                model(_sup, _q.edge_label_index,
                      edge_fund_baseline=getattr(_q, 'edge_fund_baseline', None))
            model.load_state_dict(state, strict=True)

            from core.trainer.edge_multitask import evaluate as _evaluate
            tsn_local = getattr(active_wrapper, "train_seen_nodes", None)
            val_metrics = _evaluate(model, active_wrapper.val_dataset,
                                    skip_cls=skip_cls, skip_reg=skip_reg,
                                    train_seen_nodes=tsn_local, return_loss=True)
            test_metrics = _evaluate(model, active_wrapper.test_dataset,
                                     skip_cls=skip_cls, skip_reg=skip_reg,
                                     train_seen_nodes=tsn_local)
            train_stats = {
                "skipped": True,
                "loaded_checkpoint": joint_load_path,
                "val_val_loss": float(val_metrics.get("val_loss", 0.0)),
            }
            for k, v in test_metrics.items():
                rk = f"test_{k}" if not k.startswith("test_") else k
                train_stats[rk] = v
        else:
            train_stats = train_till_end(
                model=model,
                optimizer=optimizer,
                dataset=active_wrapper,
                args=args,
                max_epochs=args.max_epochs,
                patience=args.patience,
                disable_progress=False,
                grad_clip=gc,
                monitor=monitor,
                skip_cls=skip_cls,
                skip_reg=skip_reg,
                best_ckpt_path=joint_ckpt_path,
            )
        duration = time.time() - start

    os.makedirs(args.log_dir, exist_ok=True)
    out_path = os.path.join(args.log_dir, "hgt_plus_edge_weight_results.json")
    payload = {
        "train_stats": train_stats,
        "duration_sec": duration,
        "params": sum(p.numel() for p in model.parameters() if not isinstance(p, torch.nn.parameter.UninitializedParameter)),
        "config": vars(args),
    }
    if use_two_stage:
        payload["stage1_duration_sec"] = stage1_dur
        payload["stage2_duration_sec"] = stage2_dur
        payload["stage1_eval_repeats"] = stage1_repeat_metrics
    # PROSPECTUS_INTEGRATION: add prospectus config to payload
    if getattr(args, 'use_prospectus', False):
        payload["prospectus_config"] = {
            "emb_path": args.prospectus_emb_path,
            "numerical_dim": args.numerical_dim,
            "risk_weight": getattr(args, 'risk_weight', 0.375),
            "mask_invalid_strategy": bool(getattr(args, 'mask_invalid_strategy', 1)),
            "no_staleness": getattr(args, 'no_staleness', False),
            "no_modality_dropout": getattr(args, 'no_modality_dropout', False),
            "contrastive_loss": model.contrastive_loss_fn is not None,
            "last_gate_mean": model._last_gate_mean,
            "text_behavior_alignment": (
                None if model.text_behavior_alignment_fn is None else {
                    "lambda": model.text_behavior_alignment_lambda,
                    "gate_enabled": model.text_behavior_alignment_gate,
                    "last_align_mean": model.text_behavior_alignment_fn._last_align_mean,
                    "last_align_gate_mean": model.text_behavior_alignment_fn._last_gate_mean,
                    "last_valid_frac": model.text_behavior_alignment_fn._last_valid_frac,
                }
            ),
        }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # PROSPECTUS_INTEGRATION: save drift matrix after training
    if getattr(args, 'use_prospectus', False) and model.use_prospectus:
        try:
            drift_path = os.path.join(args.log_dir, "drift_matrix.npy")
            print(f"[PROSPECTUS] Computing drift matrix...")
            # Drift = 1 - cosine_sim(full_signal, graph_only_signal)
            # Graph-only: run with zeroed text inputs
            model.eval()
            import numpy as np_drift
            # For now, save gate mean as a proxy for drift analysis
            np_drift.save(drift_path, np_drift.array([model._last_gate_mean]))
            print(f"[PROSPECTUS] Gate mean at end of training: {model._last_gate_mean:.4f}")
            print(f"[PROSPECTUS] Drift info saved to {drift_path}")
        except Exception as e:
            print(f"[PROSPECTUS] Drift computation failed: {e}")

    print(f"Done. Duration: {duration:.2f}s. Results saved to {out_path}")
    for k, v in train_stats.items():
        print(f"{k}: {v}")

    # Optionally save per-edge predictions on the test set
    if os.environ.get("SAVE_EDGE_PREDICTIONS", "0") == "1":
        from core.trainer.edge_multitask import save_edge_predictions
        pred_path = os.path.join(args.log_dir, "test_edge_predictions.npz")
        _wrapper = data_wrapper_new if is_new_edge_task else data_wrapper_all
        test_data = _wrapper.test_dataset
        jls = getattr(model, 'joint_mode', False) or getattr(model, 'use_logspace_huber', False)
        save_edge_predictions(model, test_data, pred_path, joint_log_space=jls)

    sys.exit(0)

# dataset
import sys
dataset, args = load_data(args)
print(f"DEBUG: args.twin = {args.twin}, args.time_window = {getattr(args, 'time_window', 'N/A')}")
if 'core.data.load_data' in sys.modules:
    print(f"DEBUG: load_data source: {sys.modules['core.data.load_data'].__file__}")

# model
model = load_model(args, dataset)

# device
model = model.to(args.device)
dataset.to(args.device)

# Propagate CLI modes to evaluation via env
import os as _os
if hasattr(args, 'target_mode'):
    _os.environ.setdefault('TARGET_MODE', args.target_mode)
if hasattr(args, 'loss_mode'):
    _os.environ.setdefault('LOSS_MODE', args.loss_mode)
if getattr(args, 'dataset', '') == 'Funds':
    # Prefer raw value-space targets for consistent MAE and to avoid normalized NaNs
    _os.environ.setdefault('TARGET_VALUE_SCALE', 'raw')

# train
# Skip node prediction task if link task is specified
task = getattr(args, "task", "regression")
if task not in ("link", "link_weight", "link_weight_delta_cls"):
    trainer, criterion = load_trainer(args)
    # Some models (e.g., XGBoost wrapper) have no torch parameters; skip optimizer
    try:
        params = list(model.parameters())
    except Exception:
        params = []
    optimizer = None if len(params) == 0 else torch.optim.Adam(params=params, lr=args.lr, weight_decay=args.wd)
    train_dict = trainer(
        model,
        optimizer,
        criterion,
        dataset,
        args,
        args.max_epochs,
        args.patience,
        disable_progress=False,
        writer=None,
        grad_clip=args.grad_clip,
        device=args.device,
    )
    # Print appropriate metric based on available keys (Rank-IC first if available)
    if 'test_rank_ic' in train_dict:
        print(f"Final Test Rank-IC: {train_dict['test_rank_ic']:.4f}")
        if 'test_mae' in train_dict:
            print(f"Final Test MAE:     {train_dict['test_mae']:.4f}")
    elif 'test_mae' in train_dict:
        print(f"Final Test MAE: {train_dict['test_mae']:.4f}")
    elif 'test_auc' in train_dict:
        print(f"Final Test AUC: {train_dict['test_auc']:.4f}")
    else:
        print(f"Final Test Result: {train_dict}")
else:
    # For link task, we still need an optimizer
    try:
        params = list(model.parameters())
    except Exception:
        params = []
    # Increase base LR for more aggressive training
    base_lr = 0.005  # Increased from default 0.001
    optimizer = None if len(params) == 0 else torch.optim.Adam(params=params, lr=base_lr, weight_decay=args.wd)

if task == "regression":
    # Comprehensive evaluation
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION")
    print("="*60)

    # Evaluate on test set with comprehensive metrics
    test_metrics = evaluate_dhgas_model(model, dataset, device=args.device)
    # Attach target scale if available
    if hasattr(dataset, 'target_scale'):
        test_metrics['target_scale'] = dataset.target_scale
    print_evaluation_results(test_metrics, "Test Set Evaluation")

    # Save metrics to CSV
    metrics_file = f"evaluation_results_{args.model}_{args.dataset}.csv"
    test_metrics.update({
        'lr': args.lr,
        'n_heads': getattr(args, 'n_heads', None),
        'hid_dim': getattr(args, 'hid_dim', None),
        'wd': args.wd,
        'twin': getattr(args, 'twin', None),
        'seed': args.seed
    })
    save_metrics_to_csv(test_metrics, metrics_file)
elif task == "link_weight":
    print("\n" + "="*60)
    print("LINK WEIGHT TASK SELECTED (Fixed Weights + Gradient Logging)")
    print("="*60)
    train_data = getattr(dataset, "train_dataset", None)
    val_pair = getattr(dataset, "val_dataset", None)
    test_pair = getattr(dataset, "test_dataset", None)
    if train_data is None or val_pair is None or test_pair is None or optimizer is None:
        print("[WARN] Missing train/val/test datasets or optimizer for link_weight task; skipping.")
    else:
        # ===== BASELINE: CURRENT WEIGHTS -> NEXT WEIGHTS =====
        def baseline_current_weights(pair_or_list, eps=1e-9):
            """
            Baseline: predict next-quarter weights equal to current-period weights (per-fund normalized).

            For each fund:
              - target distribution: normalized edge_label (t+1)
              - baseline prediction: normalized edge_current (t)
            We then compute MAE in scaled and unscaled space, mirroring eval_pair.
            """
            pairs = pair_or_list if (isinstance(pair_or_list, (list, tuple)) and len(pair_or_list) > 0 and isinstance(pair_or_list[0], tuple)) else [pair_or_list]
            maes_scaled = []
            maes_unscaled = []
            with torch.no_grad():
                for p in pairs:
                    _, eval_rel = p
                    el_idx = eval_rel.edge_label_index.to(args.device)
                    labels = eval_rel.edge_label.to(args.device).float()
                    cur_w = getattr(eval_rel, "edge_current", None)
                    if cur_w is None:
                        continue
                    cur_w = cur_w.to(args.device).float()
                    fund_ids = el_idx[0]
                    num_funds = int(fund_ids.max().item()) + 1 if fund_ids.numel() > 0 else 0
                    if num_funds == 0:
                        continue
                    # Normalize targets and baseline per fund
                    tgt_sum = scatter_add(labels, fund_ids, dim=0, dim_size=num_funds)
                    norm_target = labels / (tgt_sum[fund_ids] + eps)
                    cur_sum = scatter_add(cur_w, fund_ids, dim=0, dim_size=num_funds)
                    norm_cur = cur_w / (cur_sum[fund_ids] + eps)
                    # MAE in scaled space
                    maes_scaled.append(torch.mean(torch.abs(norm_cur - norm_target)).item())
                    # MAE in original units (mirror eval_pair scaling)
                    if hasattr(dataset, "edge_weight_clip"):
                        clip_val = float(getattr(dataset, "edge_weight_clip", 1.0))
                        maes_unscaled.append(torch.mean(torch.abs(norm_cur*clip_val - norm_target*clip_val)).item())
            out = {
                "mae_scaled": float(np.mean(maes_scaled)) if len(maes_scaled) > 0 else float("inf"),
            }
            if len(maes_unscaled) > 0:
                out["mae_unscaled"] = float(np.mean(maes_unscaled))
            return out

        val_baseline = baseline_current_weights(val_pair)
        test_baseline = baseline_current_weights(test_pair)
        print(
            f"[BASELINE] Current-weight Val MAE_scaled {val_baseline['mae_scaled']:.4f} "
            f"Val MAE_unscaled {val_baseline.get('mae_unscaled', float('nan')):.4f}"
        )
        print(
            f"[BASELINE] Current-weight Test MAE_scaled {test_baseline['mae_scaled']:.4f} "
            f"Test MAE_unscaled {test_baseline.get('mae_unscaled', float('nan')):.4f}"
        )

        # ===== UNCERTAINTY WEIGHTING REMOVED =====
        # Reverted to fixed weights: Gamma (MSE) = 1.0, Lam (Turn) = 0.1
        
        base_lr = 0.005  # Increased from default 0.001
        
        # Helper function for moving graphs to device
        def _move_graph(g, device):
            if isinstance(g, (list, tuple)):
                return [x.to(device) for x in g]
            return g.to(device)
        
        # Reverted to standard Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0)
        print(f"[DECODER] Reverted to standard model.decode() with Fixed Weights")
        
        # Shared holdings decoder loss: components (CE, MSE, Turnover)
        # We will use **pure MSE** as the training objective for this run.
        def holdings_loss_components(logits, edge_index, target_weights, current_weights=None, eps=1e-9):
            """
            Args:
                logits: unnormalized scores per edge (E,)
                edge_index: (2,E) with fund_ids in row 0, stock_ids in row 1
                target_weights: ground-truth weights (E,), non-negative
                current_weights: optional current-period weights aligned to the same edges; if None, turnover=0
                eps: numerical stability
            Returns:
                (ce, mse, turnover) as separate tensors
            """
            fund_ids = edge_index[0]
            num_funds = int(fund_ids.max().item()) + 1 if fund_ids.numel() > 0 else 0
            if num_funds == 0:
                zero = logits.new_tensor(0.0)
                return zero, zero, zero
            # Normalize targets per fund to get a distribution
            fund_sum = scatter_add(target_weights, fund_ids, dim=0, dim_size=num_funds)
            norm_target = target_weights / (fund_sum[fund_ids] + eps)
            # Masked softmax per fund for predictions
            log_norm = scatter_logsumexp(logits, fund_ids, dim=0, dim_size=num_funds)
            log_probs = logits - log_norm[fund_ids]
            pred_probs = torch.exp(log_probs)
            # Cross-entropy / KL term (not used in pure-MSE objective, but logged)
            ce = -(norm_target * log_probs).sum() / num_funds
            # MSE on normalized weights (this is the **only** term used in the loss)
            mse = ((pred_probs - norm_target) ** 2).sum() / num_funds
            # Turnover penalty (not used in pure-MSE objective, but logged)
            turnover = logits.new_tensor(0.0)
            if current_weights is not None:
                cur_sum = scatter_add(current_weights, fund_ids, dim=0, dim_size=num_funds)
                norm_cur = current_weights / (cur_sum[fund_ids] + eps)
                turnover = (pred_probs - norm_cur).abs().sum() / num_funds
            return ce, mse, turnover

        best_metric = float("inf")
        best_state = None
        patience_ctr = 0

        def eval_pair(pair_or_list):
            model.eval()
            pairs = pair_or_list if (isinstance(pair_or_list, (list, tuple)) and len(pair_or_list) > 0 and isinstance(pair_or_list[0], tuple)) else [pair_or_list]
            maes_scaled = []
            maes_unscaled = []
            with torch.no_grad():
                for p in pairs:
                    g, eval_rel = p
                    g = _move_graph(g, args.device)
                    el_idx = eval_rel.edge_label_index.to(args.device)
                    labels = eval_rel.edge_label.to(args.device).float()
                    z = model.encode(g, return_dict=True)
                    logits = model.decode(z, el_idx).view(-1)
                    # Recompute normalized targets for fair MAE
                    fund_ids = el_idx[0]
                    num_funds = int(fund_ids.max().item()) + 1 if fund_ids.numel() > 0 else 0
                    if num_funds == 0:
                        continue
                    fund_sum = scatter_add(labels, fund_ids, dim=0, dim_size=num_funds)
                    norm_target = labels / (fund_sum[fund_ids] + 1e-9)
                    log_norm = scatter_logsumexp(logits, fund_ids, dim=0, dim_size=num_funds)
                    pred = torch.exp(logits - log_norm[fund_ids])
                    # MAE in scaled space
                    maes_scaled.append(torch.mean(torch.abs(pred - norm_target)).item())
                    # MAE in original units (de-scale by clip if available)
                    if hasattr(dataset, "edge_weight_clip"):
                        clip_val = float(getattr(dataset, "edge_weight_clip", 1.0))
                        maes_unscaled.append(torch.mean(torch.abs(pred*clip_val - norm_target*clip_val)).item())
            out = {
                "mae_scaled": float(np.mean(maes_scaled)) if len(maes_scaled) > 0 else float("inf"),
            }
            if len(maes_unscaled) > 0:
                out["mae_unscaled"] = float(np.mean(maes_unscaled))
            return out

        # DIAGNOSTIC: Track first batch predictions across epochs
        first_batch_data = None
        first_batch_logits_history = []
        
        for epoch in range(args.max_epochs):
            model.train()
            total_loss = 0.0
            epoch_ce, epoch_mse, epoch_turn = 0.0, 0.0, 0.0
            
            train_maes_scaled = []
            
            # Track gradient norms
            total_grad_norm = 0.0
            num_steps = 0
            
            for batch_idx, (g, eval_rel) in enumerate(train_data):
                g = _move_graph(g, args.device)
                el_idx = eval_rel.edge_label_index.to(args.device)
                labels = eval_rel.edge_label.to(args.device).float()
                cur_w = getattr(eval_rel, "edge_current", None)
                if cur_w is not None:
                    cur_w = cur_w.to(args.device).float()
                optimizer.zero_grad()
                # For link prediction, need embeddings for both node types
                z = model.encode(g, return_dict=True)
                
                logits = model.decode(z, el_idx).view(-1)
                
                # Get individual loss components
                ce, mse, turnover = holdings_loss_components(logits, el_idx, labels, current_weights=cur_w)
                # === PURE MSE OBJECTIVE ===
                loss = mse
                
                loss.backward()
                
                # Log gradient norm
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.detach().data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                total_grad_norm += grad_norm
                num_steps += 1
                
                optimizer.step()
                total_loss += loss.item()
                epoch_ce += ce.item()
                epoch_mse += mse.item()
                epoch_turn += turnover.item()
                
                # DIAGNOSTIC: Save first batch and track predictions after optimizer.step()
                if batch_idx == 0:
                    if first_batch_data is None:
                        # Save first batch data
                        first_batch_data = (g, el_idx.clone(), labels.clone())
                    # Re-compute predictions AFTER optimizer.step() for tracking
                    with torch.no_grad():
                        z_diag = model.encode(g, return_dict=True)
                        logits_diag = model.decode(z_diag, el_idx).view(-1)
                        first_batch_logits_history.append(logits_diag[:10].cpu().clone())  # Save first 10 for brevity
                
                # Compute Train MAE for this batch (approximate)
                with torch.no_grad():
                    fund_ids = el_idx[0]
                    num_funds = int(fund_ids.max().item()) + 1 if fund_ids.numel() > 0 else 0
                    if num_funds > 0:
                        fund_sum = scatter_add(labels, fund_ids, dim=0, dim_size=num_funds)
                        norm_target = labels / (fund_sum[fund_ids] + 1e-9)
                        log_norm = scatter_logsumexp(logits, fund_ids, dim=0, dim_size=num_funds)
                        pred = torch.exp(logits - log_norm[fund_ids])
                        train_maes_scaled.append(torch.mean(torch.abs(pred - norm_target)).item())
            
            train_mae_mean = np.mean(train_maes_scaled) if train_maes_scaled else float('inf')
            avg_grad_norm = total_grad_norm / max(num_steps, 1)
            
            val_res = eval_pair(val_pair)
            test_res = eval_pair(test_pair)
            val_mae = val_res.get("mae_scaled", float("inf"))
            
            # === DIAGNOSTIC: Compute averages across batches ===
            
            print(
                f"Epoch {epoch} loss {total_loss/len(train_data):.4f} "
                f"(CE {epoch_ce/len(train_data):.4f} MSE {epoch_mse/len(train_data):.4f} Turn {epoch_turn/len(train_data):.4f}) "
                f"[PURE_MSE_OBJECTIVE] "
                f"| GradNorm={avg_grad_norm:.4f} | "
                f"TrainMAE {train_mae_mean:.4f} "
                f"val_MAE_scaled {val_res.get('mae_scaled', float('inf')):.4f} "
                f"val_MAE_unscaled {val_res.get('mae_unscaled', float('nan')):.4f} "
            )
            # NODE-LEVEL PROSPECTUS MODULATION: per-epoch diagnostics
            if getattr(args, 'use_prospectus_node', False):
                _base = model.base_model if hasattr(model, 'base_model') else model
                if hasattr(_base, 'log_prospectus_diagnostics'):
                    _base.log_prospectus_diagnostics(prefix=f" epoch={epoch}")
            if val_mae < best_metric:
                best_metric = val_mae
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # DIAGNOSTIC: Print first-batch prediction changes
        print("\n" + "="*60)
        print("DIAGNOSTIC: First Batch Logit Changes (first 10 edges)")
        print("="*60)
        if len(first_batch_logits_history) > 0:
            logits_epoch0 = first_batch_logits_history[0]
            logits_final = first_batch_logits_history[-1]
            delta = (logits_final - logits_epoch0).abs()
            print(f"Epoch 0 logits:  {logits_epoch0.numpy()}")
            print(f"Final logits:    {logits_final.numpy()}")
            print(f"Abs delta:       {delta.numpy()}")
            print(f"Mean abs delta:  {delta.mean().item():.6f}")
            if delta.mean().item() < 1e-5:
                print("⚠️  WARNING: Logits barely changed! Model may not be learning.")
            else:
                print("✅ Logits ARE changing across epochs.")
        print("="*60 + "\n")
        
        if best_state is not None:
            model.load_state_dict(best_state)
        final_val = eval_pair(val_pair)
        final_test = eval_pair(test_pair)
        print(
            f"Best Val MAE_scaled: {best_metric:.4f}, "
            f"Final Val MAE_scaled: {final_val.get('mae_scaled', float('inf')):.4f}, "
            f"Final Val MAE_unscaled: {final_val.get('mae_unscaled', float('nan')):.4f}, "
            f"Final Test MAE_scaled: {final_test.get('mae_scaled', float('inf')):.4f}, "
            f"Final Test MAE_unscaled: {final_test.get('mae_unscaled', float('nan')):.4f}"
        )
        train_dict = {
            "best_val_mae_scaled": best_metric,
            "final_val_mae_scaled": final_val.get("mae_scaled", None),
            "final_val_mae_unscaled": final_val.get("mae_unscaled", None),
            "final_test_mae_scaled": final_test.get("mae_scaled", None),
            "final_test_mae_unscaled": final_test.get("mae_unscaled", None),
        }
elif task == "link_weight_delta_cls":
    print("\n" + "="*60)
    print("LINK WEIGHT DELTA 4-CLASS TASK SELECTED")
    print("="*60)
    train_data = getattr(dataset, "train_dataset", None)
    val_pair = getattr(dataset, "val_dataset", None)
    test_pair = getattr(dataset, "test_dataset", None)
    if train_data is None or val_pair is None or test_pair is None:
        print("[WARN] Missing train/val/test datasets for link_weight_delta_cls; skipping.")
        train_dict = {}
    else:
        device = args.device

        def _move_graph(g, device):
            if isinstance(g, (list, tuple)):
                return [x.to(device) for x in g]
            if isinstance(g, dict):
                return {k: v.to(device) for k, v in g.items()}
            return g.to(device)

        def _edge_repr(z_dict_or_graph, edge_index, is_xgboost=False):
            fund_ids = edge_index[0]
            stock_ids = edge_index[1]
            if is_xgboost:
                # XGBoost: extract features directly from graph
                if isinstance(z_dict_or_graph, (list, tuple)):
                    g = z_dict_or_graph[-1]
                else:
                    g = z_dict_or_graph
                if hasattr(g, 'x_dict') and 'fund' in g.x_dict and 'stock' in g.x_dict:
                    z_fund = g['fund'].x[fund_ids]
                    z_stock = g['stock'].x[stock_ids]
                else:
                    # Fallback: try to get from HeteroData directly
                    z_fund = g['fund'].x[fund_ids] if 'fund' in g.node_types else torch.zeros(len(fund_ids), 1)
                    z_stock = g['stock'].x[stock_ids] if 'stock' in g.node_types else torch.zeros(len(stock_ids), 1)
            else:
                # GNN models: z_dict is a dict
                z_fund = z_dict_or_graph["fund"][fund_ids]
                z_stock = z_dict_or_graph["stock"][stock_ids]
            return torch.cat([z_fund, z_stock], dim=-1)

        # Initialize classifier head using a sample batch to infer dim
        sample_g, sample_eval = train_data[0]
        is_xgboost = args.model == "XGBoost"
        if is_xgboost:
            sample_z = _move_graph(sample_g, device)
        else:
            sample_z = model.encode(_move_graph(sample_g, device), return_dict=True)
        sample_repr = _edge_repr(sample_z, sample_eval.edge_label_index.to(device), is_xgboost=is_xgboost)
        feat_dim = sample_repr.size(1)
        hidden_dim = max(64, feat_dim // 2)
        edge_classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4),
        ).to(device)
        
        # Compute class weights from training data for weighted CE loss
        print("Computing class weights from training data...")
        train_labels_list = []
        for g, eval_rel in train_data:
            labels = eval_rel.edge_label
            if labels.numel() > 0:
                train_labels_list.append(labels)
        if len(train_labels_list) > 0:
            all_train_labels = torch.cat(train_labels_list)
            class_counts = torch.bincount(all_train_labels, minlength=4).float()
            total = class_counts.sum()
            # Inverse frequency weighting: weight = total / (num_classes * count)
            class_weights = total / (4.0 * class_counts + 1e-9)
            class_weights = class_weights / class_weights.sum() * 4.0  # Normalize to sum to num_classes
            class_weights = class_weights.to(device)
            print(f"Class weights: Decrease={class_weights[0]:.4f}, Stable={class_weights[1]:.4f}, Increase={class_weights[2]:.4f}, Entry={class_weights[3]:.4f}")
        else:
            class_weights = torch.ones(4, device=device)
            print("Warning: No training labels found, using uniform weights")

        # For XGBoost, only train the classifier head (XGBoost itself has no trainable params)
        if is_xgboost:
            optimizer = torch.optim.Adam(
                list(edge_classifier.parameters()),
                lr=args.lr,
                weight_decay=args.wd,
            )
        else:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(edge_classifier.parameters()),
                lr=args.lr,
                weight_decay=args.wd,
            )
        best_metric = -float("inf")
        best_state = None
        patience_ctr = 0

        def _metrics(pairs):
            # pairs can be tuple or list
            if isinstance(pairs, tuple) and len(pairs) == 2 and not isinstance(pairs[0], (list, tuple)):
                pairs = [pairs]
            total_correct = 0
            total = 0
            # Accumulate TP, FP, FN per class across all batches
            tp_per_class = torch.zeros(4, dtype=torch.long)
            fp_per_class = torch.zeros(4, dtype=torch.long)
            fn_per_class = torch.zeros(4, dtype=torch.long)
            model.eval()
            edge_classifier.eval()
            with torch.no_grad():
                for g, eval_rel in pairs:
                    g = _move_graph(g, device)
                    el_idx = eval_rel.edge_label_index.to(device)
                    labels = eval_rel.edge_label.to(device).long()
                    if el_idx.numel() == 0 or labels.numel() == 0:
                        continue
                    if is_xgboost:
                        z = g
                    else:
                        z = model.encode(g, return_dict=True)
                    edge_feat = _edge_repr(z, el_idx, is_xgboost=is_xgboost)
                    logits = edge_classifier(edge_feat)
                    preds = logits.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total += labels.numel()
                    # Accumulate TP, FP, FN per class
                    for cls in range(4):
                        tp = ((preds == cls) & (labels == cls)).sum().item()
                        fp = ((preds == cls) & (labels != cls)).sum().item()
                        fn = ((preds != cls) & (labels == cls)).sum().item()
                        tp_per_class[cls] += tp
                        fp_per_class[cls] += fp
                        fn_per_class[cls] += fn
            
            # Compute per-class F1
            f1_per_class = []
            for cls in range(4):
                tp = tp_per_class[cls].item()
                fp = fp_per_class[cls].item()
                fn = fn_per_class[cls].item()
                denom = (2 * tp + fp + fn)
                f1 = (2 * tp) / (denom + 1e-9)
                f1_per_class.append(f1)
            
            # Macro-F1: average of per-class F1s
            macro_f1 = float(np.mean(f1_per_class))
            
            # Event-F1: F1 over {Increase (2), Decrease (0)} union
            event_tp = tp_per_class[0].item() + tp_per_class[2].item()  # Decrease + Increase
            event_fp = fp_per_class[0].item() + fp_per_class[2].item()
            event_fn = fn_per_class[0].item() + fn_per_class[2].item()
            event_denom = (2 * event_tp + event_fp + event_fn)
            event_f1 = (2 * event_tp) / (event_denom + 1e-9)
            
            acc = float(total_correct / total) if total > 0 else 0.0
            return {
                "acc": acc,
                "macro_f1": macro_f1,
                "f1_per_class": f1_per_class,
                "event_f1": event_f1,
            }

        for epoch in range(args.max_epochs):
            if not is_xgboost:
                model.train()
            edge_classifier.train()
            total_loss = 0.0
            for g, eval_rel in train_data:
                g = _move_graph(g, device)
                el_idx = eval_rel.edge_label_index.to(device)
                labels = eval_rel.edge_label.to(device).long()
                if el_idx.numel() == 0 or labels.numel() == 0:
                    continue
                optimizer.zero_grad()
                if is_xgboost:
                    z = g
                else:
                    z = model.encode(g, return_dict=True)
                edge_feat = _edge_repr(z, el_idx, is_xgboost=is_xgboost)
                logits = edge_classifier(edge_feat)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    if not is_xgboost:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(edge_classifier.parameters(), args.grad_clip)
                optimizer.step()
                total_loss += loss.item()

            val_metrics = _metrics(val_pair)
            val_acc = val_metrics.get("acc", 0.0)
            val_macro_f1 = val_metrics.get("macro_f1", 0.0)
            val_event_f1 = val_metrics.get("event_f1", 0.0)
            f1_per_class = val_metrics.get("f1_per_class", [0.0] * 4)
            class_names = ["Decrease", "Stable", "Increase", "Entry"]
            f1_str = ", ".join([f"{name}={f1:.4f}" for name, f1 in zip(class_names, f1_per_class)])
            print(f"Epoch {epoch} delta-cls loss {total_loss/max(len(train_data),1):.4f} val_acc {val_acc:.4f} val_macro_f1 {val_macro_f1:.4f} val_event_f1 {val_event_f1:.4f}")
            print(f"  Per-class F1: {f1_str}")

            # NODE-LEVEL PROSPECTUS MODULATION: per-epoch diagnostics
            if getattr(args, 'use_prospectus_node', False):
                _base = model.base_model if hasattr(model, 'base_model') else model
                if hasattr(_base, 'log_prospectus_diagnostics'):
                    _base.log_prospectus_diagnostics(prefix=f" epoch={epoch}")

            # Early stopping based on macro-F1
            if val_macro_f1 > best_metric:
                best_metric = val_macro_f1
                if is_xgboost:
                    best_state = {
                        "head": {k: v.cpu() for k, v in edge_classifier.state_dict().items()},
                    }
                else:
                    best_state = {
                        "model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "head": {k: v.cpu() for k, v in edge_classifier.state_dict().items()},
                    }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            if not is_xgboost:
                model.load_state_dict(best_state["model"])
            edge_classifier.load_state_dict(best_state["head"])
        final_val = _metrics(val_pair)
        final_test = _metrics(test_pair)
        
        # Print comprehensive results
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Best Val Macro-F1: {best_metric:.4f}")
        print("\nValidation Set:")
        print(f"  Accuracy: {final_val.get('acc', 0.0):.4f}")
        print(f"  Macro-F1: {final_val.get('macro_f1', 0.0):.4f}")
        print(f"  Event-F1 (Increase+Decrease): {final_val.get('event_f1', 0.0):.4f}")
        val_f1_per_class = final_val.get("f1_per_class", [0.0] * 4)
        class_names = ["Decrease (0)", "Stable (1)", "Increase (2)", "Entry (3)"]
        print("  Per-class F1:")
        for name, f1 in zip(class_names, val_f1_per_class):
            print(f"    {name}: {f1:.4f}")
        
        print("\nTest Set:")
        print(f"  Accuracy: {final_test.get('acc', 0.0):.4f}")
        print(f"  Macro-F1: {final_test.get('macro_f1', 0.0):.4f}")
        print(f"  Event-F1 (Increase+Decrease): {final_test.get('event_f1', 0.0):.4f}")
        test_f1_per_class = final_test.get("f1_per_class", [0.0] * 4)
        print("  Per-class F1:")
        for name, f1 in zip(class_names, test_f1_per_class):
            print(f"    {name}: {f1:.4f}")
        print("="*70 + "\n")
        
        train_dict = {
            "best_val_macro_f1": best_metric,
            "final_val_acc": final_val.get("acc", None),
            "final_val_macro_f1": final_val.get("macro_f1", None),
            "final_val_event_f1": final_val.get("event_f1", None),
            "final_val_f1_decrease": val_f1_per_class[0] if len(val_f1_per_class) > 0 else None,
            "final_val_f1_stable": val_f1_per_class[1] if len(val_f1_per_class) > 1 else None,
            "final_val_f1_increase": val_f1_per_class[2] if len(val_f1_per_class) > 2 else None,
            "final_val_f1_entry": val_f1_per_class[3] if len(val_f1_per_class) > 3 else None,
            "final_test_acc": final_test.get("acc", None),
            "final_test_macro_f1": final_test.get("macro_f1", None),
            "final_test_event_f1": final_test.get("event_f1", None),
            "final_test_f1_decrease": test_f1_per_class[0] if len(test_f1_per_class) > 0 else None,
            "final_test_f1_stable": test_f1_per_class[1] if len(test_f1_per_class) > 1 else None,
            "final_test_f1_increase": test_f1_per_class[2] if len(test_f1_per_class) > 2 else None,
            "final_test_f1_entry": test_f1_per_class[3] if len(test_f1_per_class) > 3 else None,
        }
else:
    print("\n" + "="*60)
    print("LINK TASK SELECTED")
    print("="*60)
    # Expect train/val/test_dataset in Ecomm style: list of (graph, eval_rel) with edge_label_index/edge_label
    train_data = getattr(dataset, "train_dataset", None)
    val_pair = getattr(dataset, "val_dataset", None)
    test_pair = getattr(dataset, "test_dataset", None)
    if train_data is None or val_pair is None or test_pair is None or optimizer is None:
        print("[WARN] Missing train/val/test datasets or optimizer for link task; skipping.")
    else:
        # Check if model is temporal (HTGNN, DyHATR, DHSpace, etc.)
        is_temporal_model = args.model in ["HTGNN", "DyHATR"] or "DHSpace" in args.model
        print(f"Model: {args.model}, Is Temporal: {is_temporal_model}")
        
        bce = torch.nn.BCEWithLogitsLoss()
        best_metric = -float("inf")
        best_state = None
        patience_ctr = 0

        def eval_split(pairs):
            # pairs can be a tuple or list of tuples
            if isinstance(pairs, tuple) and len(pairs) == 2 and not isinstance(pairs[0], (list, tuple)):
                pairs = [pairs]
            metrics_accum = []
            model.eval()
            with torch.no_grad():
                for g_or_gs, eval_rel in pairs:
                    edge_label_index = eval_rel.edge_label_index.to(args.device)
                    edge_label = eval_rel.edge_label.to(args.device)
                    pos_mask = edge_label == 1
                    neg_mask = edge_label == 0
                    pos_edges = edge_label_index[:, pos_mask]
                    neg_edges = edge_label_index[:, neg_mask]
                    
                    # Handle temporal vs single-graph models
                    if is_temporal_model:
                        if isinstance(g_or_gs, list):
                            gs = [g.to(args.device) for g in g_or_gs]
                        elif isinstance(g_or_gs, dict):
                            gs = {k: g.to(args.device) for k, g in g_or_gs.items()}
                        else:
                            gs = g_or_gs.to(args.device)
                        z = model.encode(gs)
                    else:
                        g = g_or_gs.to(args.device)
                        z = model.encode(g)
                    
                    pos_score = model.decode(z, pos_edges)
                    neg_score = model.decode(z, neg_edges)
                    scores = torch.cat([pos_score, neg_score], dim=0)
                    labels = torch.cat(
                        [
                            torch.ones_like(pos_score, dtype=torch.float32),
                            torch.zeros_like(neg_score, dtype=torch.float32),
                        ],
                        dim=0,
                    )
                    from evaluation_metrics import evaluate_link_predictions
                    metrics = evaluate_link_predictions(scores, labels)
                    metrics_accum.append(metrics)
            # average metrics
            if len(metrics_accum) == 0:
                return {}
            keys = metrics_accum[0].keys()
            return {k: float(np.mean([m[k] for m in metrics_accum])) for k in keys}

        for epoch in range(args.max_epochs):
            model.train()
            total_loss = 0.0
            batches = train_data
            # resample negatives if dataset mutates per access
            for g_or_gs, eval_rel in batches:
                edge_label_index = eval_rel.edge_label_index
                edge_label = eval_rel.edge_label
                pos_mask = edge_label == 1
                neg_mask = edge_label == 0
                pos_edges = edge_label_index[:, pos_mask].to(args.device)
                neg_edges = edge_label_index[:, neg_mask].to(args.device)
                
                optimizer.zero_grad()
                
                # Handle temporal vs single-graph models
                if is_temporal_model:
                    # For temporal models, expect list or dict of graphs
                    if isinstance(g_or_gs, list):
                        gs = [g.to(args.device) for g in g_or_gs]
                    elif isinstance(g_or_gs, dict):
                        gs = {k: g.to(args.device) for k, g in g_or_gs.items()}
                    else:
                        gs = g_or_gs.to(args.device)
                    z = model.encode(gs)
                else:
                    # For single-graph models
                    g = g_or_gs.to(args.device)
                    z = model.encode(g)
                
                pos_score = model.decode(z, pos_edges)
                neg_score = model.decode(z, neg_edges)
                scores = torch.cat([pos_score, neg_score], dim=0)
                labels = torch.cat(
                    [
                        torch.ones_like(pos_score, dtype=torch.float32),
                        torch.zeros_like(neg_score, dtype=torch.float32),
                    ],
                    dim=0,
                )
                loss = bce(scores, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_metrics = eval_split(val_pair)
            val_auc = val_metrics.get("AUC", float("-inf"))
            print(
                f"Epoch {epoch} link loss {total_loss/max(len(batches),1):.4f} "
                f"val_AUC {val_auc:.4f}"
            )
            # NODE-LEVEL PROSPECTUS MODULATION: per-epoch diagnostics
            if getattr(args, 'use_prospectus_node', False):
                _base = model.base_model if hasattr(model, 'base_model') else model
                if hasattr(_base, 'log_prospectus_diagnostics'):
                    _base.log_prospectus_diagnostics(prefix=f" epoch={epoch}")
            if val_auc > best_metric:
                best_metric = val_auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        final_val = eval_split(val_pair)
        final_test = eval_split(test_pair)
        print(f"Best Val AUC: {best_metric:.4f}")
        print(f"Final Val metrics: {final_val}")
        print(f"Final Test metrics: {final_test}")
        train_dict = {
            "best_val_auc": best_metric,
            "final_val_auc": final_val.get("AUC", None),
            "final_val_ap": final_val.get("AP", None),
            "final_test_auc": final_test.get("AUC", None),
            "final_test_ap": final_test.get("AP", None),
        }
        # SAVE MODEL
        ckpt_path = os.path.join(args.log_dir, "stage1_best.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved best Stage 1 model to {ckpt_path}")

# close
from core.trainer import log_train

log_train(args.log_dir, args, train_dict, None)
