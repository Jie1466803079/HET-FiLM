import argparse
from core.models import Sta_MODEL, Homo_MODEL
import os


def setargs(args, hp):
    for k, v in hp.items():
        # Preserve CLI-provided twin to ensure search uses requested temporal window
        if k == "twin" and getattr(args, "twin", -1) != -1:
            continue
        setattr(args, k, v)


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cfg", type=int, default=1)
    # basic
    parser.add_argument("--dataset", type=str, default="Aminer")
    parser.add_argument("--task", type=str, default="regression")
    parser.add_argument("--model", type=str, default="DHSpace")
    parser.add_argument("--dhconfig", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="logs/tmp")
    parser.add_argument("--device", default="6")
    parser.add_argument("--seed", type=int, default=22)

    # auto
    parser.add_argument("--dynamic", type=int, default=-1)
    parser.add_argument("--homo", type=int, default=-1)
    parser.add_argument("--twin", type=int, default=-1)
    parser.add_argument("--test_full", type=int, default=-1)
    parser.add_argument("--predict_type", type=str, default="")
    parser.add_argument("--in_dim", type=int, default=-1)
    parser.add_argument("--hid_dim", type=int, default=-1)
    parser.add_argument("--out_dim", type=int, default=-1)
    parser.add_argument("--num_classes", type=int, default=-1)

    # optim
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--cul", type=int, default=1)

    # hp
    parser.add_argument("--norm", type=int, default=1)
    parser.add_argument("--hlinear_act", type=str, default="tanh")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--grad_clip", type=float, default=0)
    
    # DHSpaceGRU specific
    parser.add_argument("--gate_hidden", type=int, default=16, help="EdgeGRU hidden size for DHSpaceGRU")
    parser.add_argument("--aux_lambda", type=float, default=0.2, help="Auxiliary loss weight for DHSpaceGRU")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # search
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--supernet_early_stop", type=int, default=1000)
    parser.add_argument("--causal_mask", type=int, default=1, help="1 True else False")
    parser.add_argument("--node_entangle_type", type=str, default="None")
    parser.add_argument("--rel_entangle_type", type=str, default="None")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--rel_time_type", type=str, default="relative")
    parser.add_argument("--hupdate", type=int, default=1)
    parser.add_argument("--reset_type", type=int, default=0)
    parser.add_argument("--reset_type2", type=int, default=0)
    parser.add_argument("--patch_num", type=int, default=1)
    parser.add_argument("--KN", type=int, default=2)
    parser.add_argument("--KR", type=int, default=2)
    parser.add_argument("--KTO", type=int, default=10)
    parser.add_argument("--n_warmup", type=int, default=40)
    parser.add_argument("--arch_dir", type=str, default="")
    parser.add_argument("--supernet_dir", type=str, default="")
    # Meta-path integration (optional)
    parser.add_argument("--use_meta_paths", type=int, default=0, help="Enable meta-path synthetic relations in search")
    parser.add_argument(
        "--meta_paths",
        type=str,
        default="",
        help="Meta-path schemas like 'rel1+rel2;rel3+rel4'. If empty and use_meta_paths=1, model may use defaults",
    )
    # Meta-path pruning/normalization controls (match args_model for retrain)
    parser.add_argument("--mp_topk", type=int, default=0, help="Per-target top-k edges to keep for composed meta edges (0 = no pruning)")
    parser.add_argument("--mp_row_norm", type=str, default="src", help="Row normalization axis for composed meta weights: src or tar")
    # Meta-graph (DiffMG-style) toggles
    parser.add_argument("--mg_enabled", type=int, default=0)
    parser.add_argument("--mg_L", type=int, default=1)
    parser.add_argument("--mg_one_op", type=int, default=1)
    parser.add_argument("--mg_eps_start", type=float, default=0.5)
    parser.add_argument("--mg_eps_end", type=float, default=0.05)
    parser.add_argument("--mg_eps_decay", type=int, default=2000)
    parser.add_argument("--mg_fusion", type=str, default="softmax2")
    parser.add_argument("--mg_time_gate", type=int, default=1)
    parser.add_argument("--mg_time_gate_hidden", type=int, default=32)
    parser.add_argument("--mg_include_metapath_ops", type=int, default=0)
    parser.add_argument("--mg_meta_paths", type=str, default="", help="Macro meta-path ops like 'rev_review+review;rev_interact+interact'")

    # Prospectus text-fusion flags. Accepted by search_model.py so DHSpace
    # NAS jobs that plan to use text at TRAINING time don't crash at the
    # search stage. DHSearcher itself does NOT wire the fusion module — the
    # search is effectively text-free — but the discovered architecture is
    # then retrained via run_model.py which reads these same flags and
    # applies ProspectusText[Concat]Fusion through MultiTaskEdgePredictor.
    parser.add_argument("--use_prospectus", action="store_true")
    parser.add_argument("--prospectus_emb_path", type=str, default="")
    parser.add_argument("--prospectus_fusion_mode", type=str, default="gated", choices=["gated", "concat"])
    parser.add_argument("--risk_weight", type=float, default=0.0)
    parser.add_argument("--no_staleness", action="store_true")
    parser.add_argument("--no_modality_dropout", action="store_true")
    parser.add_argument("--numerical_dim", type=int, default=16)
    parser.add_argument("--mask_invalid_strategy", type=int, default=1)

    args = parser.parse_args(args)

    # full cfg
    if args.use_cfg:
        if args.dataset == "Aminer":
            hp = {
                "patch_num": 2,
                "KN": 5,
                "KR": 4,
                "KTO": 500,
                "n_layers": 3,
                "n_heads": 4,
                "n_warmup": 30,
                "twin": 8,
            }
        elif args.dataset == "Ecomm":
            hp = {
                "patch_num": 2,
                "KN": 5,
                "KR": 3,
                "KTO": 500,
                "n_layers": 2,
                "n_heads": 2,
                "n_warmup": 15,
                "twin": 7,
            }
        elif args.dataset == "Yelp-nc":
            hp = {
                "patch_num": 2,
                "KN": 5,
                "KR": 5,
                "KTO": 500,
                "n_layers": 2,
                "n_heads": 2,
                "n_warmup": 20,
                "twin": 12,
            }
        elif args.dataset == "Funds":
            hp = {
                "patch_num": 2,
                "KN": 4,   # Match DHSpace semantics: KN >= num_types (4)
                "KR": 6,   # Match DHSpace semantics: KR >= num_relations (with manager, 6)
                "KTO": 300,
                "n_layers": 2,
                "n_heads": 4,
                "n_warmup": 15,
                "twin": 8,
            }
        else:
            raise NotImplementedError(f"dataset {args.dataset} not implemented")
        setargs(args, hp)

    # post
    # Allow DHSpace-family searches (DHSpace, NodeGRU, MetaPath, MGCell, MGGlobal)
    assert args.model in ("DHSpace", "DHSpaceNodeGRU", "DHSpaceMP", "DHSpaceMeta", "DHSpaceMGCell", "DHSpaceMGGlobal"), \
        "Use --model DHSpace | DHSpaceNodeGRU | DHSpaceMP | DHSpaceMeta | DHSpaceMGCell | DHSpaceMGGlobal for search"
    args.device = f"cuda:{args.device}"
    args.dynamic = args.model not in Sta_MODEL
    args.homo = args.model in Homo_MODEL
    args.test_full = not args.dynamic  # static model use full training data for testing
    os.makedirs(args.log_dir, exist_ok=True)
    args.supernet_dir = os.path.join(args.log_dir, "supernet/")
    args.arch_dir = os.path.join(args.log_dir, "archs/")
    for d in [args.log_dir, args.supernet_dir, args.arch_dir]:
        os.makedirs(d, exist_ok=True)
    return args
