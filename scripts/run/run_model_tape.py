#!/usr/bin/env python3
"""Wrapper that runs scripts/run/run_model.py with TAPE plumbing patched in.

Builds on the same monkey-patch pattern as scripts/run/run_model_casmln_neg.py
(CasMLN dataset class swap + SimTeG env bridge) and adds:

  1. ProspectusEmbeddingLoader → TAPEEnsembleLoader rebinding inside
     core.data.prospectus_loader. After this, every `from
     core.data.prospectus_loader import ProspectusEmbeddingLoader` in
     run_model.py (lines 511, 651, 688) resolves to the TAPE loader.

  2. TAPEEnsembleLoader.EMB_DIM ← n_channels * per_channel_d. The model
     factory at run_model.py:688 reads this as the text input dim, so the
     model is built with a wider text input projection automatically.

  3. Reads --use_tape_fusion and --tape_emb_paths from sys.argv (via
     parse_known_args), strips them so run_model.py's downstream argparse
     does not choke, and stashes TAPE_EMB_PATHS in the environment for
     the loader to consume.

Existing files are not modified — this is a NEW sibling of
run_model_casmln_neg.py. Pass through any other arguments unchanged.

Usage example (in a PBS file):
    python scripts/run/run_model_tape.py \
        --use_tape_fusion \
        --tape_emb_paths /path/to/TA.h5,/path/to/P.h5,/path/to/E.h5 \
        --model HGT+ --dataset Funds \
        --task link_weight_multitask_new_twostage \
        --use_prospectus --prospectus_emb_path /path/to/TA.h5 \
        ...  # all the usual flags
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

# Ensure project root on sys.path so `core` and `scripts` resolve as packages.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 1) Consume TAPE-specific flags BEFORE delegating to run_model.py ---
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--use_tape_fusion", action="store_true",
                  help="Enable the TAPE ensemble loader (concat TA+P+E channels).")
_pre.add_argument("--tape_emb_paths", type=str, default="",
                  help="Comma-separated H5 paths in TA,P,E order.")
_tape_args, _rest = _pre.parse_known_args(sys.argv[1:])
# Reset sys.argv so run_model.py's argparse never sees the TAPE-only flags.
sys.argv = [sys.argv[0]] + _rest

# --- 2) Reuse CasMLN dataset swap (mirror run_model_casmln_neg.py) ---
from core.data import funds_edge_weight as _fw_module
from core.data.funds_edge_weight_casmln import FundsEdgeWeightDatasetCasMLN
print(f"[run_model_tape] Patching FundsEdgeWeightDataset → "
      f"FundsEdgeWeightDatasetCasMLN", flush=True)
_fw_module.FundsEdgeWeightDataset = FundsEdgeWeightDatasetCasMLN

# --- 3) Reuse SimTeG env bridge (mirror run_model_casmln_neg.py) ---
try:
    from core.args_model import get_args as _get_args
    _pre_args = _get_args()
    if getattr(_pre_args, "simteg_node_features", "none") != "none":
        os.environ["SIMTEG_NODE_FEATURES"] = _pre_args.simteg_node_features
        os.environ["SIMTEG_H5_PATH"] = _pre_args.simteg_h5_path
        print(f"[SimTeG] env bridge: SIMTEG_NODE_FEATURES="
              f"{_pre_args.simteg_node_features} "
              f"SIMTEG_H5_PATH={_pre_args.simteg_h5_path}", flush=True)
except SystemExit:
    # argparse may SystemExit on --help; let it propagate.
    raise
except Exception as _e:
    print(f"[SimTeG] env bridge skipped: {_e!r}", flush=True)

# --- 4) TAPE plumbing ---
if _tape_args.use_tape_fusion:
    paths_str = (_tape_args.tape_emb_paths
                 or os.environ.get("TAPE_EMB_PATHS", ""))
    paths = [p.strip() for p in paths_str.split(",") if p.strip()]
    if len(paths) < 2:
        print(f"[TAPE] --use_tape_fusion requires --tape_emb_paths "
              f"or TAPE_EMB_PATHS env var with >=2 H5 paths; "
              f"got {paths_str!r}", file=sys.stderr)
        sys.exit(2)
    # Stash for the loader to read at construction time.
    os.environ["TAPE_EMB_PATHS"] = ",".join(paths)

    # Peek channel-0 width so we can set EMB_DIM correctly BEFORE the
    # ProspectusEmbeddingLoader class binding is read by run_model.py:688.
    import h5py  # noqa: E402
    with h5py.File(paths[0], "r") as f:
        per_channel_d = int(f["strategy_emb"].shape[-1])
    target_emb_dim = per_channel_d * len(paths)

    # Rebind the class in the prospectus_loader module so `from
    # core.data.prospectus_loader import ProspectusEmbeddingLoader` returns
    # TAPEEnsembleLoader.
    from core.data import prospectus_loader as _pl
    from core.data.tape_ensemble_loader import TAPEEnsembleLoader
    TAPEEnsembleLoader.EMB_DIM = target_emb_dim
    _pl.ProspectusEmbeddingLoader = TAPEEnsembleLoader
    # ProspectusTextFusion.abs_proj is built from the class-level INPUT_DIM at
    # __init__ time, and the constructor takes no input_dim arg. Patch the
    # class attribute here so the first Linear becomes (target_emb_dim, 256)
    # instead of the default (1024, 256). Only effective when TAPE is on.
    from core.models.prospectus_fusion import ProspectusTextFusion
    ProspectusTextFusion.INPUT_DIM = target_emb_dim
    print(f"[TAPE] Patched ProspectusEmbeddingLoader → TAPEEnsembleLoader "
          f"with EMB_DIM={target_emb_dim} "
          f"({len(paths)} channels × {per_channel_d}-d) "
          f"from:\n  " + "\n  ".join(paths), flush=True)
    print(f"[TAPE] Patched ProspectusTextFusion.INPUT_DIM "
          f"= {target_emb_dim} (was 1024)", flush=True)
else:
    print(f"[TAPE] --use_tape_fusion not set; baseline path unchanged.",
          flush=True)

# --- 5) Delegate to run_model.py ---
target = PROJECT_ROOT / "scripts" / "run" / "run_model.py"
runpy.run_path(str(target), run_name="__main__")
