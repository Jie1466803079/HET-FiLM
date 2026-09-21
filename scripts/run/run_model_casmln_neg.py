#!/usr/bin/env python3
"""Wrapper that runs scripts/run/run_model.py with the CasMLN-style training-
universe negative sampler substituted for the default target-snapshot sampler.

Usage
-----
Identical CLI to scripts/run/run_model.py — pass through any arguments.

How it works
------------
Before run_model.py imports `FundsEdgeWeightDataset`, this wrapper replaces
the class binding inside `core.data.funds_edge_weight` with the CasMLN
variant from `core.data.funds_edge_weight_casmln`. run_model.py's own
import statement (`from core.data.funds_edge_weight import FundsEdgeWeightDataset`)
then resolves to the patched class — no modification to run_model.py needed.

This pattern keeps existing code, configs, and logs untouched.
"""
import os
import runpy
import sys
from pathlib import Path

# Ensure project root on sys.path so `core` resolves
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Patch the dataset class BEFORE run_model.py is loaded
from core.data import funds_edge_weight as _fw_module
from core.data.funds_edge_weight_casmln import FundsEdgeWeightDatasetCasMLN

print(f"[run_model_casmln_neg] Patching FundsEdgeWeightDataset → "
      f"FundsEdgeWeightDatasetCasMLN", flush=True)
_fw_module.FundsEdgeWeightDataset = FundsEdgeWeightDatasetCasMLN

# === SimTeG (Task 8): bridge CLI args → env vars for the dataset class ===
# The dataset reads SIMTEG_* env vars (see core/data/funds_edge_weight_casmln.py).
# Default "none" → no env vars set → baseline behavior unchanged.
# We parse argv via core.args_model.get_args() so we can read --simteg_* flags
# BEFORE run_model.py constructs its dataset. The downstream argparse inside
# run_model.py still re-parses sys.argv as usual (idempotent).
try:
    from core.args_model import get_args as _get_args
    _pre_args = _get_args()
    if getattr(_pre_args, "simteg_node_features", "none") != "none":
        os.environ["SIMTEG_NODE_FEATURES"] = _pre_args.simteg_node_features
        os.environ["SIMTEG_H5_PATH"] = _pre_args.simteg_h5_path
        print(f"[SimTeG] env bridge: SIMTEG_NODE_FEATURES={_pre_args.simteg_node_features} "
              f"SIMTEG_H5_PATH={_pre_args.simteg_h5_path}", flush=True)
except SystemExit:
    # argparse may SystemExit on --help; let it propagate to the downstream parser.
    raise
except Exception as _e:
    print(f"[SimTeG] env bridge skipped: {_e!r}", flush=True)

# Now invoke the original run_model.py with the patched class in place.
# runpy.run_path executes the target as if it were the main script — argparse
# in run_model.py picks up sys.argv as usual.
target = PROJECT_ROOT / "scripts" / "run" / "run_model.py"
runpy.run_path(str(target), run_name="__main__")
