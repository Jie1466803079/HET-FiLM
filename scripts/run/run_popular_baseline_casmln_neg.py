#!/usr/bin/env python3
"""Wrapper that runs scripts/diagnostics/popular_baseline.py with the
CasMLN-style training-universe negative sampler substituted for the
default target-snapshot sampler.

Same patching pattern as scripts/run/run_xgboost_baseline_casmln_neg.py.
"""
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data import funds_edge_weight as _fw_module
from core.data.funds_edge_weight_casmln import FundsEdgeWeightDatasetCasMLN

print("[run_popular_baseline_casmln_neg] Patching FundsEdgeWeightDataset → "
      "FundsEdgeWeightDatasetCasMLN", flush=True)
_fw_module.FundsEdgeWeightDataset = FundsEdgeWeightDatasetCasMLN

target = PROJECT_ROOT / "scripts" / "diagnostics" / "popular_baseline.py"
runpy.run_path(str(target), run_name="__main__")
