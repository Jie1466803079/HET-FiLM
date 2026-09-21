#!/usr/bin/env python3
"""Statistical significance tests for mutual fund prediction experiments.

Reads per-seed results from model log directories, runs paired t-tests,
and reports p-values with significance markers.

Usage:
    python analysis/significance_tests.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ── Model definitions ──────────────────────────────────────────────────────────

LOG_ROOT = Path(__file__).parent / "logs"

MODELS = {
    "HGT+ Two-Stage+JL": {
        "dir": "hgtplus_lpnew_jl_defb_portfolio",
        "type": "two-stage",
    },
    "HTGNN Joint (tuned)": {
        "dir": "htgnn_joint_defb_portfolio",
        "type": "joint",
    },
    "HGT+ Joint": {
        "dir": "hgtplus_joint_defb_portfolio",
        "type": "joint",
    },
    "HGT Joint": {
        "dir": "hgt_joint_defb_portfolio",
        "type": "joint",
    },
    "HTGNN Two-Stage+JL (tuned)": {
        "dir": "htgnn_lpnew_jl_tuned_defb_portfolio",
        "type": "two-stage",
    },
    "SE-HTGNN base": {
        "dir": "sehtgnn_joint_defb_portfolio",
        "type": "joint",
    },
    "DHSpace Joint (tuned)": {
        "dir": "dhspace_joint_defb_portfolio",
        "type": "joint",
    },
    "SE-HTGNN base (no EW-GCN)": {
        "dir": "sehtgnn_joint_defb_portfolio_no_ewgcn",
        "type": "joint",
    },
    "DHSpace Two-Stage+JL (tuned)": {
        "dir": "dhspace_lpnew_jl_defb_portfolio",
        "type": "two-stage",
    },
    # Will be added once PBS job completes:
    # "HGT Two-Stage+JL": {
    #     "dir": "hgt_lpnew_jl_defb_portfolio",
    #     "type": "two-stage",
    # },
}

METRICS = ["mae", "rmse", "auc", "ap", "r2"]
SEEDS = [0, 1, 2, 3, 4]


# ── Data loading ───────────────────────────────────────────────────────────────


def load_seed_metrics(model_name: str, model_info: dict) -> Optional[Dict[str, List[float]]]:
    """Load per-seed metrics for a model. Returns dict of metric -> list of values."""
    model_dir = LOG_ROOT / model_info["dir"]
    model_type = model_info["type"]
    results = {m: [] for m in METRICS}

    for seed in SEEDS:
        seed_dir = model_dir / f"seed_{seed}"
        json_files = list(seed_dir.glob("*edge_weight_results.json"))
        if not json_files:
            print(f"  WARNING: No results JSON in {seed_dir}", file=sys.stderr)
            return None
        with open(json_files[0]) as f:
            data = json.load(f)
        ts = data.get("train_stats", {})

        if model_type == "joint":
            for m in METRICS:
                val = ts.get(f"test_{m}")
                if val is None:
                    print(f"  WARNING: Missing test_{m} for {model_name} seed {seed}", file=sys.stderr)
                    return None
                results[m].append(float(val))
        elif model_type == "two-stage":
            stage1 = ts.get("stage1", {})
            stage2 = ts.get("stage2", {})
            # AUC and AP come from stage1 (link prediction)
            for m in ["auc", "ap"]:
                val = stage1.get(f"test_{m}")
                if val is None:
                    print(f"  WARNING: Missing stage1 test_{m} for {model_name} seed {seed}", file=sys.stderr)
                    return None
                results[m].append(float(val))
            # MAE, RMSE, R2 come from stage2 (edge weight regression)
            for m in ["mae", "rmse", "r2"]:
                val = stage2.get(f"test_{m}")
                if val is None:
                    print(f"  WARNING: Missing stage2 test_{m} for {model_name} seed {seed}", file=sys.stderr)
                    return None
                results[m].append(float(val))

    return results


def load_all_models() -> Dict[str, Dict[str, List[float]]]:
    """Load metrics for all models. Returns {model_name: {metric: [values]}}."""
    all_data = {}
    for name, info in MODELS.items():
        model_dir = LOG_ROOT / info["dir"]
        if not model_dir.exists():
            print(f"  Skipping {name}: directory not found ({model_dir})", file=sys.stderr)
            continue
        data = load_seed_metrics(name, info)
        if data is not None:
            all_data[name] = data
    return all_data


# ── Statistical tests ──────────────────────────────────────────────────────────


def significance_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Return (t-statistic, p-value) from paired t-test."""
    a_arr, b_arr = np.array(a), np.array(b)
    if np.allclose(a_arr, b_arr):
        return 0.0, 1.0
    result = stats.ttest_rel(a_arr, b_arr)
    return float(result.statistic), float(result.pvalue)


def run_comparison(
    all_data: Dict[str, Dict[str, List[float]]],
    model_a: str,
    model_b: str,
    metrics: Optional[List[str]] = None,
) -> Optional[Dict[str, dict]]:
    """Run paired t-tests between two models on specified metrics."""
    if model_a not in all_data or model_b not in all_data:
        return None
    if metrics is None:
        metrics = METRICS
    results = {}
    for m in metrics:
        vals_a = all_data[model_a][m]
        vals_b = all_data[model_b][m]
        t_stat, p_val = paired_ttest(vals_a, vals_b)
        mean_a = np.mean(vals_a)
        mean_b = np.mean(vals_b)
        results[m] = {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "diff": mean_a - mean_b,
            "t_stat": t_stat,
            "p_value": p_val,
            "sig": significance_marker(p_val),
        }
    return results


# ── Output formatting ─────────────────────────────────────────────────────────


def print_comparison_table(
    title: str,
    comparisons: List[Tuple[str, str]],
    all_data: Dict[str, Dict[str, List[float]]],
    metrics: Optional[List[str]] = None,
):
    """Print a formatted comparison table."""
    if metrics is None:
        metrics = METRICS

    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    # Header
    metric_headers = "".join(f"{'':>4}{m.upper():>8} {'p':>8} {'':>4}" for m in metrics)
    print(f"\n{'Comparison':<45}{metric_headers}")
    print("-" * (45 + len(metrics) * 24))

    for model_a, model_b in comparisons:
        if model_a not in all_data or model_b not in all_data:
            label = f"{model_a} vs {model_b}"
            print(f"{label:<45}  (data not available)")
            continue

        results = run_comparison(all_data, model_a, model_b, metrics)
        if results is None:
            continue

        label = f"{model_a} vs {model_b}"
        if len(label) > 43:
            label = label[:43] + ".."

        row = ""
        for m in metrics:
            r = results[m]
            diff = r["diff"]
            p = r["p_value"]
            sig = r["sig"]
            # For MAE/RMSE, negative diff means model_a is better (lower)
            # For AUC/AP/R2, positive diff means model_a is better (higher)
            row += f"  {diff:>+8.4f} {p:>8.4f}{sig:<4}"
        print(f"{label:<45}{row}")

    print()


def print_model_summary(all_data: Dict[str, Dict[str, List[float]]]):
    """Print mean ± std for all models."""
    print(f"\n{'=' * 100}")
    print("  Model Summary (mean ± std over 5 seeds)")
    print(f"{'=' * 100}")

    header = f"{'Model':<35}" + "".join(f"{'':>2}{m.upper():>16}" for m in METRICS)
    print(f"\n{header}")
    print("-" * (35 + len(METRICS) * 18))

    # Sort by MAE (ascending)
    sorted_models = sorted(
        all_data.items(), key=lambda x: np.mean(x[1]["mae"])
    )
    for name, data in sorted_models:
        row = f"{name:<35}"
        for m in METRICS:
            vals = data[m]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            row += f"  {mean:>7.4f}±{std:<7.4f}"
        print(row)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    print("Loading per-seed results from:", LOG_ROOT)
    print()

    all_data = load_all_models()
    if not all_data:
        print("ERROR: No model data loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(all_data)} models: {', '.join(all_data.keys())}")

    # ── Summary table ──
    print_model_summary(all_data)

    # ── 1. RTE Ablation (HGT+ vs HGT, same training regime) ──
    print_comparison_table(
        "1. RTE Ablation: HGT+ (with RTE) vs HGT (no RTE)",
        [
            ("HGT+ Joint", "HGT Joint"),
            ("HGT+ Two-Stage+JL", "HGT Two-Stage+JL"),
        ],
        all_data,
    )

    # ── 2. Training Regime: Joint vs Two-Stage+JL ──
    print_comparison_table(
        "2. Training Regime: Joint vs Two-Stage+JL (positive diff = Joint higher)",
        [
            ("HGT+ Joint", "HGT+ Two-Stage+JL"),
            ("HGT Joint", "HGT Two-Stage+JL"),
            ("HTGNN Joint (tuned)", "HTGNN Two-Stage+JL (tuned)"),
            ("DHSpace Joint (tuned)", "DHSpace Two-Stage+JL (tuned)"),
        ],
        all_data,
    )

    # ── 3. Best Model (HGT+ Two-Stage+JL) vs All Others ──
    best = "HGT+ Two-Stage+JL"
    others = [n for n in all_data if n != best]
    # Sort others by MAE
    others.sort(key=lambda n: np.mean(all_data[n]["mae"]))
    print_comparison_table(
        f"3. Best Model ({best}) vs All Others (negative MAE diff = best is better)",
        [(best, o) for o in others],
        all_data,
    )

    # ── 4. Architecture Comparisons (same training regime) ──
    print_comparison_table(
        "4. Architecture Comparisons (same training regime)",
        [
            # Joint regime
            ("HGT+ Joint", "HTGNN Joint (tuned)"),
            ("HGT+ Joint", "DHSpace Joint (tuned)"),
            ("HGT+ Joint", "SE-HTGNN base"),
            ("HTGNN Joint (tuned)", "DHSpace Joint (tuned)"),
            ("HTGNN Joint (tuned)", "SE-HTGNN base"),
            ("SE-HTGNN base", "SE-HTGNN base (no EW-GCN)"),
            # Two-stage regime
            ("HGT+ Two-Stage+JL", "HTGNN Two-Stage+JL (tuned)"),
            ("HGT+ Two-Stage+JL", "DHSpace Two-Stage+JL (tuned)"),
            ("HTGNN Two-Stage+JL (tuned)", "DHSpace Two-Stage+JL (tuned)"),
        ],
        all_data,
    )

    # ── Print significance legend ──
    print("=" * 100)
    print("  Significance: * p<0.05, ** p<0.01, *** p<0.001 (paired t-test, 5 seeds)")
    print("  Diff = Model_A - Model_B. For MAE/RMSE: negative diff = A is better.")
    print("  For AUC/AP/R2: positive diff = A is better.")
    print("=" * 100)


if __name__ == "__main__":
    main()
