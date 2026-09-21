#!/usr/bin/env python3
"""Aggregate multi-seed experiment results and report mean +/- SD.

Scans analysis/logs/*/seed_*/hgt_plus_edge_weight_results.json for all
model×config combinations and prints a formatted table.

Usage:
    python analysis/aggregate_multiseed.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np


# Map from log directory prefix -> (model, config_label)
LOG_DIR_MAP = {
    # HGT+
    "pct_twostage_adaptive":          ("HGT+", "pct / SCALE_ADAPTIVE"),
    "pct_twostage_sigmoid":           ("HGT+", "pct / SCALE_SIGMOID"),
    "pct_twostage_loss":              ("HGT+", "pct / SCALE_LOSS"),
    "logmv_twostage_mse":             ("HGT+", "logmv / MSE"),
    "logmv_twostage_adaptive":        ("HGT+", "logmv / ADAPTIVE"),
    "logmv_twostage_mse_nostd":       ("HGT+", "logmv / MSE (nostd)"),
    "logmv_twostage_adaptive_nostd":  ("HGT+", "logmv / ADAPTIVE (nostd)"),
    # HTGNN
    "htgnn_pct_twostage_adaptive":          ("HTGNN", "pct / SCALE_ADAPTIVE"),
    "htgnn_pct_twostage_sigmoid":           ("HTGNN", "pct / SCALE_SIGMOID"),
    "htgnn_pct_twostage_loss":              ("HTGNN", "pct / SCALE_LOSS"),
    "htgnn_logmv_twostage_mse":             ("HTGNN", "logmv / MSE"),
    "htgnn_logmv_twostage_adaptive":        ("HTGNN", "logmv / ADAPTIVE"),
    "htgnn_logmv_twostage_mse_nostd":       ("HTGNN", "logmv / MSE (nostd)"),
    "htgnn_logmv_twostage_adaptive_nostd":  ("HTGNN", "logmv / ADAPTIVE (nostd)"),
    # DHSpace
    "dhgas_pct_twostage_adaptive":          ("DHSpace", "pct / SCALE_ADAPTIVE"),
    "dhgas_pct_twostage_sigmoid":           ("DHSpace", "pct / SIGMOID"),
    "dhgas_pct_twostage_loss":              ("DHSpace", "pct / SCALE_LOSS"),
    "dhgas_logmv_twostage_mse":             ("DHSpace", "logmv / MSE"),
    "dhgas_logmv_twostage_adaptive":        ("DHSpace", "logmv / ADAPTIVE"),
    "dhgas_logmv_twostage_mse_nostd":       ("DHSpace", "logmv / MSE (nostd)"),
    "dhgas_logmv_twostage_adaptive_nostd":  ("DHSpace", "logmv / ADAPTIVE (nostd)"),
}

METRICS = [
    ("Stage 1 AUC",  "stage1", "test_auc"),
    ("Stage 1 AP",   "stage1", "test_ap"),
    ("Stage 2 MAE",  "stage2", "test_mae"),
    ("Stage 2 RMSE", "stage2", "test_rmse"),
]


def find_results(logs_dir: Path):
    """Scan for result JSONs and group by config directory."""
    results = defaultdict(list)  # config_dir_name -> list of dicts

    for config_dir in sorted(logs_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        dir_name = config_dir.name
        if dir_name not in LOG_DIR_MAP:
            continue

        for seed_dir in sorted(config_dir.glob("seed_*")):
            json_path = seed_dir / "hgt_plus_edge_weight_results.json"
            if not json_path.exists():
                continue
            try:
                with open(json_path) as f:
                    data = json.load(f)
                results[dir_name].append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"WARNING: Failed to read {json_path}: {e}", file=sys.stderr)

    return results


def extract_metric(data, stage_key, metric_key):
    """Extract a metric from the result JSON."""
    train_stats = data.get("train_stats", {})
    stage = train_stats.get(stage_key, {})
    return stage.get(metric_key, None)


def fmt_mean_sd(values):
    """Format as mean +/- sd with 4 decimal places."""
    if not values:
        return "  --  "
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    if len(arr) == 1:
        return f"{mean:.4f}"
    sd = np.std(arr, ddof=1)
    return f"{mean:.4f} +/- {sd:.4f}"


def print_table(results):
    """Print formatted results table grouped by model."""
    model_order = ["HGT+", "HTGNN", "DHSpace"]
    metric_names = [m[0] for m in METRICS]

    # Header
    col_w = 22
    config_w = 28
    print()
    header = f"{'Model':<10} {'Config':<{config_w}} {'Seeds':>5}"
    for mn in metric_names:
        header += f"  {mn:>{col_w}}"
    print(header)
    print("-" * len(header))

    for model in model_order:
        printed_any = False
        for dir_name, (m, config_label) in sorted(LOG_DIR_MAP.items()):
            if m != model:
                continue
            if dir_name not in results or not results[dir_name]:
                continue

            seed_data = results[dir_name]
            n_seeds = len(seed_data)

            row = f"{model:<10} {config_label:<{config_w}} {n_seeds:>5}"

            for _, stage_key, metric_key in METRICS:
                values = []
                for d in seed_data:
                    v = extract_metric(d, stage_key, metric_key)
                    if v is not None:
                        values.append(v)
                row += f"  {fmt_mean_sd(values):>{col_w}}"

            print(row)
            printed_any = True

        if printed_any:
            print()

    print("Note: SD uses Bessel's correction (ddof=1).")


def main():
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir / "logs"

    if not logs_dir.exists():
        print(f"ERROR: Logs directory not found: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    results = find_results(logs_dir)

    if not results:
        print("No results found. Check that seed directories contain "
              "hgt_plus_edge_weight_results.json files.")
        sys.exit(1)

    total_configs = len(results)
    total_seeds = sum(len(v) for v in results.values())
    print(f"\nFound {total_seeds} result files across {total_configs} configs.")

    print_table(results)


if __name__ == "__main__":
    main()
