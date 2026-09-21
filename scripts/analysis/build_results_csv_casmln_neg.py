"""Build results_defb_portfolio_2005q3_casmln_neg.csv from per-seed JSONs.

Aggregates mean +/- std across seeds for each model in MODEL_META, including
the entry-only Top-K precision and within-fund Rank IC metrics added to the
multitask trainer. Two-stage runs read AUC/AP from stage1 and regression
metrics from stage2; joint runs read everything from train_stats.

Existing rows whose Model is not in MODEL_META (e.g. "XGBoost baseline") are
preserved from the existing CSV.

Usage:
    python analysis/build_results_csv_casmln_neg.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analysis.build_per_epoch_metrics_casmln_neg import MODEL_TO_JOB

MODEL_META: dict[str, tuple[str, str]] = {
    "DySAT Joint":               ("Joint",     "No"),
    "CasMLN-NoEW Two-Stage":     ("Two-Stage", "No"),
    "CasMLN-EWMsg Two-Stage":    ("Two-Stage", "EWMsg"),
    "HGT+ Two-Stage":            ("Two-Stage", "Yes"),
    "SE-HTGNN Two-Stage (std)":  ("Two-Stage", "Yes"),
    "SE-HTGNN noew Two-Stage":   ("Two-Stage", "No"),
    "HTGNN Two-Stage Tuned":     ("Two-Stage", "Yes"),
    "CasMLN-WDeg Joint":         ("Joint",     "WDeg"),
    "HTGNN Joint":                ("Joint",     "Yes"),
    "SE-HTGNN Joint (std)":       ("Joint",     "Yes"),
    "HGT+ Joint":                 ("Joint",     "Yes"),
    "HGT Joint":                  ("Joint",     "No"),
    "DHSpace Two-Stage":          ("Two-Stage", "No"),
    "DyHATR Joint (gc=2)":        ("Joint",     "No"),
    "RGCN Joint":                 ("Joint",     "No"),
    "HAN Joint":                  ("Joint",     "No"),
    "GCN Joint":                  ("Joint",     "No"),
    "GAT Joint":                  ("Joint",     "No"),
}

COLS = [
    "Model", "Training", "Edge Weight",
    "Test MAE", "Test MAE std",
    "Test RMSE", "Test RMSE std",
    "Test AUC", "Test AUC std",
    "Test AP", "Test AP std",
    "Test R2", "Test R2 std",
    "Test EntryRankIC", "Test EntryRankIC std",
    "Test EntryP@3", "Test EntryP@3 std",
    "Test EntryP@5", "Test EntryP@5 std",
    "Seeds Done",
]


def _val(d: dict, key: str):
    v = d.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def collect_seed_metrics(repo: Path, model: str) -> list[dict]:
    job, log_dir, has_amp = MODEL_TO_JOB[model]
    base = repo / log_dir / job
    if has_amp:
        base = base / "amp_3"
    rows: list[dict] = []
    if not base.exists():
        return rows
    for seed_dir in sorted(base.glob("seed_*")):
        json_path = seed_dir / "hgt_plus_edge_weight_results.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            jdata = json.load(f)
        ts = jdata.get("train_stats", {}) or {}
        if isinstance(ts, dict) and ("stage1" in ts or "stage2" in ts):
            s1 = ts.get("stage1", {}) or {}
            s2 = ts.get("stage2", {}) or {}
            row = {
                "mae":          _val(s2, "test_mae"),
                "rmse":         _val(s2, "test_rmse"),
                "r2":           _val(s2, "test_r2"),
                "auc":          _val(s1, "test_auc"),
                "ap":           _val(s1, "test_ap"),
                "entry_p3":     _val(s2, "test_entry_precision@3"),
                "entry_p5":     _val(s2, "test_entry_precision@5"),
                "entry_rankic": _val(s2, "test_entry_rank_ic_within_fund"),
            }
        else:
            row = {
                "mae":          _val(ts, "test_mae"),
                "rmse":         _val(ts, "test_rmse"),
                "r2":           _val(ts, "test_r2"),
                "auc":          _val(ts, "test_auc"),
                "ap":           _val(ts, "test_ap"),
                "entry_p3":     _val(ts, "test_entry_precision@3"),
                "entry_p5":     _val(ts, "test_entry_precision@5"),
                "entry_rankic": _val(ts, "test_entry_rank_ic_within_fund"),
            }
        rows.append(row)
    return rows


def _mean_std(values: list[float | None]) -> tuple[float, float]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def build(repo: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for model, (training, edgew) in MODEL_META.items():
        seed_rows = collect_seed_metrics(repo, model)
        n = len(seed_rows)
        if n == 0:
            print(f"  [skip] {model}: no seed JSONs found", file=sys.stderr)
            continue
        cols: dict[str, list] = {k: [r.get(k) for r in seed_rows] for k in seed_rows[0]}
        mae_m, mae_s = _mean_std(cols["mae"])
        rmse_m, rmse_s = _mean_std(cols["rmse"])
        auc_m, auc_s = _mean_std(cols["auc"])
        ap_m, ap_s = _mean_std(cols["ap"])
        r2_m, r2_s = _mean_std(cols["r2"])
        eric_m, eric_s = _mean_std(cols["entry_rankic"])
        ep3_m, ep3_s = _mean_std(cols["entry_p3"])
        ep5_m, ep5_s = _mean_std(cols["entry_p5"])
        rows.append({
            "Model": model, "Training": training, "Edge Weight": edgew,
            "Test MAE": mae_m, "Test MAE std": mae_s,
            "Test RMSE": rmse_m, "Test RMSE std": rmse_s,
            "Test AUC": auc_m, "Test AUC std": auc_s,
            "Test AP": ap_m, "Test AP std": ap_s,
            "Test R2": r2_m, "Test R2 std": r2_s,
            "Test EntryRankIC": eric_m, "Test EntryRankIC std": eric_s,
            "Test EntryP@3": ep3_m, "Test EntryP@3 std": ep3_s,
            "Test EntryP@5": ep5_m, "Test EntryP@5 std": ep5_s,
            "Seeds Done": n,
        })
    return pd.DataFrame(rows, columns=COLS)


def merge_preserved_rows(df: pd.DataFrame, existing_csv: Path) -> pd.DataFrame:
    if not existing_csv.exists():
        return df
    try:
        existing = pd.read_csv(existing_csv)
    except Exception as e:
        print(f"  [warn] could not read existing CSV ({e}); skipping preserve step", file=sys.stderr)
        return df
    known = set(MODEL_META.keys())
    extra = existing[~existing["Model"].isin(known)].copy()
    if extra.empty:
        return df
    for c in COLS:
        if c not in extra.columns:
            extra[c] = float("nan")
    return pd.concat([df, extra[COLS]], ignore_index=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--out", default="analysis/results_defb_portfolio_2005q3_casmln_neg.csv")
    ap.add_argument("--no-preserve", action="store_true",
                    help="Drop rows from the existing CSV that are not in MODEL_META (e.g. XGBoost)")
    args = ap.parse_args()

    repo = Path(args.repo)
    df = build(repo)
    out_path = repo / args.out

    if not args.no_preserve:
        df = merge_preserved_rows(df, out_path)

    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
