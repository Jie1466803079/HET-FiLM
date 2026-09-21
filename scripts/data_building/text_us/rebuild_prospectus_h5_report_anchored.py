#!/usr/bin/env python3
"""Rebuild the prospectus H5 with report_dt anchor (no OpenAI calls).

See docs/plans/2026-05-20-prospectus-report-anchor-switch-design.md.

Run:
    cd /srv/scratch/dbgcse/jieliu/mutual_fund_prediction
    /srv/scratch/dbgcse/jieliu/anaconda3/envs/dhgas/bin/python \\
        mutual_fund_prediction/scripts/rebuild_prospectus_h5_report_anchored.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd

try:
    # When run as `python scripts/rebuild_prospectus_h5_report_anchored.py`
    # (cwd-based script execution) or with the scripts/ dir on PYTHONPATH.
    from _prospectus_report_anchor import (
        MAX_LAG, apply_anchor_policy, build_target_q_vector,
        reconstruct_filings, re_locf_anchored, resolve_max_lag,
    )
except ModuleNotFoundError:
    # When imported as `scripts.rebuild_prospectus_h5_report_anchored`
    # (e.g. from tests with the project root on sys.path).
    from scripts._prospectus_report_anchor import (  # type: ignore[no-redef]
        MAX_LAG, apply_anchor_policy, build_target_q_vector,
        reconstruct_filings, re_locf_anchored, resolve_max_lag,
    )

DEFAULT_IN_H5  = "sec_filings_project/embeddings_openai/prospectus_embeddings.h5"
DEFAULT_OUT_H5 = "sec_filings_project/embeddings_openai/prospectus_embeddings_report_anchored.h5"
DEFAULT_AUDIT  = "sec_filings_project/embeddings_openai/prospectus_report_anchor_audit.csv"
DEFAULT_LOOKUP = "mutual_fund_prediction/analysis/fund_effq_to_reportq.csv"
DEFAULT_MAP    = "sec_filings_project/fund_mapping_since2005Q3.csv"


def parse_portno_list(s: str) -> List[int]:
    return [int(x) for x in str(s).split("|")]


def load_lookup_by_portno(path: Path) -> Dict[int, Dict[str, str]]:
    df = pd.read_csv(path)
    out: Dict[int, Dict[str, str]] = {}
    for portno, eff_q, rpt_q in zip(df["crsp_portno"].astype(int),
                                    df["eff_q"], df["rpt_q"]):
        out.setdefault(int(portno), {})[str(eff_q)] = str(rpt_q)
    return out


def merge_lookups(portnos: List[int],
                  by_portno: Dict[int, Dict[str, str]]) -> Dict[str, str]:
    """Combine a fund's per-portno lookups; same eff_q across portnos →
    take the LATER report_q (the more recent reporting event)."""
    merged: Dict[str, str] = {}
    for p in portnos:
        d = by_portno.get(p)
        if not d:
            continue
        for eff_q, rpt_q in d.items():
            if eff_q not in merged or rpt_q > merged[eff_q]:
                merged[eff_q] = rpt_q
    return merged


def run(args) -> None:
    in_h5 = Path(args.in_h5)
    out_h5 = Path(args.out_h5)
    print(f"[IN] {in_h5}")
    print(f"[OUT] {out_h5}")

    by_portno = load_lookup_by_portno(Path(args.lookup))
    print(f"  lookup: {sum(len(v) for v in by_portno.values()):,} pairs"
          f" across {len(by_portno):,} portnos")
    mapping = pd.read_csv(args.mapping)
    id2portnos = {int(r.id_idx): parse_portno_list(r.crsp_portno_list)
                  for r in mapping.itertuples(index=False)}
    print(f"  id_idx→portno mapping: {len(id2portnos):,} ids")

    # Streaming I/O: the per-user cgroup caps RSS at 4 GiB, so we cannot load
    # the full (3166, 65, 1024) float32 arrays (~800 MB each) for both source
    # and destination simultaneously. We open both H5s, read/write one fund
    # row at a time, and stream-compute the per-quarter cosine_sim/delta_emb.
    import os
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_h5 = out_h5.with_suffix(out_h5.suffix + ".tmp")

    audit_rows = []
    coverage_rows = []    # per-cell — populated below
    n_lookup_dropped_oob = 0

    with h5py.File(in_h5, "r") as fin, h5py.File(tmp_h5, "w") as fout:
        N_FUNDS, N_Q, EMB = fin["strategy_emb"].shape
        assert fin["risk_emb"].shape == fin["strategy_emb"].shape
        print(f"  H5 shape: ({N_FUNDS}, {N_Q}, {EMB})")

        quarters_raw = fin["snapshot_quarters"][:]
        quarters = [q.decode() if isinstance(q, bytes) else str(q)
                    for q in quarters_raw]
        quarter_to_idx = {q: i for i, q in enumerate(quarters)}
        fund_ids_arr = fin["fund_ids"][:]

        # Pre-create destination datasets (chunked per-fund for streaming writes).
        d_strategy = fout.create_dataset(
            "strategy_emb", shape=(N_FUNDS, N_Q, EMB), dtype=np.float32,
            chunks=(1, N_Q, EMB))
        d_risk = fout.create_dataset(
            "risk_emb", shape=(N_FUNDS, N_Q, EMB), dtype=np.float32,
            chunks=(1, N_Q, EMB))
        d_abs = fout.create_dataset(
            "abs_emb", shape=(N_FUNDS, N_Q, EMB), dtype=np.float32,
            chunks=(1, N_Q, EMB))
        d_delta_emb = fout.create_dataset(
            "delta_emb", shape=(N_FUNDS, N_Q, EMB), dtype=np.float32,
            chunks=(1, N_Q, EMB))
        d_objective = fout.create_dataset(
            "objective_emb", shape=(N_FUNDS, N_Q, EMB), dtype=np.float32,
            chunks=(1, N_Q, EMB))
        d_delta_t = fout.create_dataset(
            "delta_t", shape=(N_FUNDS, N_Q), dtype=np.int8)
        d_has_text = fout.create_dataset(
            "has_text", shape=(N_FUNDS, N_Q), dtype=bool)
        d_cosine = fout.create_dataset(
            "cosine_sim", shape=(N_FUNDS, N_Q), dtype=np.float32)

        # Carry through anchor-agnostic datasets (fund_ids, snapshot_quarters).
        REWRITTEN_KEYS = {
            "strategy_emb", "risk_emb", "delta_t", "has_text",
            "abs_emb", "delta_emb", "cosine_sim", "objective_emb",
        }
        for key in fin.keys():
            if key in REWRITTEN_KEYS:
                continue
            fin.copy(key, fout)

        in_strategy = fin["strategy_emb"]
        in_risk     = fin["risk_emb"]
        in_delta_t  = fin["delta_t"]
        zero_emb_row = np.zeros((N_Q, EMB), dtype=np.float32)  # constants reused per row

        # Resolve --max-lag once: any non-positive value collapses to N_Q
        # (effectively unbounded inside the snapshot window).
        effective_max_lag = resolve_max_lag(args.max_lag, N_Q)

        for r in range(N_FUNDS):
            fid_raw = fund_ids_arr[r]
            fid = int(fid_raw) if not isinstance(fid_raw, bytes) \
                  else int(fid_raw.decode())
            portnos = id2portnos.get(fid, [])
            explicit_raw = merge_lookups(portnos, by_portno) if portnos else {}
            # Drop lookup entries whose report_q lies outside the H5's quarter
            # range (e.g., eff_q=2005Q3 with report_q=2005Q2 — H5 starts at 2005Q3).
            explicit = {k: v for k, v in explicit_raw.items() if v in quarter_to_idx}
            n_lookup_dropped_oob += len(explicit_raw) - len(explicit)

            target_q_carry = build_target_q_vector(explicit, quarters, quarter_to_idx)
            # Apply the chosen anchor policy:
            #   strict: cells without an explicit (eff_q, report_q) lookup are
            #           forced to cold-start (-1).
            #   carry:  non-explicit cells keep the carry-forward target_q from
            #           the most-recent explicit anchor.
            explicit_mask = np.array([q in explicit for q in quarters], dtype=bool)
            target_q = apply_anchor_policy(target_q_carry, explicit_mask, args.anchor_policy)

            # Pull just this fund's row from the H5 (no full-tensor load).
            strat_row = in_strategy[r]
            risk_row  = in_risk[r]
            dt_row_old = in_delta_t[r].astype(np.int64)

            filings = reconstruct_filings(strat_row, risk_row, dt_row_old)
            s, _rk, dt, ht = re_locf_anchored(filings, target_q, N_Q, EMB, effective_max_lag)

            # Strategy-only contract: risk channel intentionally left at zero
            # (loader-compatible placeholder).
            d_strategy[r]  = s
            d_risk[r]      = zero_emb_row
            d_delta_t[r]   = dt
            d_has_text[r]  = ht

            # Per-cell coverage data — one row per (fund, snapshot quarter).
            # dt[qi] == -1  → cold-start; dt[qi] == 0 → fresh; dt[qi] >= 1 → forward-filled.
            for qi in range(N_Q):
                dti = int(dt[qi])
                if dti == -1:
                    status = "cold_start"
                    picked_src = ""
                elif dti == 0:
                    status = "fresh"
                    picked_src = qi
                else:
                    status = "forward_filled"
                    picked_src = qi - dti
                coverage_rows.append({
                    "id_idx": fid,
                    "eff_q": quarters[qi],
                    "status": status,
                    "delta_t": dti,
                    "target_q_idx": int(target_q[qi]) if int(target_q[qi]) >= 0 else "",
                    "picked_src_q_idx": picked_src,
                    "policy": args.anchor_policy,
                    "max_lag_effective": effective_max_lag,
                })

            # Recompute aux datasets row-by-row from the re-anchored strategy.
            # abs_emb == strategy_emb matches ProspectusEmbeddingLoader at
            # risk_weight=0.0 (the user default per the riskweight0 ablation memory).
            d_abs[r] = s
            delta_emb_row = np.zeros((N_Q, EMB), dtype=np.float32)
            delta_emb_row[1:] = s[1:] - s[:-1]
            d_delta_emb[r] = delta_emb_row
            d_objective[r] = zero_emb_row
            # cosine_sim across adjacent quarters within this fund row.
            cos_row = np.zeros(N_Q, dtype=np.float32)
            curr = s[1:]
            prev = s[:-1]
            nc = np.linalg.norm(curr, axis=-1)
            np_ = np.linalg.norm(prev, axis=-1)
            denom = nc * np_
            mask = denom > 0
            if mask.any():
                dots = (curr * prev).sum(axis=-1)
                cos_slice = np.zeros_like(nc)
                cos_slice[mask] = (dots[mask] / denom[mask]).astype(np.float32)
                cos_row[1:] = cos_slice
            d_cosine[r] = cos_row

            was_active = (dt_row_old != -1)
            now_active = (dt != -1)
            audit_rows.append({
                "id_idx": fid,
                "n_portnos": len(portnos),
                "n_lookup_entries": len(explicit),
                "n_filings_reconstructed": len(filings),
                "n_cells_old_active": int(was_active.sum()),
                "n_cells_new_active": int(now_active.sum()),
                "n_cells_lost":   int((was_active & ~now_active).sum()),
                "n_cells_gained": int((~was_active & now_active).sum()),
                "n_cells_dt_changed": int(((dt != dt_row_old) & now_active).sum()),
                "mean_dt_old": float(dt_row_old[was_active].mean()) if was_active.any() else float('nan'),
                "mean_dt_new": float(dt[now_active].mean()) if now_active.any() else float('nan'),
            })

            if (r + 1) % 500 == 0:
                print(f"  ...processed {r+1:,}/{N_FUNDS:,} funds")

        fout.attrs["anchor"] = "report_dt"
        fout.attrs["sections"] = "strategy"
        fout.attrs["fallback"] = "carry_forward" if args.anchor_policy == "carry" else "cold_start"
        fout.attrs["source_h5"] = str(in_h5)
        fout.attrs["max_lag"] = effective_max_lag   # resolved value, not raw arg
        fout.attrs["anchor_policy"] = args.anchor_policy

    os.replace(tmp_h5, out_h5)
    print(f"  wrote {out_h5}")

    # Legacy per-fund audit (delta-vs-old) preserved at a sibling path to
    # avoid collision with the new per-cell coverage CSV that uses args.audit.
    audit_stem = Path(args.audit).stem
    if audit_stem.endswith("_audit"):
        legacy_stem = audit_stem[:-len("_audit")] + "_legacy_per_fund_audit"
    else:
        legacy_stem = audit_stem + "_legacy_per_fund_audit"
    legacy_audit_path = Path(args.audit).with_name(legacy_stem + ".csv")
    audit_df = pd.DataFrame(audit_rows)
    legacy_audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(legacy_audit_path, index=False)
    print(f"  wrote {legacy_audit_path} ({len(audit_df):,} rows; legacy per-fund delta-vs-old)")
    print()
    print("=== Summary ===")
    print(f"  total funds              : {len(audit_df):,}")
    print(f"  cells active old         : {audit_df.n_cells_old_active.sum():,}")
    print(f"  cells active new         : {audit_df.n_cells_new_active.sum():,}")
    print(f"  cells lost (drop-out)    : {audit_df.n_cells_lost.sum():,}")
    print(f"  cells gained             : {audit_df.n_cells_gained.sum():,}")
    print(f"  cells delta_t changed    : {audit_df.n_cells_dt_changed.sum():,}")
    print(f"  lookup entries dropped (out-of-range report_q): {n_lookup_dropped_oob:,}")
    valid = audit_df.mean_dt_old.notna() & audit_df.mean_dt_new.notna()
    if valid.any():
        print(f"  mean delta_t (old anchor, active cells) : {audit_df.loc[valid, 'mean_dt_old'].mean():.2f}")
        print(f"  mean delta_t (new anchor, active cells) : {audit_df.loc[valid, 'mean_dt_new'].mean():.2f}")

    write_coverage_report(coverage_rows, args, N_FUNDS, N_Q)


def derive_default_paths(base_out: str, policy: str, max_lag_arg: int):
    """Derive (out_h5, audit_csv, coverage_json) paths from a base output H5,
    suffixing by policy and max_lag so artifacts never collide.

    Naming rules:
      strict + max_lag=6   -> base_out unchanged (matches existing baseline)
      strict + max_lag<=0  -> base + '_lag_unbounded'
      carry  + max_lag<=0  -> base + '_carry'         (new default)
      carry  + max_lag=6   -> base + '_carry_lag6'    (matches SimTeG sibling)
      strict + other N     -> base + f'_lag{N}'
      carry  + other N     -> base + f'_carry_lag{N}'
    """
    base_path = Path(base_out)
    stem = base_path.stem                       # e.g. 'prospectus_embeddings_report_anchored'
    parent = base_path.parent

    if policy == "strict" and max_lag_arg == 6:
        out_suffix = ""
    elif policy == "strict" and max_lag_arg <= 0:
        out_suffix = "_lag_unbounded"
    elif policy == "strict":
        out_suffix = f"_lag{max_lag_arg}"
    elif policy == "carry" and max_lag_arg <= 0:
        out_suffix = "_carry"
    elif policy == "carry" and max_lag_arg == 6:
        out_suffix = "_carry_lag6"
    else:  # carry + explicit positive non-6
        out_suffix = f"_carry_lag{max_lag_arg}"

    out_h5 = parent / f"{stem}{out_suffix}.h5"

    # Audit CSV mirrors the H5's suffix; replace 'embeddings_report_anchored'
    # -> 'report_anchor' to match the existing audit filename convention.
    audit_stem = stem.replace("prospectus_embeddings_report_anchored",
                              "prospectus_report_anchor")
    audit_csv = parent / f"{audit_stem}{out_suffix}_audit.csv"
    coverage_json = parent / f"{audit_stem}{out_suffix}_coverage.json"

    return out_h5, audit_csv, coverage_json


def write_coverage_report(coverage_rows, args, n_funds: int, n_q: int) -> dict:
    """Write per-cell coverage CSV and per-summary JSON. Also print a summary
    block to stdout. Returns the summary dict (also useful for tests)."""
    import json
    cov_df = pd.DataFrame(coverage_rows)
    audit_path = Path(args.audit)
    coverage_path = Path(args.coverage_json)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    cov_df.to_csv(audit_path, index=False)

    total = n_funds * n_q
    n_fresh = int((cov_df.status == "fresh").sum())
    n_ff = int((cov_df.status == "forward_filled").sum())
    n_cold = int((cov_df.status == "cold_start").sum())

    ff_df = cov_df.loc[cov_df.status == "forward_filled"]
    dts = ff_df["delta_t"].astype(int).values
    bucket_1_3   = int(((ff_df.delta_t >= 1) & (ff_df.delta_t <= 3)).sum())
    bucket_4_7   = int(((ff_df.delta_t >= 4) & (ff_df.delta_t <= 7)).sum())
    bucket_8_15  = int(((ff_df.delta_t >= 8) & (ff_df.delta_t <= 15)).sum())
    bucket_16p   = int((ff_df.delta_t >= 16).sum())

    funds_with_cold = int(cov_df.loc[cov_df.status == "cold_start"]
                              .groupby("id_idx").ngroups)
    # "all-cold" funds = funds with N_Q cold-start cells
    cold_per_fund = (cov_df.loc[cov_df.status == "cold_start"]
                          .groupby("id_idx").size())
    funds_all_cold = int((cold_per_fund == n_q).sum())
    funds_fully_covered = n_funds - funds_with_cold

    def pct(x): return 100.0 * x / total if total else 0.0

    summary = {
        "policy": args.anchor_policy,
        "max_lag_effective": int(coverage_rows[0]["max_lag_effective"]) if coverage_rows else None,
        "n_funds": n_funds,
        "n_quarters": n_q,
        "n_fund_quarters": total,
        "n_fresh": n_fresh, "pct_fresh": pct(n_fresh),
        "n_forward_filled": n_ff, "pct_forward_filled": pct(n_ff),
        "ff_bucket_1_3": bucket_1_3,
        "ff_bucket_4_7": bucket_4_7,
        "ff_bucket_8_15": bucket_8_15,
        "ff_bucket_16_plus": bucket_16p,
        "n_cold_start": n_cold, "pct_cold_start": pct(n_cold),
        "delta_t_mean": float(dts.mean()) if dts.size else None,
        "delta_t_median": float(np.median(dts)) if dts.size else None,
        "delta_t_p90": float(np.percentile(dts, 90)) if dts.size else None,
        "delta_t_max": int(dts.max()) if dts.size else None,
        "funds_fully_covered": funds_fully_covered,
        "funds_with_cold_cells": funds_with_cold,
        "funds_all_cold": funds_all_cold,
    }
    with open(coverage_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # Stdout summary block.
    print()
    print(f"Coverage report — {args.out_h5}")
    print(f"  policy={summary['policy']}, "
          f"max_lag={'unbounded' if args.max_lag <= 0 else args.max_lag} "
          f"(N_Q={n_q})")
    print(f"  funds: {n_funds:,}   quarters: {n_q}   "
          f"fund-quarters: {total:,}")
    print()
    print("  fund-quarter status:")
    print(f"    fresh (Δ=0):                 {n_fresh:>9,} ({pct(n_fresh):6.2f}%)")
    print(f"    forward-filled (Δ≥1):        {n_ff:>9,} ({pct(n_ff):6.2f}%)")
    print(f"       Δ=1–3   (≤1yr stale):     {bucket_1_3:>9,} ({pct(bucket_1_3):6.2f}%)")
    print(f"       Δ=4–7   (1–2yr stale):    {bucket_4_7:>9,} ({pct(bucket_4_7):6.2f}%)")
    print(f"       Δ=8–15  (2–4yr stale):    {bucket_8_15:>9,} ({pct(bucket_8_15):6.2f}%)")
    print(f"       Δ≥16    (≥4yr stale):     {bucket_16p:>9,} ({pct(bucket_16p):6.2f}%)")
    print(f"    cold-start (no filing):      {n_cold:>9,} ({pct(n_cold):6.2f}%)")
    print()
    if dts.size:
        print(f"  staleness Δ (forward-filled cells only): "
              f"mean={summary['delta_t_mean']:.2f}  "
              f"median={summary['delta_t_median']:.1f}  "
              f"p90={summary['delta_t_p90']:.1f}  "
              f"max={summary['delta_t_max']}")
    print()
    print("  per-fund breakdown:")
    print(f"    funds fully covered:          {funds_fully_covered:>6,} "
          f"({100*funds_fully_covered/n_funds:5.2f}%)")
    print(f"    funds with ≥1 cold cell:      {funds_with_cold:>6,} "
          f"({100*funds_with_cold/n_funds:5.2f}%)")
    print(f"    funds with 0 in-force filings:{funds_all_cold:>6,} "
          f"({100*funds_all_cold/n_funds:5.2f}%)")
    print(f"  wrote {audit_path} ({len(cov_df):,} rows)")
    print(f"  wrote {coverage_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-h5", default=DEFAULT_IN_H5)
    ap.add_argument("--out-h5", default=None,
                    help="Output H5 path. If omitted, derived from --anchor-policy "
                         "and --max-lag (see derive_default_paths).")
    ap.add_argument("--lookup", default=DEFAULT_LOOKUP)
    ap.add_argument("--mapping", default=DEFAULT_MAP)
    ap.add_argument("--audit", default=None,
                    help="Audit CSV path. If omitted, derived from --out-h5.")
    ap.add_argument("--coverage-json", default=None,
                    help="Coverage summary JSON path. If omitted, derived from --out-h5.")
    ap.add_argument("--anchor-policy", choices=("strict", "carry"), default="carry",
                    help="strict: cells without explicit (eff_q, report_q) lookup "
                         "are cold-start (matches existing baseline H5). "
                         "carry: forward-fill the most recent explicit anchor "
                         "(default — strategy signal stays fresh-ish until superseded).")
    ap.add_argument("--max-lag", type=int, default=0,
                    help="Cap on (target_q - src) in re_locf_anchored. "
                         "0 (default) or any non-positive value = unbounded. "
                         "Pass 6 to mirror the original encode_prospectus_openai.py "
                         "convention (used by the existing strict baseline H5).")
    args = ap.parse_args()

    # If user didn't override --out-h5, derive from policy + max_lag.
    if args.out_h5 is None:
        out_h5, audit_csv, coverage_json = derive_default_paths(
            DEFAULT_OUT_H5, args.anchor_policy, args.max_lag)
        args.out_h5 = str(out_h5)
        if args.audit is None:
            args.audit = str(audit_csv)
        if args.coverage_json is None:
            args.coverage_json = str(coverage_json)
    else:
        # User passed --out-h5 explicitly. Derive audit/coverage from it
        # unless those were also passed.
        out_path = Path(args.out_h5)
        if args.audit is None:
            args.audit = str(out_path.with_name(
                out_path.stem.replace("prospectus_embeddings_report_anchored",
                                       "prospectus_report_anchor")
                + "_audit.csv"))
        if args.coverage_json is None:
            args.coverage_json = str(out_path.with_name(
                out_path.stem.replace("prospectus_embeddings_report_anchored",
                                       "prospectus_report_anchor")
                + "_coverage.json"))

    run(args)


if __name__ == "__main__":
    main()
