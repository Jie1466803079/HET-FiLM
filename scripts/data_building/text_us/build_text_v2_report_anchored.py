"""Build a report-anchored v2 prospectus text JSON.

Re-LOCFs v2_clean text records against report_q for each fund. The output is a
drop-in replacement for the existing dataset.py lookup: same keys
((id_idx, timestamp=eff_q_end_date)), but the text content reflects the filing
available by report_q(id_idx, eff_q) within MAX_LAG=6.

See docs/plans/2026-05-27-simteg-v2-report-anchored-retrain-design.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _prospectus_report_anchor import MAX_LAG, build_target_q_vector  # noqa: E402


def quarter_idx_of_date(date_str: str, quarter_to_idx: Dict[str, int]) -> Optional[int]:
    """Convert YYYY-MM-DD to a 'YYYYQN' quarter and look up its index.

    Returns None if the quarter is outside the H5 snapshot range."""
    y, m, _ = date_str.split("-")
    qn = (int(m) - 1) // 3 + 1
    return quarter_to_idx.get(f"{y}Q{qn}")


def dedupe_filings_by_date(
    records: Iterable[dict],
    quarter_to_idx: Dict[str, int],
) -> Dict[int, dict]:
    """Group records by filing_date_quarter_idx; keep first-seen per filing.

    The v2 JSON's LOCF means the same filing's text appears in many cells; we
    only need one copy per distinct filing_date for the re-anchoring picker."""
    out: Dict[int, dict] = {}
    for r in records:
        fq_idx = quarter_idx_of_date(r["filing_date"], quarter_to_idx)
        if fq_idx is None:
            continue
        if fq_idx in out:
            continue
        out[fq_idx] = {
            "filing_date": r["filing_date"],
            "strategy": r.get("strategy") or "",
            "risk": r.get("risk") or "",
            "objective": r.get("objective") or "",
            "source": r.get("source", ""),
            "series_id": r.get("series_id", ""),
        }
    return out


def merge_explicit_lookups(
    portnos: List[int],
    by_portno: Dict[int, Dict[str, str]],
) -> Dict[str, str]:
    """Combine per-portno eff_q → report_q maps; on conflict, later report_q wins.

    Mirrors merge_lookups() in scripts/rebuild_prospectus_h5_report_anchored.py."""
    merged: Dict[str, str] = {}
    for p in portnos:
        d = by_portno.get(p)
        if not d:
            continue
        for eff_q, rpt_q in d.items():
            if eff_q not in merged or rpt_q > merged[eff_q]:
                merged[eff_q] = rpt_q
    return merged


def pick_filing(
    filing_q_idxs: Set[int],
    target_q: int,
    max_lag: int = MAX_LAG,
) -> Optional[int]:
    """Return largest filing_q ≤ target_q with target_q − filing_q ≤ max_lag.

    Returns None if no candidate fits."""
    best: Optional[int] = None
    for fq in filing_q_idxs:
        if fq > target_q:
            continue
        if target_q - fq > max_lag:
            continue
        if best is None or fq > best:
            best = fq
    return best


def parse_portno_list(s: str) -> List[int]:
    return [int(x) for x in str(s).split("|")]


def load_lookup_by_portno(path: Path) -> Dict[int, Dict[str, str]]:
    df = pd.read_csv(path)
    out: Dict[int, Dict[str, str]] = {}
    for portno, eff_q, rpt_q in zip(df["crsp_portno"].astype(int),
                                    df["eff_q"], df["rpt_q"]):
        out.setdefault(int(portno), {})[str(eff_q)] = str(rpt_q)
    return out


def main():
    # Absolute paths so the script works regardless of cwd / $PBS_O_WORKDIR.
    # Data lives at OUTER /srv/scratch/dbgcse/jieliu/mutual_fund_prediction/;
    # code lives at INNER .../mutual_fund_prediction/mutual_fund_prediction/.
    _ROOT = "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction"
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", default=(
        f"{_ROOT}/sec_filings_project/extracted/"
        "fund_485bpos_sections_temporal_v2_clean.json"))
    ap.add_argument("--out_json", default=(
        f"{_ROOT}/sec_filings_project/extracted/"
        "fund_485bpos_sections_temporal_v2_clean_report_anchored.json"))
    ap.add_argument("--audit_csv", default=(
        f"{_ROOT}/sec_filings_project/extracted/"
        "fund_485bpos_v2_report_anchor_audit.csv"))
    ap.add_argument("--lookup", default=(
        f"{_ROOT}/mutual_fund_prediction/analysis/fund_effq_to_reportq.csv"))
    ap.add_argument("--mapping", default=(
        f"{_ROOT}/sec_filings_project/fund_mapping_since2005Q3.csv"))
    ap.add_argument("--schema_h5", default=(
        f"{_ROOT}/sec_filings_project/embeddings_v2_openai/prospectus_embeddings.h5"))
    ap.add_argument("--anchor_policy", choices=("strict", "carry"), default="strict",
                    help="strict: cells without an explicit (eff_q, report_q) "
                         "lookup are cold-start (matches OpenAI v2ra). "
                         "carry: build_target_q_vector's carry-forward fallback "
                         "fills those cells with the most-recent-known target_q.")
    args = ap.parse_args()
    print(f"[build] anchor_policy={args.anchor_policy}", flush=True)

    # --- Schema (snapshot_quarters defines the 65-quarter universe) ---
    with h5py.File(args.schema_h5, "r") as f:
        quarters_raw = f["snapshot_quarters"][:]
    quarters = [q.decode() if isinstance(q, bytes) else str(q)
                for q in quarters_raw]
    quarter_to_idx = {q: i for i, q in enumerate(quarters)}
    n_q = len(quarters)
    month_end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    quarter_to_date = {}
    for q in quarters:
        yr, qn = q.split("Q")
        quarter_to_date[q] = f"{yr}-{month_end[int(qn)]}"

    # --- Lookups ---
    by_portno = load_lookup_by_portno(Path(args.lookup))
    mapping = pd.read_csv(args.mapping)
    id2portnos = {int(r.id_idx): parse_portno_list(r.crsp_portno_list)
                  for r in mapping.itertuples(index=False)}
    print(f"[build] lookup: {sum(len(v) for v in by_portno.values()):,} (portno, eff_q) pairs across {len(by_portno):,} portnos", flush=True)
    print(f"[build] mapping: {len(id2portnos):,} id_idx → portnos", flush=True)

    # --- Group input records by id_idx ---
    print(f"[build] loading {args.in_json}", flush=True)
    with open(args.in_json) as f:
        in_records = json.load(f)
    print(f"[build] {len(in_records):,} input records", flush=True)

    by_fund: Dict[int, List[dict]] = defaultdict(list)
    for r in in_records:
        try:
            f_id = int(r["id_idx"])
        except (KeyError, ValueError, TypeError):
            continue
        by_fund[f_id].append(r)
    print(f"[build] {len(by_fund):,} distinct funds", flush=True)
    del in_records  # free ~370 MB

    # --- Per-fund re-anchor ---
    out_records: List[dict] = []
    audit_rows: List[dict] = []
    n_realtext = 0
    n_coldstart = 0
    funds_no_portnos = 0
    funds_no_filings = 0
    n_dropped_oob = 0

    for f_id, frecs in by_fund.items():
        portnos = id2portnos.get(f_id, [])
        if not portnos:
            funds_no_portnos += 1
        explicit_raw = merge_explicit_lookups(portnos, by_portno)
        # Filter out report_q values outside the H5 snapshot range (e.g. 2005Q2
        # when the universe starts at 2005Q3). Mirrors the same filter applied
        # by rebuild_prospectus_h5_report_anchored.py before calling
        # build_target_q_vector.
        explicit = {k: v for k, v in explicit_raw.items() if v in quarter_to_idx}
        n_dropped_oob += len(explicit_raw) - len(explicit)
        filings = dedupe_filings_by_date(frecs, quarter_to_idx)
        if not filings:
            funds_no_filings += 1
            for eff_qi in range(n_q):
                audit_rows.append({
                    "id_idx": f_id, "eff_q": quarters[eff_qi],
                    "target_q": "", "picked_filing_q": "",
                    "picked_filing_date": "", "has_match": 0,
                })
                n_coldstart += 1
            continue

        target_q_carry = build_target_q_vector(explicit, quarters, quarter_to_idx)
        if args.anchor_policy == "strict":
            # Match rebuild_prospectus_h5_report_anchored.py:157-158: cells
            # without an explicit (eff_q, report_q) lookup are cold-start
            # (mark with -1 to signal pick_filing must produce no match).
            target_q = np.array(
                [target_q_carry[i] if q in explicit else -1
                 for i, q in enumerate(quarters)],
                dtype=np.int64,
            )
        else:  # carry — use build_target_q_vector's carry-forward fallback as-is
            target_q = target_q_carry

        for eff_qi in range(n_q):
            tq = int(target_q[eff_qi])
            if tq < 0:
                audit_rows.append({
                    "id_idx": f_id, "eff_q": quarters[eff_qi],
                    "target_q": "", "picked_filing_q": "",
                    "picked_filing_date": "", "has_match": 0,
                })
                n_coldstart += 1
                continue
            picked = pick_filing(set(filings.keys()), tq, MAX_LAG)
            if picked is None:
                audit_rows.append({
                    "id_idx": f_id, "eff_q": quarters[eff_qi],
                    "target_q": quarters[tq], "picked_filing_q": "",
                    "picked_filing_date": "", "has_match": 0,
                })
                n_coldstart += 1
                continue
            fil = filings[picked]
            out_records.append({
                "id_idx": str(f_id),
                "timestamp": quarter_to_date[quarters[eff_qi]],
                "series_id": fil["series_id"],
                "source": fil["source"],
                "filing_date": fil["filing_date"],
                "objective": fil["objective"],
                "strategy": fil["strategy"],
                "risk": fil["risk"],
            })
            audit_rows.append({
                "id_idx": f_id, "eff_q": quarters[eff_qi],
                "target_q": quarters[tq], "picked_filing_q": quarters[picked],
                "picked_filing_date": fil["filing_date"], "has_match": 1,
            })
            n_realtext += 1

    print(f"[build] real-text cells:  {n_realtext:,}", flush=True)
    print(f"[build] cold-start cells: {n_coldstart:,}", flush=True)
    print(f"[build] funds w/o portnos in mapping: {funds_no_portnos}", flush=True)
    print(f"[build] funds w/o any filings:        {funds_no_filings}", flush=True)
    print(f"[build] lookup entries dropped (report_q out of range): {n_dropped_oob:,}", flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_records, f)
    print(f"[build] wrote {args.out_json} ({len(out_records):,} records)", flush=True)

    pd.DataFrame(audit_rows).to_csv(args.audit_csv, index=False)
    print(f"[build] wrote {args.audit_csv} ({len(audit_rows):,} rows)", flush=True)


if __name__ == "__main__":
    main()
