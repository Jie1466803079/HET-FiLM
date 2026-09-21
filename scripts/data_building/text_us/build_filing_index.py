#!/usr/bin/env python3
"""Build a (series_id) -> [filings] index from raw 485BPOS submissions.

Scans the sec-edgar-filings tree for `full-submission.txt` files, parses
their SGML header to pull FILED-AS-OF-DATE and <SERIES-ID> tags, and emits
a JSON index used by the LLM extractor (`extract_485bpos_llm.py`).

NEW FILE — does not modify the production extractor `extract_485bpos_sections.py`.

Output: <SCRIPT_DIR>/extracted/filing_index.json
    {
      series_id: [
        {"filing_date": "YYYY-MM-DD", "filepath": "...", "cik": "0000..."},
        ...
      ]
    }
Filings within each series list are sorted by filing_date ascending.
A `__cik_only__` key holds entries for filings that had no <SERIES-ID> tags
(pre-2006 filings), grouped by CIK for fallback lookups.
"""
import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Two filing roots: sec-edgar-filings (newer, 2010+) and all_485bpos (older,
# 2004-2009). The user nominated sec-edgar-filings as the most comprehensive,
# but it lacks pre-2010 filings — those live in all_485bpos. To match the
# 2005-2008 era of failed strategy extractions we must scan both.
DEFAULT_ROOTS = [
    os.path.join(SCRIPT_DIR, "sec-edgar-filings"),
    os.path.join(SCRIPT_DIR, "all_485bpos"),
]
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "extracted", "filing_index.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("build_filing_index")


_SERIES_ID_RE = re.compile(r"<SERIES-ID>\s*([^<\s]+)", re.IGNORECASE)
_FILED_DATE_RE = re.compile(r"FILED AS OF DATE:\s*(\d{8})")


def parse_header(fpath: str) -> Tuple[Optional[str], List[str]]:
    """Return (filing_date_iso, [series_ids]) parsed from the SGML header.

    Reads only until </SEC-HEADER> to keep IO cheap. filing_date returned
    as 'YYYY-MM-DD' or None if not found.
    """
    filing_date: Optional[str] = None
    series_ids: List[str] = []
    try:
        with open(fpath, "r", errors="replace") as f:
            for line in f:
                if filing_date is None:
                    m = _FILED_DATE_RE.search(line)
                    if m:
                        ds = m.group(1)
                        filing_date = f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
                m = _SERIES_ID_RE.search(line)
                if m:
                    sid = m.group(1).strip()
                    if sid and sid not in series_ids:
                        series_ids.append(sid)
                if "</SEC-HEADER>" in line:
                    break
    except Exception as e:
        log.warning(f"header parse failed: {fpath}: {e}")
        return None, []
    return filing_date, series_ids


def _fallback_date_from_acc(acc_dir: str) -> Optional[str]:
    """Recover an approximate filing_date from an accession number like
    `0000891804-06-001565` (YY in middle). Returns 'YYYY-01-01' as a coarse
    fallback for filings missing FILED AS OF DATE in the header.
    """
    parts = acc_dir.split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        yy = int(parts[1])
        yr = yy + 2000 if yy < 50 else yy + 1900
        return f"{yr}-01-01"
    return None


def build_index(roots: List[str]) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]], dict]:
    """Walk one or more filing trees and build series_id -> [filings] and
    cik -> [filings]. Dedupes by (CIK, accession): if the same accession
    appears under multiple roots, the FIRST root listed wins. List roots in
    preference order (e.g. sec-edgar-filings first, all_485bpos second).

    Returns: (series_to_filings, cik_to_filings, stats)
    """
    series_to_filings: Dict[str, List[dict]] = defaultdict(list)
    cik_to_filings: Dict[str, List[dict]] = defaultdict(list)
    seen_acc: Dict[Tuple[str, str], str] = {}  # (cik, accession) -> root
    n_filings = 0
    n_filings_with_series = 0
    n_filings_no_header_date = 0
    n_dups = 0

    if isinstance(roots, str):
        roots = [roots]

    for root in roots:
        if not os.path.isdir(root):
            log.warning(f"filing root not found: {root}; skipping")
            continue
        log.info(f"Scanning root: {root}")
        cik_dirs = sorted(d for d in os.listdir(root) if not d.startswith("."))
        for i, cik in enumerate(cik_dirs):
            bpos_dir = os.path.join(root, cik, "485BPOS")
            if not os.path.isdir(bpos_dir):
                continue
            for acc in sorted(os.listdir(bpos_dir)):
                fpath = os.path.join(bpos_dir, acc, "full-submission.txt")
                if not os.path.isfile(fpath):
                    continue
                key = (cik, acc)
                if key in seen_acc:
                    n_dups += 1
                    continue
                seen_acc[key] = root
                n_filings += 1
                filing_date, series_ids = parse_header(fpath)
                if filing_date is None:
                    filing_date = _fallback_date_from_acc(acc)
                    n_filings_no_header_date += 1
                entry = {
                    "filing_date": filing_date,
                    "filepath": fpath,
                    "cik": cik,
                    "accession": acc,
                }
                if series_ids:
                    n_filings_with_series += 1
                    for sid in series_ids:
                        series_to_filings[sid].append(entry)
                cik_to_filings[cik].append(entry)
            if (i + 1) % 100 == 0:
                log.info(f"  [{root}] processed {i + 1}/{len(cik_dirs)} CIKs, {n_filings} filings so far")

    # Sort by date asc within each list (None dates sink to bottom)
    def _key(e):
        return e.get("filing_date") or "9999-99-99"
    for sid in series_to_filings:
        series_to_filings[sid].sort(key=_key)
    for cik in cik_to_filings:
        cik_to_filings[cik].sort(key=_key)

    stats = {
        "n_filings_total": n_filings,
        "n_filings_with_series": n_filings_with_series,
        "n_filings_no_header_date": n_filings_no_header_date,
        "n_dups_skipped": n_dups,
        "n_series": len(series_to_filings),
        "n_ciks": sum(1 for c in cik_to_filings if cik_to_filings[c]),
        "roots": roots,
    }
    return series_to_filings, cik_to_filings, stats


def build_series_to_ciks(
    series_csv: str,
    fund_map_csv: str,
    cik_map_csv: str,
) -> Dict[str, List[str]]:
    """Compose series_id -> [CIK,...] using the same three-CSV chain the
    legacy extractor uses (series→fund_group→crsp_fundno→cik). Read-only.
    """
    try:
        import pandas as pd
    except Exception:
        log.warning("pandas unavailable; skipping series->cik mapping")
        return {}
    if not (os.path.isfile(series_csv) and os.path.isfile(fund_map_csv) and os.path.isfile(cik_map_csv)):
        log.warning("series/fund/cik CSVs missing; skipping series->cik mapping")
        return {}
    series_map = pd.read_csv(series_csv)
    fm = pd.read_csv(fund_map_csv)
    cik_map = pd.read_csv(cik_map_csv)
    cik_map["crsp_fundno"] = cik_map["crsp_fundno"].astype(int)
    cik_map["cik"] = cik_map["cik"].astype(str).str.zfill(10)
    fundno_to_cik = cik_map.set_index("crsp_fundno")["cik"].to_dict()
    fg_to_fundnos = (
        fm.groupby("fund_group")["crsp_fundno"]
        .apply(lambda x: x.dropna().astype(int).tolist())
        .to_dict()
    )
    series_to_ciks: Dict[str, List[str]] = {}
    for sid in series_map["series_cik"].unique():
        fg = series_map.loc[series_map["series_cik"] == sid, "fund_group"].iloc[0]
        fundnos = fg_to_fundnos.get(fg, [])
        ciks = sorted({fundno_to_cik[fn] for fn in fundnos if fn in fundno_to_cik})
        if ciks:
            series_to_ciks[sid] = ciks
    return series_to_ciks


def validate_match_rate(
    series_to_filings: Dict[str, List[dict]],
    cik_to_filings: Dict[str, List[dict]],
    series_to_ciks: Dict[str, List[str]],
    temporal_json: str,
    strategy_min_chars: int = 50,
) -> None:
    """Check what fraction of empty/short-strategy entries are matchable.

    Two-tier match:
      * Series match: a filing in our index whose <SERIES-ID> tags
        include this series and filing_date <= entry.filing_date.
      * CIK fallback: any filing under any CIK linked to this series
        with filing_date <= entry.filing_date — used because pre-2010
        filings predate the SERIES-ID tag.
    """
    if not os.path.isfile(temporal_json):
        log.warning(f"temporal JSON not found: {temporal_json}; skipping validation")
        return
    with open(temporal_json) as f:
        entries = json.load(f)

    failed = [
        e for e in entries
        if not e.get("strategy") or len(e.get("strategy", "")) < strategy_min_chars
    ]
    n_failed = len(failed)

    matched_by_series = 0
    matched_any = 0
    matched_by_cik = 0
    for e in failed:
        sid = e.get("series_id")
        fdate_e = e.get("filing_date") or e.get("timestamp")
        hit = False
        cands = series_to_filings.get(sid, [])
        if cands:
            usable = [c for c in cands if (not fdate_e) or (c.get("filing_date") and c["filing_date"] <= fdate_e)]
            if usable:
                matched_by_series += 1
                matched_any += 1
                hit = True
        if not hit:
            ciks = series_to_ciks.get(sid, [])
            usable_cik = []
            for cik in ciks:
                for c in cik_to_filings.get(cik, []):
                    if (not fdate_e) or (c.get("filing_date") and c["filing_date"] <= fdate_e):
                        usable_cik.append(c)
            if usable_cik:
                matched_by_cik += 1
                matched_any += 1
                hit = True
        if not hit and sid in series_to_filings:
            matched_any += 1

    log.info("=" * 60)
    log.info(f"Validation against {temporal_json}:")
    log.info(f"  Failed entries (empty/short strategy): {n_failed}")
    log.info(f"  Matched by series_id (date-ok):        {matched_by_series} ({100*matched_by_series/max(n_failed,1):.1f}%)")
    log.info(f"  Matched via CIK fallback (date-ok):    {matched_by_cik} ({100*matched_by_cik/max(n_failed,1):.1f}%)")
    log.info(f"  Total date-matched (series OR cik):    {matched_by_series + matched_by_cik} ({100*(matched_by_series + matched_by_cik)/max(n_failed,1):.1f}%)")
    log.info(f"  Series_id appears in index at all:     {matched_any} ({100*matched_any/max(n_failed,1):.1f}%)")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="filings root (repeatable; defaults to both sec-edgar-filings and all_485bpos)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="output JSON path")
    parser.add_argument(
        "--temporal-json",
        default=os.path.join(SCRIPT_DIR, "extracted", "fund_485bpos_sections_temporal.json"),
        help="path to existing extractor output for validation",
    )
    parser.add_argument(
        "--series-csv",
        default=os.path.join(SCRIPT_DIR, "series_id_to_fund_node_2005Q3.csv"),
        help="series_id -> fund_group mapping CSV",
    )
    parser.add_argument(
        "--fund-map-csv",
        default=os.path.join(SCRIPT_DIR, "fund_mapping_since2005Q3.csv"),
        help="fund_group -> crsp_fundno mapping CSV",
    )
    parser.add_argument(
        "--cik-csv",
        default=os.path.join(SCRIPT_DIR, "crsp_fundno_to_cik.csv"),
        help="crsp_fundno -> CIK CSV",
    )
    parser.add_argument("--min-strategy-chars", type=int, default=50)
    args = parser.parse_args()

    roots = args.root if args.root else DEFAULT_ROOTS
    log.info(f"Scanning roots: {roots}")
    series_to_filings, cik_to_filings, stats = build_index(roots)
    log.info(f"Stats: {json.dumps(stats, indent=2)}")

    if not series_to_filings:
        log.error("No filings indexed. Aborting.")
        sys.exit(2)

    log.info("Building series_id -> CIK mapping from CRSP CSVs ...")
    series_to_ciks = build_series_to_ciks(args.series_csv, args.fund_map_csv, args.cik_csv)
    log.info(f"  series_to_ciks entries: {len(series_to_ciks)}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "__stats__": stats,
        "__cik_only__": cik_to_filings,
        "series": series_to_filings,
        "series_to_ciks": series_to_ciks,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Wrote index to {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

    validate_match_rate(
        series_to_filings,
        cik_to_filings,
        series_to_ciks,
        args.temporal_json,
        strategy_min_chars=args.min_strategy_chars,
    )


if __name__ == "__main__":
    main()
