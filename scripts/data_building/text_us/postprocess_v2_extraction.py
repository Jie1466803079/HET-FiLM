#!/usr/bin/env python3
"""Post-process v2 extraction output to remove residual bug-class bodies.

After v2 finishes the full extraction, a small fraction (~0.8% of bodies
in 2006-2010 timestamps) start with prose-continuation fragments like:
  - "and risks of each Fund. Information about..."
  - "and risks 73  - ---..." (TOC remnants)
  - "Respecting the Micro Cap Fund 4..." (multi-line TOC continuation)

These all come from heading patterns matched in mid-prose contexts that
v2's TOC/heading checks didn't catch. They are easy to detect by:
  (a) Body starts with "and risks" / "and Risks" / numeric / "Respecting"
  (b) Body is short (<150 chars typically — heading match terminated
      quickly at the next real heading) OR is at the 15k cap (overrun)

Policy: remove the `strategy` field from those entries. Other sections
(objective, risk) are left intact. The entry record is preserved so the
downstream loader sees a missing-strategy row instead of a missing
fund-quarter row.

Outputs:
  - fund_485bpos_sections_temporal_v2.json (modified in-place after backup)
  - fund_485bpos_sections_temporal_v2.json.bak (timestamped backup)
  - postprocess_v2_report.json (audit log of removals)

Usage:
  python postprocess_v2_extraction.py
"""

import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict

PATH = ("/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/sec_filings_project"
        "/extracted/fund_485bpos_sections_temporal_v2.json")
BACKUP = PATH + ".bak"
REPORT = PATH.replace(".json", "_postprocess_report.json")

# Bug-class body detection rules (case-insensitive, applied to body start)
_AND_RISKS = re.compile(r"^\s*(?:and|or)\s+(?:risks?|policies|performance)\b", re.I)
_PAGE_NUM_START = re.compile(r"^\s*\d{1,3}\s")
_RESPECTING_START = re.compile(r"^\s*Respecting\s+the\s+\w", re.I)
# "and risks <intermediate words> <page-num>" anywhere in first 150 chars
_TOC_REMNANT = re.compile(
    r"^\s*(?:and|or)\s+(?:risks?|policies)\s+(?:[\w][\w'-]*\s+){0,8}\d{1,3}\s",
    re.I,
)


def is_bug_body(body: str) -> str:
    """Return bug-class name if body matches a known bug pattern, else ''."""
    if not body:
        return ""
    head = body[:200]
    if _AND_RISKS.match(head):
        return "and_risks_continuation"
    if _RESPECTING_START.match(head):
        return "respecting_toc"
    if _TOC_REMNANT.match(head):
        return "toc_remnant_with_page"
    if _PAGE_NUM_START.match(head):
        return "page_number_start"
    return ""


def main():
    if not os.path.exists(PATH):
        print(f"ERROR: extraction JSON not found at {PATH}")
        sys.exit(1)

    print(f"Loading {PATH}...")
    with open(PATH) as f:
        rows = json.load(f)
    print(f"  {len(rows):,} entries loaded")

    # Backup
    print(f"Backing up to {BACKUP}...")
    shutil.copy2(PATH, BACKUP)
    sz_mb = os.path.getsize(BACKUP) / (1024 * 1024)
    print(f"  Backup size: {sz_mb:.1f} MB")

    # Detect and remove
    removals_by_class = defaultdict(int)
    removals_by_year = defaultdict(int)
    audit = []
    for r in rows:
        strat = r.get("strategy")
        if not strat:
            continue
        bug = is_bug_body(strat)
        if not bug:
            continue
        # Empirical: ALL surviving "and risks" body starts are SAI/TOC junk.
        # Examples checked across tiers 500-15000 chars: every one is
        # cross-reference prose ("and risks of investing in each Fund are
        # described in the Prospectus...") or TOC ("AND RISKS 70 PORTFOLIO
        # HOLDINGS INFORMATION 91..."). Real strategy never starts with
        # "and risks". Remove regardless of length.
        audit.append({
            "id_idx": r.get("id_idx"),
            "timestamp": r.get("timestamp"),
            "series_id": r.get("series_id"),
            "filing_date": r.get("filing_date"),
            "bug_class": bug,
            "len": len(strat),
            "head_60": strat[:60].replace("\n", " "),
        })
        del r["strategy"]
        removals_by_class[bug] += 1
        removals_by_year[r.get("timestamp", "")[:4]] += 1

    total_removed = sum(removals_by_class.values())
    print(f"\nDetected {total_removed} bug-class bodies (strategy removed, entry preserved):")
    for cls, n in sorted(removals_by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls:<30} {n:>5}")
    print(f"\nBy year:")
    for y, n in sorted(removals_by_year.items()):
        print(f"  {y}: {n}")

    # Final stats
    with_strat = sum(1 for r in rows if r.get("strategy"))
    print(f"\nFinal: {with_strat:,}/{len(rows):,} entries with strategy "
          f"({with_strat/len(rows)*100:.1f}%)")

    # Write cleaned JSON
    print(f"\nWriting cleaned JSON to {PATH}...")
    with open(PATH, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    sz_mb_new = os.path.getsize(PATH) / (1024 * 1024)
    print(f"  Size: {sz_mb_new:.1f} MB (was {sz_mb:.1f} MB)")

    # Write audit report
    with open(REPORT, "w") as f:
        json.dump({
            "timestamp_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_entries": len(rows),
            "total_removed": total_removed,
            "removals_by_class": dict(removals_by_class),
            "removals_by_year": dict(removals_by_year),
            "audit_first_50": audit[:50],
        }, f, indent=2)
    print(f"Audit report: {REPORT}")


if __name__ == "__main__":
    main()
