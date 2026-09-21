#!/usr/bin/env python3
"""One-shot reclean of fund_485bpos_sections_temporal.json.

Loads the existing extracted JSON, applies the (now-fixed) strip_tags()
to each (objective, strategy, risk) section, drops TOC-garbage sections,
and writes the cleaned content back in place. Saves a .bak2 backup first.

Run from the repo root:
    python sec_filings_project/clean_extracted_text.py
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

# Make extract_485bpos_sections importable
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_485bpos_sections import strip_tags  # noqa: E402

JSON_PATH = SCRIPT_DIR / "extracted" / "fund_485bpos_sections_temporal.json"
BACKUP_PATH = SCRIPT_DIR / "extracted" / "fund_485bpos_sections_temporal.json.bak2"
SECTIONS = ("objective", "strategy", "risk")

# TOC garbage detector: short section dominated by dot-leader patterns
_DOT_LEADER = re.compile(r"\.{5,}")
_WHITESPACE = re.compile(r"\s+")


def is_toc_garbage(text: str) -> bool:
    """Return True if text looks like a captured TOC fragment.

    Heuristic: any dot-leader run (\\.{5,}) signals TOC formatting. Real
    prospectus prose effectively never uses dot-leaders, so this is a
    high-precision marker regardless of section length.
    """
    if not text:
        return False
    n_leaders = len(_DOT_LEADER.findall(text))
    return n_leaders >= 1


def clean_section(text):
    if not text:
        return None
    cleaned = strip_tags(text)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if not cleaned:
        return None
    if is_toc_garbage(cleaned):
        return None
    return cleaned


def section_stats(entries, label):
    print(f"\n=== {label} ===")
    for sec in SECTIONS:
        n_nonempty = sum(1 for e in entries if e.get(sec))
        n_tags = sum(1 for e in entries if "<" in (e.get(sec) or ""))
        n_ent = sum(1 for e in entries if "&#" in (e.get(sec) or ""))
        n_amp = sum(1 for e in entries if "&" in (e.get(sec) or ""))
        n_dbl = sum(1 for e in entries if "  " in (e.get(sec) or ""))
        total_chars = sum(len(e.get(sec) or "") for e in entries)
        print(
            f"  {sec:10s}: {n_nonempty:6d} non-empty | "
            f"<: {n_tags:6d} | &#: {n_ent:6d} | &: {n_amp:6d} | "
            f"  : {n_dbl:6d} | total chars: {total_chars:>12,}"
        )


def print_samples(entries):
    print("\n=== Random samples (post-clean) ===")
    random.seed(42)
    eligible = [e for e in entries if all(e.get(s) for s in SECTIONS)]
    if not eligible:
        print("  (no entries with all 3 sections)")
        return
    for i, e in enumerate(random.sample(eligible, min(5, len(eligible)))):
        print(f"\n  Sample {i+1}: id_idx={e['id_idx']}  ts={e['timestamp']}")
        for sec in SECTIONS:
            t = e[sec]
            preview = t[:200] + ("..." if len(t) > 200 else "")
            print(f"    {sec} ({len(t)} chars): {preview!r}")


def main():
    if not JSON_PATH.exists():
        sys.exit(f"ERROR: {JSON_PATH} not found")

    rerun = "--rerun" in sys.argv
    if BACKUP_PATH.exists() and not rerun:
        sys.exit(
            f"ERROR: {BACKUP_PATH} already exists. Refusing to overwrite the "
            f"pristine backup. If you intentionally want to re-run, manually "
            f"delete or rename the .bak2 file first, OR pass --rerun to skip "
            f"the backup step entirely."
        )

    print(f"Loading {JSON_PATH} ...")
    with open(JSON_PATH) as f:
        entries = json.load(f)
    print(f"  {len(entries)} entries")

    section_stats(entries, "BEFORE")

    if not rerun:
        print(f"\nBacking up to {BACKUP_PATH} ...")
        shutil.copy2(JSON_PATH, BACKUP_PATH)
    else:
        print(f"\n--rerun: skipping backup (using existing {BACKUP_PATH})")

    print("Cleaning all sections ...")
    n_dropped = {s: 0 for s in SECTIONS}
    for e in entries:
        for sec in SECTIONS:
            before = e.get(sec)
            after = clean_section(before)
            if before and not after:
                n_dropped[sec] += 1
            e[sec] = after

    print("\nDropped sections (TOC garbage / empty after clean):")
    for sec in SECTIONS:
        print(f"  {sec}: {n_dropped[sec]}")

    section_stats(entries, "AFTER")

    print(f"\nWriting cleaned JSON back to {JSON_PATH} ...")
    # Use ensure_ascii=False so unicode chars (e.g. U+2019) stay as real chars,
    # not as \u escapes — keeps file inspectable and avoids re-introducing &#xxxx
    tmp_path = JSON_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)
    os.replace(tmp_path, JSON_PATH)

    print_samples(entries)
    print("\nDone. Now run: python sec_filings_project/verify_clean.py")


if __name__ == "__main__":
    main()
