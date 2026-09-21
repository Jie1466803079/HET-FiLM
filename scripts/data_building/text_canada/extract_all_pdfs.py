"""Comprehensive re-extraction of fund-strategy records over ALL downloaded
PDFs in sedar_prospectuses/. Same routine as add_2025.py but with no cover-
date filter — the diagnostic in coverage_out_v2/per_filing.csv (Jun 2026)
was generated against an older extractor/regex combination, so re-running
the current pipeline over the current corpus can recover missed records.

Writes to a NEW jsonl (sedar_fund_strategy_all_pdfs.jsonl) — does NOT touch
the baseline sedar_fund_strategy.jsonl. Records are dedupped against the
baseline by (fundno, effective_q): only genuinely-new (fund, quarter) cells
are written.

Skips AIF-typed rows (already handled by extract_aif_only.py).
"""
import csv, json, re, sys
from collections import defaultdict, Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pypdfium2 as pdfium
from rapidfuzz import fuzz

import sedar_mapper as M
import parse_by_fund_pattern as pbf
from parse_by_fund_pattern import MANAGER_PREFIXES

PROS      = M.SNAP / "sedar_prospectuses"
BASELINE  = M.OUT / "sedar_fund_strategy.jsonl"
OUT_JSONL = M.OUT / "sedar_fund_strategy_all_pdfs.jsonl"
DIAG_CSV  = M.OUT / "sedar_per_filing_all_pdfs.csv"


def is_aif_row(row: dict) -> bool:
    """Skip AIF-typed files (handled by extract_aif_only.py)."""
    url = row.get("source_url", "").lower()
    filename = urlparse(url).path.rsplit("/", 1)[-1]
    return ("aif" in filename
            or "annual-information-form" in filename
            or "annual_information_form" in filename)


def full_text(path: str) -> str:
    doc = pdfium.PdfDocument(str(path)); pages = []
    try:
        for i in range(len(doc)):
            pg = doc[i]; tp = pg.get_textpage()
            try:  pages.append(tp.get_text_range() or "")
            finally: tp.close(); pg.close()
    finally:
        doc.close()
    return "\n".join(pages)


def extract(full: str, prefixes: list[str]) -> list[dict]:
    pat = pbf._fund_name_regex(prefixes)
    occ, canon = [], {}
    for m in pat.finditer(full):
        cn = pbf._strip_bad_leading_words(pbf._canonical_name(re.sub(r"\s+", " ", m.group(0)).strip()))
        if not cn or len(cn.split()) < 3:
            continue
        occ.append((m.start(), m.end(), cn))
        canon.setdefault(cn, {"canon": cn, "occ": 0})["occ"] += 1
    sect = M._strategy_sections(full, occ)
    return [{"canon": e["canon"], "occ": e["occ"], "strat_text": sect.get(M.core(e["canon"]), "")}
            for e in canon.values()]


def mscope(mid: str) -> set:
    sc = set()
    for p in MANAGER_PREFIXES.get(mid, []):
        toks = re.sub(r"[^A-Za-z& ]", " ", p).upper().split()
        if toks:
            sc |= M.first1.get(toks[0], set())
            if len(toks) >= 2:
                sc |= M.first2.get(toks[0] + " " + toks[1], set())
    return sc


def main():
    with open(PROS / "manifest.csv") as f:
        all_rows = list(csv.DictReader(f))
    # Keep only successfully-downloaded SP rows (skip AIF; AIF is handled elsewhere)
    sp_rows = [r for r in all_rows
               if r.get("status") == "downloaded" and not is_aif_row(r)]
    print(f"[1/4] manifest total rows                  : {len(all_rows):,}", flush=True)
    print(f"      SP-status-downloaded (non-AIF) rows  : {len(sp_rows):,}", flush=True)

    print("\n[2/4] Extracting fund strategies from each SP PDF...", flush=True)
    rows = []
    per_pdf_stats = []
    for r in sp_rows:
        mid  = r["manager"]
        path = r["local_path"]
        if mid not in MANAGER_PREFIXES:
            print(f"  skip {mid} (no prefixes registered)", flush=True)
            continue
        if not path or not Path(path).exists():
            print(f"  skip {mid} (local_path missing): {path}", flush=True)
            continue
        try:
            full = full_text(path)
        except Exception as e:
            print(f"  read-fail {mid} {Path(path).name}: {e}", flush=True); continue
        try:
            eq = pd.to_datetime(r["effective_date_cover"]).to_period("Q") \
                    .to_timestamp("D", how="end").date().isoformat()
        except Exception:
            print(f"  bad-cover-date {mid} {Path(path).name}: {r.get('effective_date_cover')}", flush=True); continue
        scope = mscope(mid)
        fex = extract(full, MANAGER_PREFIXES[mid])
        n_extracted = len(fex); n_matched = 0; n_with_text = 0
        for fe in fex:
            fno, sc, un = M.best_match(fe["canon"].upper(), scope)
            if fno is None or sc < M.ACCEPT:
                continue
            n_matched += 1
            if fe.get("strat_text"):
                n_with_text += 1
            rows.append({
                "fundno": fno, "effective_q": eq, "sedar_issuer": mid, "score": round(sc, 1),
                "occurrences": fe["occ"], "fund_name_pdf": fe["canon"], "fund_name_universe": un,
                "zip": Path(path).name, "pdf": Path(path).name,
                "strategy_text": fe["strat_text"],
            })
        per_pdf_stats.append({"manager": mid, "pdf": Path(path).name, "eff_q": eq,
                              "extracted": n_extracted, "matched": n_matched, "with_text": n_with_text})
        print(f"  {mid:<18} {eq}  ext={n_extracted:>4}  matched={n_matched:>4}  w_text={n_with_text:>4}  {Path(path).name[:60]}", flush=True)

    # Fold to one record per (fundno, quarter) — pick the best match by score,
    # preferring longer strategy text among high-score candidates.
    print(f"\n[3/4] Folding to one record per (fundno, quarter)...", flush=True)
    meta_cols = ["fundno", "effective_q", "sedar_profile", "sedar_issuer", "mgrcoab", "score",
                 "occurrences", "strategy_chars", "fund_name_pdf", "fund_name_universe",
                 "fund_name_current", "zip", "pdf"]
    bycell = defaultdict(list)
    for r in rows:
        bycell[(r["fundno"], r["effective_q"])].append(r)
    feat: dict[tuple[int, str], dict] = {}
    for (fno, eq), rs in bycell.items():
        top = max(rs, key=lambda r: r["score"]); tp = top["fund_name_pdf"].upper()
        cand = max((r for r in rs if r.get("strategy_text") and r["score"] >= top["score"] - 12
                    and fuzz.token_set_ratio(r["fund_name_pdf"].upper(), tp) >= 80),
                   key=lambda r: r["score"], default=None)
        chosen = cand or top
        rec = {c: chosen.get(c, "") for c in meta_cols}
        rec["fundno"]            = fno
        rec["sedar_profile"]     = ""
        rec["fund_name_current"] = M.fundno_cur.get(fno, "")
        rec["mgrcoab"]           = M.fundno_mgrcoab.get(fno, "")
        rec["strategy_chars"]    = len(chosen.get("strategy_text") or "")
        rec["via_sibling_of"]    = ""
        rec["strategy_text"]     = chosen.get("strategy_text", "")
        rec["strategy_src"]      = "self" if chosen.get("strategy_text") else ""
        rec["source"]            = "manager_site_reextract"
        feat[(fno, eq)] = rec

    # Dedup against baseline by (fundno, quarter) — only keep genuinely-new cells.
    with open(BASELINE) as f:
        existing = [json.loads(l) for l in f]
    ek = {(int(r["fundno"]), r["effective_q"]) for r in existing}
    # Also filter out records where our own extraction produced empty text
    # (they'd be dropped by build_text_canada_carry.py anyway).
    new_all = [rec for (fno, eq), rec in feat.items() if (int(fno), eq) not in ek]
    new_wt  = [r for r in new_all if r.get("strategy_chars", 0) > 0]

    print(f"      total (fundno, quarter) cells extracted : {len(feat):,}", flush=True)
    print(f"      cells already in SP baseline (dropped)  : {len(feat) - len(new_all):,}", flush=True)
    print(f"      NEW (fund, quarter) cells found         : {len(new_all):,}", flush=True)
    print(f"      NEW cells with non-empty strategy text  : {len(new_wt):,}", flush=True)

    print(f"\n[4/4] Writing NEW-only JSONL -> {OUT_JSONL}", flush=True)
    with open(OUT_JSONL, "w") as f:
        for rec in new_wt:      # only write records that will actually contribute text
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_JSONL}  ({OUT_JSONL.stat().st_size / 1024:.0f} KB)", flush=True)

    # Per-PDF diagnostic
    with open(DIAG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["manager", "eff_q", "pdf", "extracted", "matched", "with_text"])
        w.writeheader()
        for r in per_pdf_stats:
            w.writerow(r)
    print(f"Wrote {DIAG_CSV}", flush=True)

    # Per-manager summary
    print()
    print("Per-manager NEW-cell summary (top 20 by new cells w/ text):")
    by_mgr = Counter()
    for r in new_wt:
        by_mgr[r["sedar_issuer"]] += 1
    for mgr, n in by_mgr.most_common(20):
        print(f"  {mgr:<20}: {n:>4} NEW cells with text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
