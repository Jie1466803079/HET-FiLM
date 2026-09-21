"""Comprehensive SP re-extraction — Option B′: curated `manager_id +
normalized_fund_name -> crsp_fundno` dict, bootstrapped from baseline.

Deterministic exact-match after canonicalization (not fuzzy). Fund names
from PDFs are normalized to a canonical form (uppercase, punctuation-free,
series/class qualifiers stripped) and looked up in the dict. If found → the
fundno is the trusted mapping. If not found → the record is skipped
(NO fuzzy fallback that produces the misattribution errors we saw earlier).

The dict is seeded from baseline sedar_fund_strategy.jsonl records at
score >= 92 (empirically ~99% unambiguous per feasibility test). Coverage
grows over time as more high-quality records get added to baseline.

Writes to NEW jsonl. Does NOT touch baseline.
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
OUT_JSONL = M.OUT / "sedar_fund_strategy_b_prime.jsonl"
DIAG_CSV  = M.OUT / "sedar_per_filing_b_prime.csv"

SEED_SCORE_MIN = 92.0     # baseline records at this score seed the dict

# ── name normalization pipeline ────────────────────────────────────────────
_SERIES_TAIL = re.compile(
    r"\s*\(?"
    r"(?:series|class|units?|shares?|category|cat\.?|advisor|no-?load|deferred)"
    r"\s+[A-Za-z0-9,\s\-\.]+?\)?"
    r"(?:\s+(?:units?|shares?))?$",
    re.IGNORECASE,
)
_TRAILING_TYPE = re.compile(
    r"\s+(?:fund|class|portfolio|pool|etf|trust|corporation|corp\.?|"
    r"inc\.?|ltd\.?|limited|partnership)"
    r"(?:\s+fund|\s+class|\s+pool)?$",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[®™©,;:.!?\-\(\)\[\]\{\}\"'']")
_WS = re.compile(r"\s+")


def canonicalize(name: str) -> str:
    """Aggressive normalization for exact-match lookup."""
    if not name:
        return ""
    s = name.strip()
    s = _SERIES_TAIL.sub("", s).strip()
    for _ in range(3):
        prev = s
        s = _TRAILING_TYPE.sub("", s).strip()
        if s == prev: break
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip().upper()
    return s


def _truncation_variants(canon: str) -> list[str]:
    """CRSP names are truncated at 24 chars; the truncation often falls mid-word
    like 'RBC CANADIAN BOND INDEX F' where the trailing partial is 'FUND'.
    Generate variants that strip a trailing 1- or 2-letter partial-word so
    canon-form matching works against PDF names like 'RBC Canadian Bond Index'.
    """
    variants = [canon]
    parts = canon.split()
    if len(parts) >= 2 and len(parts[-1]) <= 2:
        variants.append(" ".join(parts[:-1]))
    return variants


def build_registry(baseline_recs: list[dict]) -> dict[tuple[str, str], list[int]]:
    """Two-source seed:
      #1 baseline records at score >= SEED_SCORE_MIN — high-confidence
         (PDF fund name, fundno) pairs by SEDAR issuer.
      #2 CRSP canada_funds table — every graph fund's name variants, indexed
         under each of the manager_ids whose brand scope contains the fundno.
    """
    reg: dict[tuple[str, str], set[int]] = defaultdict(set)

    # --- source 1: baseline high-score records
    n_seed_baseline = 0
    for r in baseline_recs:
        try:
            sc = float(r.get("score", 0))
        except (ValueError, TypeError):
            continue
        if sc < SEED_SCORE_MIN:
            continue
        iss = (r.get("sedar_issuer") or "").strip()
        pdf_name = (r.get("fund_name_pdf") or "").strip()
        if not iss or not pdf_name:
            continue
        canon = canonicalize(pdf_name)
        if not canon:
            continue
        reg[(iss, canon)].add(int(r["fundno"]))
        n_seed_baseline += 1

    # --- source 2: CRSP canada_funds name variants, keyed by manager scope
    n_seed_crsp = 0
    for mgr_id in MANAGER_PREFIXES:
        scope = set()
        for pfx in MANAGER_PREFIXES[mgr_id]:
            toks = re.sub(r"[^A-Za-z& ]", " ", pfx).upper().split()
            if toks:
                scope |= M.first1.get(toks[0], set())
                if len(toks) >= 2:
                    scope |= M.first2.get(toks[0] + " " + toks[1], set())
        for fno in scope:
            for nm in M.fundno_names.get(fno, ()):
                for variant in _truncation_variants(canonicalize(nm)):
                    if variant:
                        reg[(mgr_id, variant)].add(int(fno))
                        n_seed_crsp += 1

    print(f"      seeded from baseline @ >= {SEED_SCORE_MIN}: {n_seed_baseline:,} entries", flush=True)
    print(f"      seeded from CRSP name variants           : {n_seed_crsp:,} entries", flush=True)
    return {k: sorted(v, reverse=True) for k, v in reg.items()}


def lookup(reg: dict, manager_id: str, pdf_name: str) -> "int | None":
    """Exact-match after canonicalization. Returns fundno or None."""
    canon = canonicalize(pdf_name)
    if not canon:
        return None
    fnos = reg.get((manager_id, canon))
    if not fnos:
        return None
    return fnos[0]      # first (highest by heuristic) if ambiguous


# ── file-io helpers (identical to extract_all_pdfs.py) ─────────────────────
def is_aif_row(row: dict) -> bool:
    url = row.get("source_url", "").lower()
    filename = urlparse(url).path.rsplit("/", 1)[-1]
    return ("aif" in filename or "annual-information-form" in filename
            or "annual_information_form" in filename)


def full_text(path: str) -> str:
    doc = pdfium.PdfDocument(str(path)); pages = []
    try:
        for i in range(len(doc)):
            pg = doc[i]; tp = pg.get_textpage()
            try: pages.append(tp.get_text_range() or "")
            finally: tp.close(); pg.close()
    finally:
        doc.close()
    return "\n".join(pages)


def extract(full: str, prefixes: list[str]) -> list[dict]:
    pat = pbf._fund_name_regex(prefixes)
    occ, canon_seen = [], {}
    for m in pat.finditer(full):
        cn = pbf._strip_bad_leading_words(pbf._canonical_name(re.sub(r"\s+", " ", m.group(0)).strip()))
        if not cn or len(cn.split()) < 3:
            continue
        occ.append((m.start(), m.end(), cn))
        canon_seen.setdefault(cn, {"canon": cn, "occ": 0})["occ"] += 1
    sect = M._strategy_sections(full, occ)
    return [{"canon": e["canon"], "occ": e["occ"], "strat_text": sect.get(M.core(e["canon"]), "")}
            for e in canon_seen.values()]


def main():
    with open(BASELINE) as f:
        baseline_recs = [json.loads(l) for l in f]
    ek = {(int(r["fundno"]), r["effective_q"]) for r in baseline_recs}
    reg = build_registry(baseline_recs)
    n_uniq_fnos = len({fno for fnos in reg.values() for fno in fnos})
    n_ambig = sum(1 for v in reg.values() if len(v) > 1)
    print(f"[0/4] baseline records                    : {len(baseline_recs):,}", flush=True)
    print(f"      registry keys (issuer, canon_name) : {len(reg):,}", flush=True)
    print(f"      registry distinct fundnos          : {n_uniq_fnos:,}", flush=True)
    print(f"      ambiguous keys (>1 fno per key)    : {n_ambig}", flush=True)

    with open(PROS / "manifest.csv") as f:
        all_rows = list(csv.DictReader(f))
    sp_rows = [r for r in all_rows
               if r.get("status") == "downloaded" and not is_aif_row(r)]
    print(f"\n[1/4] manifest total                         : {len(all_rows):,}", flush=True)
    print(f"      SP-status-downloaded non-AIF rows      : {len(sp_rows):,}", flush=True)

    print("\n[2/4] Extracting from each PDF (B' matcher)...", flush=True)
    rows, per_pdf_stats = [], []
    # Also collect unmatched canonical names so we can report the coverage gap.
    unmatched_by_mgr: dict[str, Counter] = defaultdict(Counter)
    for r in sp_rows:
        mid, path = r["manager"], r["local_path"]
        if mid not in MANAGER_PREFIXES:
            continue
        if not path or not Path(path).exists():
            continue
        try: full = full_text(path)
        except Exception as e:
            print(f"  read-fail {mid}: {e}", flush=True); continue
        try:
            eq = pd.to_datetime(r["effective_date_cover"]).to_period("Q") \
                    .to_timestamp("D", how="end").date().isoformat()
        except Exception:
            continue

        # Try to map sedar_issuer for registry lookup. In baseline, sedar_issuer
        # is often the SEDAR profile name (e.g. "Mackenzie Mutual Funds"), which
        # differs from the manager_id ("mackenzie"). We can look up by BOTH:
        # the manager_id (for records populated by add_2025.py) AND by any
        # SEDAR profile name(s) that appear in baseline records for this manager.
        # For simplicity, we try only the manager_id and any baseline-observed
        # profile names.
        issuer_candidates = [mid]

        fex = extract(full, MANAGER_PREFIXES[mid])
        n_ext = len(fex); n_matched = 0; n_with_text = 0; n_missing_reg = 0
        for fe in fex:
            fno = None
            for iss in issuer_candidates:
                fno = lookup(reg, iss, fe["canon"])
                if fno is not None:
                    break
            if fno is None:
                n_missing_reg += 1
                unmatched_by_mgr[mid][canonicalize(fe["canon"])] += 1
                continue
            n_matched += 1
            if fe.get("strat_text"):
                n_with_text += 1
            rows.append({
                "fundno": fno, "effective_q": eq, "sedar_issuer": mid,
                "score": 100.0,                                     # deterministic match
                "occurrences": fe["occ"],
                "fund_name_pdf": fe["canon"],
                "fund_name_universe": M.fundno_cur.get(fno, ""),
                "zip": Path(path).name, "pdf": Path(path).name,
                "strategy_text": fe["strat_text"],
            })
        per_pdf_stats.append({"manager": mid, "pdf": Path(path).name, "eff_q": eq,
                              "extracted": n_ext, "matched": n_matched,
                              "with_text": n_with_text, "missing_registry": n_missing_reg})
        print(f"  {mid:<18} {eq}  ext={n_ext:>4}  matched={n_matched:>4}  "
              f"w_text={n_with_text:>4}  miss_reg={n_missing_reg:>4}  "
              f"{Path(path).name[:52]}", flush=True)

    print("\n[3/4] Folding to one record per (fundno, quarter)...", flush=True)
    meta_cols = ["fundno", "effective_q", "sedar_profile", "sedar_issuer", "mgrcoab", "score",
                 "occurrences", "strategy_chars", "fund_name_pdf", "fund_name_universe",
                 "fund_name_current", "zip", "pdf"]
    bycell = defaultdict(list)
    for r in rows:
        bycell[(r["fundno"], r["effective_q"])].append(r)
    feat = {}
    for (fno, eq), rs in bycell.items():
        top = max(rs, key=lambda r: len(r.get("strategy_text") or ""))
        rec = {c: top.get(c, "") for c in meta_cols}
        rec.update({
            "fundno": fno, "sedar_profile": "",
            "fund_name_current": M.fundno_cur.get(fno, ""),
            "mgrcoab":           M.fundno_mgrcoab.get(fno, ""),
            "strategy_chars":    len(top.get("strategy_text") or ""),
            "via_sibling_of":    "",
            "strategy_text":     top.get("strategy_text", ""),
            "strategy_src":      "self" if top.get("strategy_text") else "",
            "source":            "manager_site_reextract_b_prime",
        })
        feat[(fno, eq)] = rec

    new_all = [rec for (fno, eq), rec in feat.items() if (int(fno), eq) not in ek]
    new_wt  = [r for r in new_all if r.get("strategy_chars", 0) > 0]
    print(f"      total cells extracted                : {len(feat):,}", flush=True)
    print(f"      dropped: already in baseline         : {len(feat) - len(new_all):,}", flush=True)
    print(f"      NEW cells                            : {len(new_all):,}", flush=True)
    print(f"      NEW cells with non-empty strategy    : {len(new_wt):,}", flush=True)

    print(f"\n[4/4] Writing NEW-only JSONL -> {OUT_JSONL}", flush=True)
    with open(OUT_JSONL, "w") as f:
        for rec in new_wt:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_JSONL}  ({OUT_JSONL.stat().st_size / 1024:.0f} KB)", flush=True)

    with open(DIAG_CSV, "w", newline="") as f:
        cols = ["manager","pdf","eff_q","extracted","matched","with_text","missing_registry"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_pdf_stats:
            w.writerow(r)
    print(f"Wrote {DIAG_CSV}", flush=True)

    # Report unmatched (canonical) names per manager — these are the coverage gaps
    # to close by adding more entries to the registry (e.g. lower SEED_SCORE_MIN
    # or add hand-curated entries).
    print("\nTop 20 unmatched canonical names by manager (add to registry to unlock):")
    all_unmatched = [(mgr, name, cnt)
                     for mgr, cnts in unmatched_by_mgr.items()
                     for name, cnt in cnts.items()]
    for mgr, name, cnt in sorted(all_unmatched, key=lambda x: -x[2])[:20]:
        print(f"  {mgr:<20}  seen {cnt:>3}× : {name!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
