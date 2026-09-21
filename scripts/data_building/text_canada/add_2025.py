"""Ingest 2025 prospectus text from the manager-site set (sedar_prospectuses/),
which the bulk SEDAR+ scrape (2015-2024) is missing. Per-manager books ->
MANAGER_PREFIXES; reuse the FIXED per-fund strategy segmentation + matching.
NO share-class folding (universe is fund-portfolio based). Appends 2025 records."""
import csv, json, re, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium
from rapidfuzz import fuzz

import sedar_mapper as M
import parse_by_fund_pattern as pbf
from parse_by_fund_pattern import MANAGER_PREFIXES

PROS = M.SNAP / "sedar_prospectuses"
man = [r for r in csv.DictReader(open(PROS / "manifest.csv"))
       if r["status"] == "downloaded" and r["effective_date_cover"].startswith("2025")]
print(f"2025 downloaded PDFs in manifest: {len(man)}", flush=True)


def full_text(path):
    doc = pdfium.PdfDocument(str(path)); ps = []
    try:
        for i in range(len(doc)):
            pg = doc[i]; tp = pg.get_textpage()
            try: ps.append(tp.get_text_range() or "")
            finally: tp.close(); pg.close()
    finally:
        doc.close()
    return "\n".join(ps)


def extract(full, prefixes):
    pat = pbf._fund_name_regex(prefixes)
    occ, canon = [], {}
    for m in pat.finditer(full):
        cn = pbf._strip_bad_leading_words(pbf._canonical_name(re.sub(r"\s+", " ", m.group(0)).strip()))
        if not cn or len(cn.split()) < 3:
            continue
        occ.append((m.start(), m.end(), cn)); canon.setdefault(cn, {"canon": cn, "occ": 0})["occ"] += 1
    sect = M._strategy_sections(full, occ)
    return [{"canon": e["canon"], "occ": e["occ"], "strat_text": sect.get(M.core(e["canon"]), "")}
            for e in canon.values()]


def mscope(mid):
    sc = set()
    for p in MANAGER_PREFIXES.get(mid, []):
        toks = re.sub(r"[^A-Za-z& ]", " ", p).upper().split()
        if toks:
            sc |= M.first1.get(toks[0], set())
            if len(toks) >= 2:
                sc |= M.first2.get(toks[0] + " " + toks[1], set())
    return sc


rows = []
for r in man:
    mid = r["manager"]; path = r["local_path"]
    if mid not in MANAGER_PREFIXES:
        print(f"  skip {mid} (no prefixes)", flush=True); continue
    try: full = full_text(path)
    except Exception as e:
        print(f"  read-fail {mid}: {e}", flush=True); continue
    eq = pd.to_datetime(r["effective_date_cover"]).to_period("Q").to_timestamp("D", how="end").date().isoformat()
    scope = mscope(mid)
    fex = extract(full, MANAGER_PREFIXES[mid])
    acc = 0
    for fe in fex:
        fno, sc, un = M.best_match(fe["canon"].upper(), scope)
        if fno is None or sc < M.ACCEPT:
            continue
        rows.append({"fundno": fno, "effective_q": eq, "sedar_issuer": mid, "score": round(sc, 1),
                     "occurrences": fe["occ"], "fund_name_pdf": fe["canon"], "fund_name_universe": un,
                     "zip": Path(path).name, "pdf": Path(path).name, "strategy_text": fe["strat_text"]})
        acc += 1
    print(f"  {mid:18} {eq} scope={len(scope):>4} ext={len(fex):>3} matched={acc:>3}", flush=True)

meta = ["fundno", "effective_q", "sedar_profile", "sedar_issuer", "mgrcoab", "score",
        "occurrences", "strategy_chars", "fund_name_pdf", "fund_name_universe",
        "fund_name_current", "zip", "pdf"]
bycell = defaultdict(list)
for r in rows:
    bycell[(r["fundno"], r["effective_q"])].append(r)
feat = {}
for (fno, eq), rs in bycell.items():           # one record per (fundno, quarter); NO folding
    top = max(rs, key=lambda r: r["score"]); tp = top["fund_name_pdf"].upper()
    cand = max((r for r in rs if r.get("strategy_text") and r["score"] >= top["score"] - 12
                and fuzz.token_set_ratio(r["fund_name_pdf"].upper(), tp) >= 80),
               key=lambda r: r["score"], default=None)
    chosen = cand or top; shadow = chosen is not top
    rec = {c: chosen.get(c, "") for c in meta}
    rec["fundno"] = fno; rec["sedar_profile"] = ""
    rec["fund_name_current"] = M.fundno_cur.get(fno, ""); rec["mgrcoab"] = M.fundno_mgrcoab.get(fno, "")
    rec["strategy_chars"] = len(chosen.get("strategy_text") or "")
    rec["via_sibling_of"] = ""
    rec["strategy_text"] = chosen.get("strategy_text", "")
    rec["strategy_src"] = ("" if not chosen.get("strategy_text") else "shadow" if shadow else "self")
    rec["source"] = "manager_site_2025"
    feat[(fno, eq)] = rec

existing = [json.loads(l) for l in open(M.OUT / "sedar_fund_strategy.jsonl")]
ek = set((int(r["fundno"]), r["effective_q"]) for r in existing)
new = [rec for k, rec in feat.items() if (int(rec["fundno"]), rec["effective_q"]) not in ek]
with open(M.OUT / "sedar_fund_strategy.jsonl", "a") as f:
    for rec in new:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
with open(M.OUT / "sedar_per_filing_2025.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=meta, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
nf = len(set(r["fundno"] for r in new if r.get("strategy_text")))
print(f"\nappended {len(new)} 2025 records ({nf} equity-MF funds with a 2025 strategy); "
      f"raw 2025 matches -> sedar_per_filing_2025.csv ({len(rows)} rows)", flush=True)
