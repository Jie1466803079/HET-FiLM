"""Pattern-based fund-name extractor.

Different approach from parse_one_prospectus.py: instead of detecting Part B
SECTION HEADERS, we directly find all occurrences of a fund-name pattern
("MANAGER + Title-Case Words + Fund/Class/Pool/...") anywhere in the PDF.
The pattern is per-manager (known prefix list). Each unique fund name is
paired with the nearest "Investment objectives" / "Investment strategies"
anchor for strategy text.

This handles three structural failure modes of the section-header approach:
  - PDFs that embed fund names inside paragraphs (AGF)
  - PDFs where section headers vary across managers (Mackenzie / NEI / Fidelity)
  - PDFs with TOC entries that look like fund names (false positives suppressed
    via dedup and minimum-frequency filter)

Public function:
    extract_funds_with_strategy(pdf_path, manager_id)
        → list[dict] with keys: fund_name_raw, fund_name_canonical,
          strategy_text, anchor_page, occurrences
"""
from __future__ import annotations
import re
from pathlib import Path

import pypdfium2 as pdfium

# Manager → list of (prefix, regex_pattern). Many managers have name variants
# (PH&N vs PHN, BMO vs BMO Mutual Funds), so we allow multiple prefixes.
MANAGER_PREFIXES: dict[str, list[str]] = {
    "rbc_gam":          ["RBC"],
    "phn":              ["PH&N", "Phillips, Hager", "Phillips Hager"],
    "mackenzie":        ["Mackenzie"],
    "bmo":              ["BMO"],
    "sun_life_gi":      ["Sun Life", "SunLife"],
    "mawer":            ["Mawer"],
    "picton_mahoney":   ["Picton Mahoney"],
    "td_am":            ["TD", "TD Mutual"],
    "nei":              ["NEI"],
    "dynamic_1832am":   ["Dynamic", "Scotia"],
    "beutel_goodman":   ["Beutel Goodman"],
    "fidelity_ca":      ["Fidelity"],
    "ci":               ["CI"],
    "franklin_templeton_ca": ["Franklin Templeton", "Franklin", "Templeton"],
    "agf":              ["AGF"],
    "ia_clarington":    ["IA Clarington", "iA Clarington", "Clarington", "iA"],
    "desjardins":       ["Desjardins"],
    "edgepoint":        ["EdgePoint", "Edgepoint"],
    # mid-tier (added 2026-06-05)
    "cibc":             ["CIBC"],
    "renaissance":      ["Renaissance", "Axiom"],
    "ig_wealth":        ["IG", "Investors Group", "iProfile"],
    "scotia_funds":     ["Scotia", "Pinnacle"],
    "empire_life":      ["Empire"],
    "horizons_globalx": ["Horizons", "Global X", "BetaPro"],
    "purpose":          ["Purpose"],
    "canada_life":      ["Canada Life", "GWL", "Great-West", "Great West"],
    "guardian":         ["Guardian"],
    "nbi":              ["NBI", "National Bank", "Westwood"],
    "middlefield":      ["Middlefield"],
}

# Fund-name suffix terms (what canonical fund names end with).
SUFFIX_TERMS = r"(?:Fund|Class|Pool|Portfolio|Trust|ETF|Suite|Index|Strategy|Series)"

# Words allowed within a fund name between prefix and suffix.
# Title-case words, conjunctions, prepositions, hyphenated terms, currency tokens.
_INNER_WORD = (
    r"(?:[A-Z][a-zA-Z\$\.\-/'’]+"
    r"|(?:[Aa]nd|[Oo]f|[Tt]he|[Ff]or|[Tt]o)"
    r"|[A-Z]\&[A-Z]"
    r"|[a-z]+\-[A-Z][a-z]+)"
)

# Series qualifiers to strip from a raw fund name to get canonical name.
SERIES_STRIPS = [
    re.compile(r"\s+MF\s+Series\s*$", re.I),
    re.compile(r"\s+Series\s+[A-Z0-9\-]+\s*$", re.I),
    re.compile(r"\s+Class\s+[A-Z0-9\-]+\s*$", re.I),
    re.compile(r"\s+\([A-Z0-9]+\)\s*$"),
    re.compile(r"\s+Advisor\s+Series\s*$", re.I),
    re.compile(r"\s+Institutional(?:\s+Series)?\s*$", re.I),
    re.compile(r"\s+ETF\s+Series\s*$", re.I),
    re.compile(r"\s+Premium\s+Series\s*$", re.I),
    re.compile(r"\s+Offering\s+Mutual\s+Fund\s+Series\s*$", re.I),
]


def _fund_name_regex(prefixes: list[str]) -> re.Pattern:
    """Build a regex that matches: PREFIX + 1-8 inner words + SUFFIX."""
    prefix_alt = "|".join(re.escape(p) for p in prefixes)
    return re.compile(
        rf"(?:{prefix_alt})\s+"
        rf"(?:{_INNER_WORD}\s+){{1,8}}"
        rf"{SUFFIX_TERMS}\b",
    )


def _canonical_name(raw: str) -> str:
    """Strip series/class qualifiers, normalise whitespace."""
    s = re.sub(r"\s+", " ", raw).strip()
    for pat in SERIES_STRIPS:
        s = pat.sub("", s)
    return s.strip()


def _strip_bad_leading_words(name: str) -> str:
    """Remove garbage tokens at start (e.g. 'AGF GROUP OF FUNDS Offering')."""
    name = re.sub(r"^[A-Z]+\s+GROUP\s+OF\s+FUNDS\s+", "", name, flags=re.I)
    return name.strip()


def extract_pages(pdf_path: Path) -> list[str]:
    """Read PDF pages as text (streaming, low memory)."""
    pages: list[str] = []
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            try:
                pages.append(tp.get_text_range() or "")
            finally:
                tp.close()
                page.close()
    finally:
        doc.close()
    return pages


# Strategy / objective anchors — any of these on its own line marks a Part B body.
# Extended 2026-07-23 with AIF-specific alternates so that Annual Information
# Form documents (which use "Fundamental investment objectives" and "Investment
# objectives and strategies" as headers instead of the SP-canonical
# "Investment strategies") also produce a hit. The multiline ^...$ requirement
# is preserved so inline occurrences (e.g. TOC entries, prose mentions) still
# do not fire.
STRATEGY_ANCHOR_RE = re.compile(
    r"^(?:\s*)(?:Investment\s+objectives?|"
    r"Investment\s+strategies?|"
    r"Investment\s+objectives?\s+and\s+strategies?|"        # AIF: combined header
    r"Fundamental\s+investment\s+objectives?|"              # AIF: NI 81-101 canonical
    r"Fundamental\s+investment\s+objectives?\s+and\s+strategies?|"
    r"Investment\s+approach|"                                # occasionally used
    r"What\s+does\s+(?:the\s+)?fund\s+invest\s+in|"
    r"What\s+the\s+fund\s+invests\s+in)\s*[:?.]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _find_strategy_after_offset(full: str, offset: int, max_chars: int = 3500) -> str:
    """Find the first strategy anchor at or after `offset`, return body."""
    m = STRATEGY_ANCHOR_RE.search(full, offset)
    if not m:
        return ""
    body = full[m.end():m.end() + max_chars]
    # Stop at next ALL-CAPS heading or another strategy-anchor occurrence
    body = re.split(r"\n\s*[A-Z][A-Z\s]{10,}\s*\n", body)[0]
    return body.strip()


def extract_funds_with_strategy(
    pdf_path: Path | str,
    manager_id: str,
    min_occurrences: int = 1,
) -> list[dict]:
    """Extract (fund_name_canonical, strategy_text) pairs from a PDF.

    Args:
        pdf_path: PDF file.
        manager_id: key into MANAGER_PREFIXES.
        min_occurrences: drop fund names that appear < this many times (likely
            false positives from intro paragraphs).

    Returns: list of dicts with keys
        fund_name_raw           first raw form of the name as it appeared
        fund_name_canonical     after stripping series qualifiers
        strategy_text           Investment-objectives body after first occurrence
        anchor_page             1-based page index of the strategy anchor
        occurrences             how many times this canonical name appeared
    """
    pdf_path = Path(pdf_path)
    prefixes = MANAGER_PREFIXES.get(manager_id)
    if not prefixes:
        return []

    pages = extract_pages(pdf_path)
    full = "\n".join(pages)
    if not full.strip():
        return []

    # Page-offset table so we can locate (offset → page index)
    page_offsets: list[int] = []
    acc = 0
    for p in pages:
        page_offsets.append(acc)
        acc += len(p) + 1   # +1 for the joining "\n"

    def offset_to_page(off: int) -> int:
        # binary-search-equivalent (small list, linear is fine)
        for i in range(len(page_offsets) - 1, -1, -1):
            if page_offsets[i] <= off:
                return i + 1
        return 1

    pat = _fund_name_regex(prefixes)
    # Group occurrences by canonical name; for each, keep first-occurrence offset
    canon_to_entry: dict[str, dict] = {}
    for m in pat.finditer(full):
        raw = re.sub(r"\s+", " ", m.group(0)).strip()
        canon = _strip_bad_leading_words(_canonical_name(raw))
        if not canon or len(canon.split()) < 3:
            continue
        # Skip generic family labels that aren't actually fund names
        if re.search(r"\bMutual\s+Fund\s+Series\b", canon, re.I):
            continue
        entry = canon_to_entry.get(canon)
        if entry is None:
            canon_to_entry[canon] = {
                "fund_name_raw": raw,
                "fund_name_canonical": canon,
                "first_offset": m.start(),
                "anchor_page": offset_to_page(m.start()),
                "occurrences": 1,
            }
        else:
            entry["occurrences"] += 1

    # Drop low-frequency
    candidates = [e for e in canon_to_entry.values() if e["occurrences"] >= min_occurrences]

    # Pair each with strategy text: scan forward from first occurrence
    out = []
    for entry in candidates:
        strategy = _find_strategy_after_offset(full, entry["first_offset"])
        out.append({
            **entry,
            "strategy_text": strategy,
            "strategy_chars": len(strategy),
        })

    out.sort(key=lambda r: (-r["occurrences"], r["fund_name_canonical"]))
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: parse_by_fund_pattern.py <pdf_path> <manager_id>")
        sys.exit(1)
    res = extract_funds_with_strategy(sys.argv[1], sys.argv[2])
    print(f"{len(res)} unique fund names extracted")
    for r in res[:40]:
        print(f"  occ={r['occurrences']:>3}  p{r['anchor_page']:>3}  "
              f"strat_chars={r['strategy_chars']:>4}  {r['fund_name_canonical']!r}")
