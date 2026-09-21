#!/usr/bin/env python3
"""
V2 HTML/text extractor for 485BPOS filings.

Targets the four failure modes diagnosed for pre-2011 filings while leaving
the XBRL path (which delivers ~99% coverage from 2012 onward) untouched.

Failure modes addressed:
  1. Heading variants — adds patterns for "PRINCIPAL STRATEGY",
     "Our investment strategies", "What are the Fund's main investment
     strategies?", "How the Fund invests", bare "Investment Strategy",
     "Investment Strategies and Policies".
  2. Body terminator biting prose — section body now runs from heading end
     to the next *real* heading position (line-start, non-TOC), not to the
     next occurrence of a terminator word anywhere in the prose.
  3. TOC entries matched as headings — every heading match is filtered by
     line-start anchoring and TOC-style rejection (leader dots, trailing
     page number, multi-column page-list pattern).
  4. No-SERIES-ID single-fund filings — when the SGML header has 0 or 1
     SERIES-ID, proximity disambiguation is skipped (the first non-TOC
     body heading is reliable).

XBRL extraction is reused from extract_485bpos_sections (v1) unchanged.

Library only — no main(). Pair with compare_v1_v2_sample.py to evaluate
recovery on the failing subset of v1 output before committing to a full
re-extraction.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse v1 utilities to avoid duplicating well-tested code
from extract_485bpos_sections import (  # noqa: E402
    strip_tags,
    _get_prospectus_documents,
    extract_xbrl,
    _looks_like_prose,
    _PROSE_GATED_SECTIONS,
    _EXPECTED_SECTIONS,
    _read_filing_date,
    _get_pit_fund_name,
)
# Reuse the existing fund-section slicer for cross-fund routing. The slicer
# does robust tokenization (drops org suffixes like "Fund"/"Trust" that don't
# always appear in body text), strict token-order matching, and handles
# "and"/"&" aliasing — strictly better than v1's _name_search_regex for
# multi-fund mega-prospectuses where the target fund sits at char 400K-1M.
from fund_section_slicer import (  # noqa: E402
    _fund_name_tokens,
    _build_fund_name_pattern,
    _find_anchor_after,
    SECTION_ANCHORS,
)


# ── Heading patterns (sections we extract) ───────────────────────────

# Each entry: (compiled regex, strict_line_end_required).
# strict_line_end=True is reserved for "bare" patterns prone to matching body
# prose (e.g. ".. due to differences in the\\ninvestment strategies or
# restrictions..."). Specific multi-word patterns can use the looser check.
_EXTRACT_PATTERNS = {
    "objective": [
        (re.compile(r"investment\s+(?:objective|goal)s?", re.I), True),
        (re.compile(
            r"(?:the\s+)?fund(?:'|’)?s\s+(?:investment\s+)?(?:objective|goal)s?",
            re.I,
        ), True),
        (re.compile(
            r"what\s+is\s+the\s+fund(?:'|’)?s\s+(?:investment\s+)?(?:goal|objective)s?\??",
            re.I,
        ), False),
    ],
    "strategy": [
        # COMPOUND first: "Investment Strategies and Risks" / "Principal
        # Investment Strategies and Principal Risks" / "Investment Strategy
        # and Risks". Some 2005-era prospectuses (LifeStyle, Allegiant) put
        # both strategy and risk content under one compound heading; we
        # keep the whole section together per user preference. Must come
        # FIRST so the dedup keeps the longer (compound) match end-position;
        # otherwise the bare "Principal Investment Strategies" pattern wins
        # and the body starts at "and Risks <body>...", which is wrong.
        (re.compile(
            r"(?:principal\s+|main\s+|the\s+fund(?:'|’)?s\s+)?"
            r"investment\s+strateg(?:y|ies)\s+and\s+"
            r"(?:principal\s+|main\s+)?risks?",
            re.I,
        ), False),
        # Canonical (covers "Principal Investment Strategies" and "Principal Strategy")
        (re.compile(r"principal\s+(?:investment\s+)?strateg(?:y|ies)", re.I), False),
        (re.compile(r"investment\s+goal\s+and\s+principal\s+strateg", re.I), False),
        # Variants observed in 2005-2009 filings
        (re.compile(r"our\s+(?:principal\s+)?investment\s+strateg(?:y|ies)", re.I), False),
        (re.compile(
            r"what\s+are\s+the\s+fund(?:'|’)?s\s+"
            r"(?:main\s+|principal\s+)?investment\s+strateg(?:y|ies)\??",
            re.I,
        ), False),
        (re.compile(r"how\s+(?:does\s+)?the\s+fund\s+invests?(?:\s+its\s+assets)?", re.I), False),
        (re.compile(r"investment\s+strateg(?:y|ies)\s+and\s+policies", re.I), False),
        # Last resort: bare "Investment Strategy" — STRICT (must end a line)
        (re.compile(r"(?<![a-z])investment\s+strateg(?:y|ies)\b", re.I), True),
    ],
    "risk": [
        (re.compile(r"principal\s+(?:investment\s+)?risks?", re.I), False),
        (re.compile(r"primary\s+risks?(?:\s+for\s+(?:the\s+)?fund)?", re.I), False),
        (re.compile(r"(?:main|principal)\s+risks?\s+of\s+investing", re.I), False),
        (re.compile(r"main\s+risks?(?:\s+of\s+investing\s+in\s+the\s+fund)?", re.I), False),
        (re.compile(r"risk\s+factors?(?:\s+and\s+special(?:\s+considerations?)?)?", re.I), False),
        (re.compile(r"what\s+are\s+the\s+(?:main|principal)\s+risks?", re.I), False),
    ],
}

# ── Terminator patterns (used to bound section body, not extracted) ──

_TERMINATOR_PATTERNS = [
    re.compile(r"fees?\s+and\s+expenses(?:\s+of\s+the\s+fund)?", re.I),
    re.compile(r"shareholder\s+fees", re.I),
    re.compile(r"annual\s+fund\s+operating\s+expenses", re.I),
    re.compile(r"(?:annual\s+)?fee\s+table", re.I),
    re.compile(r"past\s+performance", re.I),
    re.compile(r"performance\s+(?:summary|history|information)", re.I),
    re.compile(r"performance\s+of\s+the\s+fund", re.I),
    re.compile(r"(?:average\s+)?annual\s+total\s+returns?", re.I),
    re.compile(r"portfolio\s+turnover", re.I),
    re.compile(r"management\s+of\s+the\s+fund", re.I),
    re.compile(r"portfolio\s+manager(?:s)?", re.I),
    re.compile(r"investment\s+adviser(?:\s+and\s+portfolio\s+manager)?", re.I),
    re.compile(r"purchase\s+and\s+sale\s+of\s+fund\s+shares", re.I),
    re.compile(r"tax\s+information", re.I),
    re.compile(r"payments\s+to\s+broker-dealers", re.I),
    re.compile(r"(?:additional|other)\s+investment\s+strateg", re.I),
    re.compile(r"(?:summary|overview)\s+of\s+(?:the\s+)?fund", re.I),
    re.compile(r"financial\s+highlights", re.I),
    re.compile(r"distributions?\s+and\s+taxes?", re.I),
    re.compile(r"table\s+of\s+contents", re.I),
    re.compile(r"more\s+about\s+the\s+fund", re.I),
    # Case-SENSITIVE: all-caps "X RISK(S)" subsection headers. Catches
    # "STOCK MARKET RISKS", "FOREIGN CURRENCY RISK", "CREDIT RISK", etc.
    # without firing on body prose like "stock market risks". 3+ char
    # preamble required to avoid one-letter noise.
    re.compile(r"[A-Z][A-Z\s/&\-,]{2,}RISKS?\b"),
]


# ── Heading detection (line-start + TOC rejection) ──────────────────


def _at_line_start(text: str, pos: int) -> bool:
    """True if `pos` begins a line (only whitespace on the line up to pos)."""
    line_start = text.rfind("\n", 0, pos) + 1
    return text[line_start:pos].strip() == ""


def _is_real_heading(text: str, m_start: int, m_end: int,
                     strict_line_end: bool = False) -> bool:
    """Qualifier: True if the match is structurally a heading.

    Required: the match must BEGIN a logical section break:
      (a) start at line beginning (whitespace-only prefix), OR
      (b) preceded by sentence-final punctuation on the same line, OR
      (c) preceded by a fund-name-like noun-phrase ending — last word on
          the line is capitalized and ends with "Fund", "Trust",
          "Portfolio", "Series", or is a recognized fund-name suffix.
          Catches collapsed-HTML cases like "Brown Advisory Growth Equity
          Fund Principal Investment Strategies Under normal..." where
          strip_tags produced 5 newlines across 1.27 MB.

    Additional gate for low-specificity patterns (`strict_line_end=True`):
      the match must END a line — newline within ~10 chars of match end,
      no body content between. This filters body wraps like "...due to
      differences in the\\ninvestment strategies or restrictions..."
      where the bare "investment strategies" pattern incidentally lands
      at line start mid-sentence.
    """
    if strict_line_end:
        after = text[m_end:m_end + 10]
        nl_idx = after.find("\n")
        if nl_idx == -1:
            return False
        if after[:nl_idx].strip() != "":
            return False

    if _at_line_start(text, m_start):
        return True
    line_start = text.rfind("\n", 0, m_start) + 1
    prefix = text[line_start:m_start]
    prefix_stripped = prefix.rstrip()
    # Short article preamble at line start ("The Principal Investment Strategies
    # and Policies of the Fund", "Our investment strategies"). The article makes
    # _at_line_start return False even though the heading is structurally on
    # its own line.
    if prefix_stripped.lower() in {"the", "our", "an", "a", "its"}:
        return True
    if prefix_stripped.endswith((".", "?", "!", ":", ")")):
        return True
    # Fund-name preamble: last word is a fund-noun ("Fund", "Trust", ...)
    # AND the preceding word is also capitalized. This catches multi-word
    # proper nouns like "Brown Advisory Growth Equity Fund" or "DOMESTIC
    # FUNDS" while rejecting body prose like "...economic trends and Fund
    # investment strategies that significantly affected..." where "Fund"
    # is preceded by a lowercase function word.
    last_two = re.search(r"(\w+)\s+(\w+)\s*$", prefix_stripped)
    if last_two:
        prev, lw = last_two.group(1), last_two.group(2)
        if (lw[0].isupper() and lw.lower() in {
            "fund", "funds", "trust", "trusts", "portfolio", "portfolios",
            "series", "inc", "corp", "company", "account", "etf", "etfs",
        } and prev and prev[0].isupper()):
            return True
    return False


def _is_toc_line(text: str, m_start: int, m_end: int) -> bool:
    """Heuristic: True if the match is a Table-of-Contents entry.

    A heading is treated as TOC when any of these holds:
      (a) The line containing the match has leader dots (e.g. ".....")
      (b) The line ends with a small integer (page number) after some
          whitespace, AND the line is short (heading-like length)
      (c) Within ~40 chars after the match, the next non-blank text is
          "<spaces><digits>\\n" or "<spaces><digits> <Capitalized>"
          (single-line or multi-column TOC pattern)
    """
    line_start = text.rfind("\n", 0, m_start) + 1
    line_end = text.find("\n", m_end)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]

    if re.search(r"\.{3,}", line):
        return True
    if len(line) < 200 and re.search(r"\s{2,}\d{1,4}\s*$", line):
        return True

    after = text[m_end:min(len(text), m_end + 60)]
    # "<heading> 2\n" → TOC entry with page number
    if re.match(r"\s{1,20}\d{1,4}\s*\n", after):
        return True
    # "<heading> 2 <NextHeading>" → multi-column TOC row
    if re.match(r"\s{1,20}\d{1,4}\s+[A-Z][A-Za-z][A-Za-z]+", after):
        return True
    # Bare match of "Investment Strategies" followed by "and Risks
    # [intermediate words] <page#>" — TOC entries that the bare canonical
    # pattern consumes only partially. Janus 2005 has the tight form
    # "Investment Strategies and Risks 73"; some prospectuses have
    # "Investment Strategies and Risks Respecting the Active Income Fund 4"
    # with intermediate words before the page number. Up to ~10
    # intermediate words still counts as a TOC entry.
    if re.match(
        r"\s+(?:and|or)\s+\w+s?(?:\s+\w[\w'-]*){0,10}\s+\d{1,3}\s*\n",
        after, re.I,
    ):
        return True
    # Multi-line layout: "<heading text wraps>\n<page#>\n<next heading>"
    # Common in older prospectus TOCs that put page numbers on their own line.
    end_chunk = text[m_end:min(len(text), m_end + 300)]
    if re.match(r"[^\n]{0,200}\n\s*\d{1,4}\s*\n", end_chunk):
        return True
    # TOC entry on the line AFTER the heading match:
    # "<heading>\n<TOC continuation ending with page#>\n"
    # 1st line below heading is a short text ending in a small integer.
    if re.match(
        r"\n[^\n]{0,80}\s\d{1,3}\s*\n",
        end_chunk,
    ):
        return True
    # Multi-line wrapped TOC: heading match was inside a TOC line where
    # the page number is several lines down. Example (S000000850 2007):
    #   "Principal Investment Strategies and Risks Respecting\n\nthe
    #    Micro Cap Fund\n4\n\nPrincipal Investment Strategies and Risks
    #    Respecting\n\nthe Active Income Fund\n4\n..."
    # Two short text segments (3-80 chars each) followed by a digit alone.
    if re.match(
        r"[^\n]{0,80}\n+[^\n]{3,80}\n+\d{1,3}\s*\n",
        end_chunk,
    ):
        return True
    return False


def _find_all_headings(clean: str):
    """Return sorted [(start, end, section_or_'_term')] of detected headings.

    A heading qualifies only if:
      - It is at line start (whitespace-only prefix on its line)
      - The line is not a TOC entry

    Includes terminator patterns (tagged '_term') so callers can use any
    subsequent heading to bound a section body, regardless of type.
    """
    out = []
    for sect, entries in _EXTRACT_PATTERNS.items():
        for pat, strict in entries:
            for m in pat.finditer(clean):
                if not _is_real_heading(clean, m.start(), m.end(), strict_line_end=strict):
                    continue
                if _is_toc_line(clean, m.start(), m.end()):
                    continue
                out.append((m.start(), m.end(), sect))
    for pat in _TERMINATOR_PATTERNS:
        for m in pat.finditer(clean):
            if not _is_real_heading(clean, m.start(), m.end()):
                continue
            if _is_toc_line(clean, m.start(), m.end()):
                continue
            out.append((m.start(), m.end(), "_term"))
    # Sort by start position only (stable). Tuple sort would tie-break on end,
    # putting the shorter-end match first — which then wins dedup, dropping
    # longer compound matches ("Investment Strategies and Risks" → kept as
    # "Investment Strategies"). Stable sort preserves pattern-iteration order,
    # so the compound pattern (listed first in _EXTRACT_PATTERNS) wins.
    out.sort(key=lambda x: x[0])

    # Dedup matches < 30 chars apart (same heading caught by multiple regex).
    # When dedup collapses a terminator and an extractable section at the
    # same position, prefer the extractable label.
    dedup = []
    for s, e, sect in out:
        if dedup and s - dedup[-1][0] < 30:
            if dedup[-1][2] == "_term" and sect != "_term":
                dedup[-1] = (s, e, sect)
            continue
        dedup.append((s, e, sect))
    return dedup


# ── Helpers for multi-fund disambiguation ────────────────────────────


_XBRL_INFRA_FILENAME = re.compile(r"(?:^r\d+\.htm$|\.xml$|\.js$|\.xsd$)", re.I)
_XBRL_INFRA_TYPE = re.compile(r"^EX-101", re.I)


def _is_xbrl_infra(d: str) -> bool:
    """True if document is XBRL infrastructure (R*.htm, EX-101.*, .xml, .js)."""
    head = d[:500]
    type_m = re.search(r"<TYPE>([^\s<\n]+)", head)
    if type_m and _XBRL_INFRA_TYPE.match(type_m.group(1)):
        return True
    fname_m = re.search(r"<FILENAME>([^\s<\n]+)", head)
    if fname_m and _XBRL_INFRA_FILENAME.search(fname_m.group(1)):
        return True
    return False


def _v2_prospectus_documents(content: str):
    """Yield prospectus candidate docs WITH an `is_xbrl_infra` flag.

    Returns list of (doc, is_xbrl_infra) tuples. XBRL infrastructure files
    (EX-101.INS, R*.htm viewer files) are still included because some
    filings have the only prospectus prose embedded there (when the
    485BPOS HTM is just a cover page). They are flagged so the caller
    can apply stricter prose gating to bodies extracted from them:
    chrome-only extractions like "Investment Strategy, Heading
    rr_StrategyHeading ..." get rejected by `_looks_like_prose`, while
    real prose embedded in the viewer file (e.g. Valley Forge 2013's
    "...The Fund's principal investment strategy is value investing...")
    passes through.
    """
    return [(d, _is_xbrl_infra(d)) for d in _get_prospectus_documents(content)]


def _count_series_in_header(content: str) -> int:
    """Count distinct SERIES-IDs in the SGML header.

    Used to decide whether multi-fund proximity disambiguation is needed.
    Returns 0 if the filing predates SERIES-ID conventions.
    """
    head = content[:200_000]
    return len(set(re.findall(r"<SERIES-ID>\s*(S\d+)", head)))


def _find_name_positions(clean: str, fund_name: str):
    """Find fund-name positions in cleaned text, excluding TOC mentions.

    Uses fund_section_slicer's tokenization + pattern builder, which is
    materially more robust than v1's `_name_search_regex`:
      - Drops org suffixes ("Fund", "Trust", "Inc", "Series", "Portfolio")
        so "Brown Advisory Growth Equity Fund" matches body prose that
        only says "Brown Advisory Growth Equity"
      - Requires tokens in consecutive order with only whitespace/
        punctuation between them — no spurious matches via far-apart
        tokens
      - Handles "and"/"&"/"&amp;" alternation natively

    TOC-style mentions (leader dots, trailing page number) are still
    rejected here so proximity matching anchors on body occurrences.
    """
    if not fund_name:
        return []
    tokens = _fund_name_tokens(fund_name)
    pat = _build_fund_name_pattern(tokens)
    if pat is None:
        return []
    # The slicer's pattern operates on lowercased text; clean is mixed
    # case. We can either lowercase clean or compile with re.IGNORECASE
    # (the slicer already sets re.IGNORECASE). Use the pattern as-is.
    positions = []
    for m in pat.finditer(clean):
        lookahead = clean[m.end():m.end() + 80]
        if re.search(r"\.{4,}|\s\d{1,4}\s*$", lookahead):
            continue
        positions.append(m.start())
    return positions


def _pick_by_proximity(candidates, name_positions, max_gap: int = 1500):
    """Pick candidate (s, e) heading nearest *after* any fund-name mention."""
    best = None
    best_gap = float("inf")
    for np_ in name_positions:
        for s, e in candidates:
            gap = s - np_
            if 0 <= gap < max_gap and gap < best_gap:
                best = (s, e)
                best_gap = gap
    return best


def _slicer_anchor_for_strategy(clean: str, fund_name: str):
    """Multi-fund routing: locate the strategy heading position for the
    target fund using slicer-style verification, with v2's broader
    strategy pattern set as the anchor.

    Returns (heading_start, heading_end) or None.

    Why this exists (vs v2's strict heading detection + proximity):
      - In mega-prospectuses (3+ MB, 30-50 funds), per-fund headings
        often appear inlined with icon labels ("(TELESCOPE ICON)
        INVESTMENT STRATEGY"). `_is_real_heading` correctly rejects
        these as not heading-shaped — but the slicer-style check
        (fund-name match + strategy pattern within 5-1500 chars)
        confirms the position because the fund-name preamble alone
        is strong structural evidence.

    Why this exists (vs the underlying slicer):
      - The slicer's `SECTION_ANCHORS` list is narrower than v2's
        strategy patterns (no "what are the fund's main investment
        strategies", no "our investment strategies", no "principal
        strateg(y|ies)" variant case sensitivity). Using v2's full
        pattern set catches more fund's real headings.

    TOC filter is applied so the JPMorgan-style "Investment
    Strategies ... 64" TOC line gets rejected.
    """
    if not fund_name:
        return None
    tokens = _fund_name_tokens(fund_name)
    pat = _build_fund_name_pattern(tokens)
    if pat is None:
        return None

    candidates = []  # (fund_pos, anchor_start, anchor_end)
    for m in pat.finditer(clean.lower()):
        fund_pos = m.start()
        search_lo = fund_pos + 5
        search_hi = fund_pos + 1500
        for sp, _strict in _EXTRACT_PATTERNS["strategy"]:
            for sm in sp.finditer(clean, search_lo, search_hi):
                anc_start, anc_end = sm.start(), sm.end()
                if _is_toc_line(clean, anc_start, anc_end):
                    continue
                candidates.append((fund_pos, anc_start, anc_end))
                break  # earliest match per pattern per fund position
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    _, anc_start, anc_end = candidates[0]
    return (anc_start, anc_end)


def _list_other_fund_names(content: str, target_name: str) -> list:
    """Return SERIES-NAMEs from the SGML header excluding the target fund.

    Used by `_trim_to_target_fund` to detect when the extracted body
    crosses into a sibling fund's section.
    """
    head = content[:300_000]
    others = []
    target_tokens = set(_fund_name_tokens(target_name)) if target_name else set()
    for m in re.finditer(r"<SERIES-NAME>\s*([^\n<]+?)\s*(?:\n|<)", head):
        name = m.group(1).strip()
        if not name:
            continue
        name_tokens = set(_fund_name_tokens(name))
        if target_tokens and name_tokens == target_tokens:
            continue
        others.append(name)
    return others


def _trim_to_target_fund(body: str, target_name: str, other_names: list) -> str:
    """If the body contains a sibling fund's section header, truncate at it.

    Decision: only truncate when a sibling-fund name appears (a) AFTER the
    target name's first occurrence in the body (so we keep the target's
    section), AND (b) at a paragraph break (preceded by \\n\\n or starting
    a line), so list-context co-mentions like "Each of the Micro-Cap,
    Emerging Growth and Small-Cap Funds primarily invests..." don't
    trigger truncation.
    """
    if not target_name or not other_names:
        return body

    target_tokens = _fund_name_tokens(target_name)
    target_pat = _build_fund_name_pattern(target_tokens)
    if target_pat is None:
        return body

    target_first = None
    target_m = target_pat.search(body.lower())
    if target_m:
        target_first = target_m.start()

    earliest_other = None
    for other in other_names:
        other_tokens = _fund_name_tokens(other)
        if not other_tokens or set(other_tokens) <= set(target_tokens):
            continue
        opat = _build_fund_name_pattern(other_tokens)
        if opat is None:
            continue
        for m in opat.finditer(body.lower()):
            pos = m.start()
            # Must be after target's first mention (keep target section
            # intact)
            if target_first is not None and pos <= target_first + 50:
                continue
            # Must be at a paragraph break (\n\n before, or position 0)
            line_start = body.rfind("\n", 0, pos) + 1
            line_prefix = body[line_start:pos]
            preceded_by_para_break = (
                pos == 0
                or (line_prefix.strip() == "" and pos >= 2
                    and body[max(0, pos - 2):pos] == "\n\n")
            )
            if not preceded_by_para_break:
                continue
            if earliest_other is None or pos < earliest_other:
                earliest_other = pos
            break  # earliest per other-name is enough

    if earliest_other is not None:
        return body[:earliest_other].rstrip()
    return body


def _pick_by_longest_body(candidates, all_starts, doc_end):
    """Pick the heading whose extractable body (up to next heading) is longest.

    Used in single-fund mode when multiple heading candidates remain after
    TOC and prose-wrap filtering. Real body headings are followed by
    substantial prose; surviving TOC entries and cross-references typically
    cut off at the next list item or page-number boundary.
    """
    best = None
    best_len = -1
    for s, e in candidates:
        next_pos = doc_end
        for hs in all_starts:
            if hs > e + 20:
                next_pos = hs
                break
        body_len = min(next_pos - e, 15000)
        if body_len > best_len:
            best_len = body_len
            best = (s, e)
    return best


# ── Top-level extraction ─────────────────────────────────────────────


def extract_html_headings_v2(content: str, fund_name: str = None,
                              num_series: int = None) -> dict:
    """V2 HTML/text extractor.

    Args:
        content: full submission text
        fund_name: PIT fund name (used only for multi-fund proximity)
        num_series: distinct SERIES-IDs in header; if None, counted from
                    content. When 0 or 1, proximity is skipped — the
                    first non-TOC body heading is selected. (Many pre-2010
                    single-fund filings have no SERIES-ID at all.)
    """
    if num_series is None:
        num_series = _count_series_in_header(content)
    # Pre-compute sibling fund names from SGML header for multi-fund
    # trim filter (cheap, only ~300KB header scan).
    other_fund_names = (
        _list_other_fund_names(content, fund_name)
        if fund_name and num_series > 1 else []
    )

    result = {}

    for prospectus, is_xbrl_infra in _v2_prospectus_documents(content):
        clean = strip_tags(prospectus)
        if len(clean) < 200:
            continue

        all_headings = _find_all_headings(clean)
        if not all_headings:
            continue

        cands = {
            sect: [(s, e) for s, e, ss in all_headings if ss == sect]
            for sect in _EXTRACT_PATTERNS
        }
        all_starts = [s for s, _, _ in all_headings]

        name_positions = (
            _find_name_positions(clean, fund_name)
            if fund_name and num_series > 1 else []
        )

        for section in _EXTRACT_PATTERNS:
            if section in result:
                continue
            section_cands = cands.get(section, [])
            if not section_cands:
                continue

            if name_positions and len(section_cands) > 1:
                best = _pick_by_proximity(section_cands, name_positions, max_gap=1500)
                if best is None and section == "strategy" and len(section_cands) <= 5:
                    # Wider second pass ONLY when there are few candidates
                    # (Oberweis-style "of the Domestic Funds" describing 3
                    # funds collectively, headings at ~4-5K from fund name).
                    # JPMorgan-style 30+ candidates use 1500 only — wider
                    # would otherwise emit a sibling fund's heading.
                    best = _pick_by_proximity(section_cands, name_positions, max_gap=5000)
                if best is None and section == "strategy":
                    # Slicer fallback for multi-fund mega-prospectuses.
                    slc = _slicer_anchor_for_strategy(clean, fund_name)
                    if slc is not None:
                        best = slc
                if best is None:
                    # Multi-fund mode: return empty if no confident routing.
                    # Falling back to section_cands[0] or longest-body emits
                    # wrong-fund content (e.g. SmartRetirement Income for a
                    # JPMorgan Growth and Income request, or SAI text for
                    # Oberweis if name proximity simply doesn't match).
                    continue
            elif len(section_cands) > 1:
                # Single-fund or no-name mode: prefer the candidate with the
                # longest extractable body. Older prospectuses often have a
                # TOC entry, a bulleted cross-reference ("• The principal
                # investment strategies of the Fund – how..."), and the
                # actual body heading all matching the regex. The real
                # heading is followed by the longest substantive body before
                # the next section.
                best = _pick_by_longest_body(section_cands, all_starts, len(clean))
            else:
                best = section_cands[0]

            head_end = best[1]
            # Compound heading detection: when the matched heading is
            # "Investment Strategies and X" (e.g. "...and Risks",
            # "...and Policies", "...and Investment Approach"), the
            # prospectus has merged multiple sections under one heading.
            # Body must extend past any subsection whose name was in the
            # compound, otherwise we'd truncate at the first subsection
            # of the same compound — leaving the body starting with TOC
            # fragments like "AND RISKS Advisor LifeStyle...".
            heading_text = clean[best[0]:best[1]]
            compound_words = set()
            if section == "strategy":
                m_tail = re.search(
                    r"strateg(?:y|ies)\b(.*)$",
                    heading_text, re.I | re.DOTALL,
                )
                if m_tail:
                    tail = m_tail.group(1).lower()
                    _STOPWORDS = {
                        "and", "or", "the", "a", "an", "of", "for", "in",
                        "on", "fund", "funds", "with", "principal", "main",
                    }
                    for w in re.findall(r"[a-z]+", tail):
                        if w in _STOPWORDS or len(w) < 3:
                            continue
                        compound_words.add(w)
                        if w.endswith("s"):
                            compound_words.add(w[:-1])
                        else:
                            compound_words.add(w + "s")
            is_compound = bool(compound_words)

            next_pos = len(clean)
            if is_compound:
                # Skip any subsequent heading or terminator whose text
                # contains a compound-tail word — those are subsections
                # of the same compound section. Terminate at the first
                # heading whose text shares no word with the compound
                # tail (typically Fees / Performance / Management).
                for hs, he, _ in all_headings:
                    if hs <= head_end + 20:
                        continue
                    if any(w in clean[hs:he].lower() for w in compound_words):
                        continue
                    next_pos = hs
                    break
                body_end = min(next_pos, head_end + 30000)
            else:
                for s in all_starts:
                    if s > head_end + 20:  # +20 to skip the heading's own line
                        next_pos = s
                        break
                body_end = min(next_pos, head_end + 15000)
            body = clean[head_end:body_end].strip()
            if len(body) < 30:
                continue
            # Apply prose gate when scraping XBRL infrastructure docs
            # (EX-101.INS, R*.htm). These files mix real prose with viewer
            # metadata ("rr_StrategyHeading", "Investment Strategy,
            # Narrative") that _looks_like_prose detects via
            # _XBRL_VIEWER_JUNK. Bodies from the main 485BPOS HTM aren't
            # gated — they may legitimately contain short summaries.
            if is_xbrl_infra and not _looks_like_prose(body):
                continue
            # Multi-fund safety: when the body crosses into a sibling
            # fund's section (different fund name as a paragraph-break
            # header), truncate at that boundary. Collective discussions
            # like "Each of the Micro-Cap, Emerging Growth and Small-Cap
            # Funds primarily invests..." aren't truncated — the trim
            # function requires sibling names to appear at a paragraph
            # break, not in list context.
            if num_series > 1 and fund_name and other_fund_names:
                body = _trim_to_target_fund(body, fund_name, other_fund_names)
                if len(body) < 30:
                    continue
            result[section] = body

        if all(s in result for s in _EXPECTED_SECTIONS):
            break

    return result


def extract_filing_v2(fpath: str, series_id: str, match_type: str) -> dict:
    """V2 per-filing extractor.

    XBRL path is delegated to v1's extract_xbrl (unchanged). When XBRL is
    incomplete or absent, falls back to extract_html_headings_v2.
    """
    try:
        with open(fpath, "r", errors="replace") as f:
            content = f.read()
    except Exception:
        return {}

    xbrl_sections = extract_xbrl(content, series_id) if match_type == "series" else {}
    if xbrl_sections and all(sec in xbrl_sections for sec in _EXPECTED_SECTIONS):
        xbrl_sections["_source"] = "xbrl"
        return xbrl_sections

    fname = None
    fd = _read_filing_date(content)
    if fd:
        fname = _get_pit_fund_name(series_id, fd)

    num_series = _count_series_in_header(content)
    html_sections = extract_html_headings_v2(
        content, fund_name=fname, num_series=num_series
    )

    if not xbrl_sections and not html_sections:
        return {}

    merged = dict(xbrl_sections)
    html_contributed = False
    gate_active = bool(xbrl_sections)
    for sec in _EXPECTED_SECTIONS:
        if sec in merged:
            continue
        cand = html_sections.get(sec)
        if not cand:
            continue
        if gate_active and sec in _PROSE_GATED_SECTIONS and not _looks_like_prose(cand):
            continue
        merged[sec] = cand
        html_contributed = True

    if not merged:
        return {}

    if xbrl_sections and html_contributed:
        merged["_source"] = "xbrl+html"
    elif xbrl_sections:
        merged["_source"] = "xbrl"
    else:
        merged["_source"] = "html"
    return merged
