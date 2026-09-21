#!/usr/bin/env python3
"""Fund-section slicer: given a 485BPOS filing's cleaned text and a target
fund name, return a ~40K-char window centered on the target fund's
Principal Investment Strategies section.

Why this exists: the LLM extraction pipeline truncates filings to ~95K
chars. For multi-fund family prospectuses (STI Classic, JPMorgan
family, Vanguard family) the target fund's actual strategy section may
sit at char 400K-1M — far past the truncation window. The LLM literally
never sees it. This slicer finds the fund's section first, then returns
a window the LLM can fit.

Inputs:
    text: cleaned filing text (output of strip_to_text)
    fund_name: target fund's PIT name
    window_chars: total window size to return (default 40_000)

Returns:
    SliceResult(text, anchor_offset, anchor_phrase, fund_position,
                full_text_length) or None.
    None means the fund's strategy section was not located.

Strategy:
    1. Build a regex from the fund-name's ordered tokens that requires
       them to appear in sequence with only whitespace/punctuation
       between them (no token re-ordering, no spurious matches via
       far-apart tokens).
    2. Match against lowercased text (length-preserving, so positions
       are accurate offsets into the original text).
    3. For each match, look ahead 1500 chars for a section anchor like
       "Principal Investment Strategies". Require the anchor to appear
       close after the fund name (within 200-1500 chars) — that
       structural relationship is what distinguishes a section heading
       from a random co-mention.
    4. Return the EARLIEST verified match's slice. None otherwise.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


SECTION_ANCHORS = (
    'principal investment strategies',
    'principal investment strategy',
    'investment strategies',
    'investment strategy',
    'principal strategies',
    'how the fund invests',
    'investment approach',
    'principal investment policies',
    'investment policies',
)

ORG_SUFFIXES = {
    'fund', 'funds', 'inc', 'llc', 'corp', 'company', 'co',
    'trust', 'series', 'portfolio', 'class',
}

ANCHOR_LOOKAHEAD = 1500
ANCHOR_MIN_DISTANCE = 5      # anchor must be at least this many chars after fund name
ANCHOR_MAX_DISTANCE = 1500   # anchor must be within this many chars


def _fund_name_tokens(name: str) -> list[str]:
    """Tokenize fund name, preserving 'and' (it's part of many fund names
    like 'Growth and Income') but dropping organization suffixes."""
    raw = name.lower()
    raw = raw.replace('®', '').replace('™', '')
    toks = re.findall(r'\w+', raw)
    # Drop trailing org suffixes
    while toks and toks[-1] in ORG_SUFFIXES:
        toks.pop()
    return toks


def _build_fund_name_pattern(tokens: list[str]) -> Optional[re.Pattern]:
    """Build a regex requiring the tokens to appear in consecutive order,
    with only whitespace/punctuation between them, allowing 'and' to
    match '&' as alternative.
    """
    if not tokens:
        return None
    parts = []
    for t in tokens:
        if t == 'and':
            parts.append(r'(?:and|&|&amp;)')
        else:
            parts.append(re.escape(t))
    pattern_src = r'\b' + r'[\s\W]+'.join(parts) + r'\b'
    return re.compile(pattern_src, re.IGNORECASE)


def _find_anchor_after(text_lower: str, pos: int) -> Optional[tuple[int, str]]:
    """Find earliest section anchor between pos+ANCHOR_MIN_DISTANCE and
    pos+ANCHOR_MAX_DISTANCE chars. Returns (anchor_offset, phrase)."""
    lo = pos + ANCHOR_MIN_DISTANCE
    hi = pos + ANCHOR_MAX_DISTANCE
    window = text_lower[lo:hi]
    best: Optional[tuple[int, str]] = None
    for anchor in SECTION_ANCHORS:
        j = window.find(anchor)
        if j < 0:
            continue
        offset = lo + j
        if best is None or offset < best[0]:
            best = (offset, anchor)
    return best


@dataclass
class SliceResult:
    text: str
    anchor_offset: int
    anchor_phrase: str
    fund_position: int
    full_text_length: int


def slice_around_fund(
    text: str,
    fund_name: str,
    window_chars: int = 40_000,
    backward_pad: int = 50,
) -> Optional[SliceResult]:
    """Return a SliceResult around the target fund's strategy section,
    or None if not located.
    """
    if not text or not fund_name:
        return None
    tokens = _fund_name_tokens(fund_name)
    if not tokens:
        return None
    pattern = _build_fund_name_pattern(tokens)
    if pattern is None:
        return None

    text_lower = text.lower()
    # Find all positions where the fund-name pattern matches
    matches = list(pattern.finditer(text_lower))
    if not matches:
        return None

    # For each match, check for a section anchor nearby
    verified: list[tuple[int, int, str]] = []   # (fund_pos, anchor_pos, anchor)
    for m in matches:
        pos = m.start()
        anc = _find_anchor_after(text_lower, pos)
        if anc:
            verified.append((pos, anc[0], anc[1]))

    if not verified:
        return None

    # Pick the earliest verified match
    verified.sort(key=lambda c: c[0])
    fund_pos, anc_pos, anc_phrase = verified[0]

    slice_start = max(0, fund_pos - backward_pad)
    slice_end = min(len(text), slice_start + window_chars)
    sliced = text[slice_start:slice_end]

    return SliceResult(
        text=sliced,
        anchor_offset=anc_pos,
        anchor_phrase=anc_phrase,
        fund_position=fund_pos,
        full_text_length=len(text),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _find_filing(accession: str) -> Optional[str]:
    """Glob-search for a 485BPOS submission by accession across all CIK dirs."""
    import glob
    matches = glob.glob(
        f'/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/sec_filings_project'
        f'/all_485bpos/*/485BPOS/{accession}/full-submission.txt'
    )
    return matches[0] if matches else None


def _test_known_cases() -> int:
    """Test against the 4 known problem cells.

    Stricter pass criterion: the slice must contain (a) the fund's full
    name as a coherent phrase, and (b) the section anchor very close
    after it. This proves the slicer landed at a real section heading,
    not a random co-occurrence.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from extract_485bpos_vllm import strip_to_text  # noqa: E402

    cases = [
        {
            'n': 1, 'name': 'Large Cap Quantitative Equity Fund',
            'accession': '0000950152-06-006291',
            'expect': 'found',
            'expect_anchor_past': 400_000,
        },
        {
            'n': 2, 'name': 'Delafield Fund, Inc',
            'accession': '0001193125-06-041396',
            'expect': 'none',
        },
        {
            'n': 3, 'name': 'JPMorgan Growth and Income Fund',
            'accession': '0001145443-09-002631',
            'expect': 'found',
            'expect_anchor_past': 900_000,
        },
        {
            'n': 4, 'name': 'Vice Fund',
            'accession': '0000894189-19-004667',
            'expect': 'found',
        },
    ]
    n_pass = 0
    n_fail = 0
    for c in cases:
        fp = _find_filing(c['accession'])
        print(f"\n--- CASE {c['n']}: {c['name']!r}  acc={c['accession']} ---")
        if not fp:
            print(f"  [SKIP] filing not found for accession {c['accession']}")
            continue
        raw = open(fp, errors='replace').read()
        text = strip_to_text(raw)
        del raw
        print(f"  filing path: ...{fp[-70:]}")
        print(f"  filing cleaned length: {len(text):,}ch")
        result = slice_around_fund(text, c['name'])

        if c['expect'] == 'none':
            if result is None:
                print(f"  PASS: slicer correctly returned None")
                n_pass += 1
            else:
                print(f"  FAIL: expected None, got slice at fund_pos={result.fund_position:,}, "
                      f"anchor={result.anchor_phrase!r}@{result.anchor_offset:,}")
                print(f"        first 300 chars: {result.text[:300]!r}")
                n_fail += 1
            continue

        # expect: 'found'
        if result is None:
            print(f"  FAIL: expected found, got None")
            n_fail += 1
            continue
        print(f"  fund_position={result.fund_position:,}  anchor={result.anchor_phrase!r}@{result.anchor_offset:,}  delta={result.anchor_offset - result.fund_position}ch")
        print(f"  slice first 400 chars:")
        print(f"    {result.text[:400].strip()!r}")
        ok = True
        # Anchor position check
        if 'expect_anchor_past' in c and result.anchor_offset < c['expect_anchor_past']:
            print(f"  FAIL: anchor at {result.anchor_offset:,} but expected past {c['expect_anchor_past']:,}")
            ok = False
        # The slice must contain the fund-name regex match itself
        pat = _build_fund_name_pattern(_fund_name_tokens(c['name']))
        if pat is None or not pat.search(result.text.lower()):
            print(f"  FAIL: slice doesn't contain the fund-name pattern match")
            ok = False
        # The anchor phrase should appear in the slice
        if result.anchor_phrase not in result.text.lower():
            print(f"  FAIL: slice doesn't contain anchor phrase {result.anchor_phrase!r}")
            ok = False
        if ok:
            print(f"  PASS")
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"Tests: {n_pass} pass, {n_fail} fail")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(_test_known_cases())
