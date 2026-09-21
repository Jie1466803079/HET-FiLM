"""Pure functions for rebuilding the prospectus H5 with report_dt anchor.

Kept separate from the CLI script so they can be unit-tested without h5py
fixtures. See docs/plans/2026-05-20-prospectus-report-anchor-switch-design.md.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

MAX_LAG = 6   # quarters; matches encode_prospectus_openai.py


def build_target_q_vector(
    explicit: Dict[str, str],
    quarters: List[str],
    quarter_to_idx: Dict[str, int],
) -> np.ndarray:
    """Return length-N int64 vector of target-quarter indices.

    For each eff_q in `quarters`:
      - if eff_q in `explicit`: use quarter_to_idx[explicit[eff_q]]
      - else: carry forward the last known target_q_idx
      - else (no prior known): fall back to eff_q's own index

    Args:
        explicit: dict mapping eff_q string -> report_q string
                  (real-world: this fund's rows from fund_effq_to_reportq.csv)
        quarters: H5 snapshot_quarters as a list of "YYYYQN" strings
        quarter_to_idx: position lookup for `quarters`

    Returns:
        ndarray of shape (len(quarters),), dtype int64
    """
    n = len(quarters)
    out = np.empty(n, dtype=np.int64)
    last_known: Optional[int] = None
    for i, eff_q in enumerate(quarters):
        if eff_q in explicit:
            tq = quarter_to_idx[explicit[eff_q]]
            out[i] = tq
            last_known = tq
        elif last_known is not None:
            out[i] = last_known
        else:
            out[i] = i   # final fallback: eff_q itself
    return out


def reconstruct_filings(
    strategy_row: np.ndarray,   # (N_QUARTERS, EMB_DIM)
    risk_row: np.ndarray,       # (N_QUARTERS, EMB_DIM)
    delta_t_row: np.ndarray,    # (N_QUARTERS,) — int, -1 means cold-start
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Recover per-source-quarter filing embeddings from a single fund row.

    Each non-cold-start cell at index qi holds the embedding of a filing
    posted in quarter src_q = qi - delta_t[qi]. The same src_q can appear
    in multiple LOCF'd cells; we record it once (the values are identical
    by construction of the original LOCF).
    """
    filings: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    n = delta_t_row.shape[0]
    for qi in range(n):
        d = int(delta_t_row[qi])
        if d == -1:
            continue
        src = qi - d
        if src in filings:
            continue
        filings[src] = (strategy_row[qi].copy(), risk_row[qi].copy())
    return filings


def re_locf_anchored(
    filings: Dict[int, Tuple[np.ndarray, np.ndarray]],
    target_q: np.ndarray,   # (N,) int64 — target quarter index per eff_qi
    N: int,
    emb_dim: int,
    max_lag: int = MAX_LAG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply LOCF over `filings.keys()` anchored at target_q[eff_qi].

    Returns (strategy, risk, delta_t, has_text) all of shape (N, ...).
    delta_t is int8 to match encoder output; has_text is bool.
    """
    new_strat = np.zeros((N, emb_dim), dtype=np.float32)
    new_risk  = np.zeros((N, emb_dim), dtype=np.float32)
    new_dt    = np.full(N, -1, dtype=np.int8)
    new_ht    = np.zeros(N, dtype=bool)

    if not filings:
        return new_strat, new_risk, new_dt, new_ht

    sorted_srcs = sorted(filings.keys())
    for eff_qi in range(N):
        tq = int(target_q[eff_qi])
        # Pick the largest src ≤ tq with tq - src ≤ max_lag.
        best: Optional[int] = None
        for src in sorted_srcs:
            if src > tq:
                break
            if tq - src <= max_lag:
                best = src
        if best is None:
            continue
        s_emb, r_emb = filings[best]
        new_strat[eff_qi] = s_emb
        new_risk[eff_qi]  = r_emb
        new_dt[eff_qi]    = eff_qi - best
        new_ht[eff_qi]    = (best == eff_qi)
    return new_strat, new_risk, new_dt, new_ht


def apply_anchor_policy(
    target_q_carry: np.ndarray,    # (N,) int64 from build_target_q_vector
    explicit_mask: np.ndarray,     # (N,) bool — True where this cell has an explicit lookup
    policy: str,                   # "strict" or "carry"
) -> np.ndarray:
    """Return the final target_q vector to feed into re_locf_anchored.

    strict: cells where explicit_mask is False are forced to -1 (cold-start).
    carry:  target_q_carry is passed through unchanged (non-explicit cells
            keep the carry-forward target from the most-recent explicit anchor).
    """
    if policy == "strict":
        return np.where(explicit_mask, target_q_carry, -1).astype(np.int64)
    if policy == "carry":
        return target_q_carry.astype(np.int64)
    raise ValueError(f"unknown anchor_policy={policy!r}; expected 'strict' or 'carry'")


def resolve_max_lag(max_lag_arg: int, n_q: int) -> int:
    """Translate the user-facing --max_lag value to the value passed to
    re_locf_anchored. Any non-positive input means "unbounded": we return n_q,
    which is strictly larger than any possible (tq - src) gap inside the
    snapshot window, so the cap never binds."""
    if max_lag_arg <= 0:
        return n_q
    return max_lag_arg
