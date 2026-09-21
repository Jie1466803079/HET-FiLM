"""Reproduce build_canada_graph_pkl.py's `global_funds = sorted(all_funds)` so
that the (fundno -> unified_id) mapping used by the graph can be recovered.

The graph pickle stores `fund.id_idx` per snapshot but does NOT store the
fundno-side of the mapping. Downstream artifacts (e.g. a Canadian text
embedding H5 keyed by unified_id) need this mapping, so we replay Phase A's
pass_hard filter + Phase B's `is_matched_equity` cusip join and collect the
union of fundnos that contribute at least one edge to the graph.

Output: canada_fundno_to_unified_id.pkl.gz
    { 'fundno_to_unified': dict[int, int],
      'global_funds':       list[int]   # sorted; global_funds[i] = fundno of unified_id i
    }
"""
from __future__ import annotations
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SNAP = Path(__file__).resolve().parent
sys.path.insert(0, str(SNAP))

# Reuse the build script's loaders + constants — single source of truth.
from build_canada_graph_pkl import (  # noqa: E402
    load,
    load_holdings_year,
    EQUITY_SHARE_THRESHOLD,
    MIN_HOLDINGS,
    NAME_EXCLUDE,
)


def main():
    print("[1/4] Loading crosswalk + fund metadata...", flush=True)
    xw = load("compustat_na_crosswalk_unified")[["cusip8", "gvkey", "iid"]].drop_duplicates("cusip8")
    fnd = load("canada_funds")[["fundno", "fundname"]].drop_duplicates("fundno")

    print("[2/4] Phase A — per-year (fundno, quarter) aggregations...", flush=True)
    per_year = []
    for yr in range(2015, 2026):
        h = load_holdings_year(yr, xw)
        a = (h.groupby(["fundno", "quarter"], as_index=False)
               .agg(mkt_val_total =("mkt_val",    "sum"),
                    mkt_val_equity=("mkt_val_eq", "sum"),
                    n_holdings    =("weight_norm", "size")))
        per_year.append(a)
        del h
    agg = pd.concat(per_year, ignore_index=True)
    del per_year

    agg["equity_share"] = agg["mkt_val_equity"] / agg["mkt_val_total"].where(agg["mkt_val_total"] > 0)
    agg = agg.merge(fnd, on="fundno", how="left")
    name = agg["fundname"].fillna("").astype(str)
    agg["name_clean"] = ~name.str.contains(NAME_EXCLUDE)
    agg["pass_hard"] = (
        (agg["equity_share"] >= EQUITY_SHARE_THRESHOLD)
        & (agg["n_holdings"]   >= MIN_HOLDINGS)
        & agg["name_clean"]
    )
    pass_keys = agg.loc[agg["pass_hard"], ["fundno", "quarter"]]
    pass_set = set(zip(pass_keys["fundno"].tolist(), pass_keys["quarter"].tolist()))
    print(f"      pass_hard rows: {agg['pass_hard'].sum():,}  passing (fundno,quarter): {len(pass_set):,}", flush=True)

    print("[3/4] Phase B — per-year edge-survival filter (matched_equity ∩ pass_set)...", flush=True)
    all_funds: set[int] = set()
    for yr in range(2015, 2026):
        h = load_holdings_year(yr, xw)
        h["_pass"] = list(zip(h["fundno"].tolist(), h["quarter"].tolist()))
        h = h[h["_pass"].map(lambda k: k in pass_set)]
        h = h[h["is_matched_equity"]]
        all_funds.update(int(x) for x in h["fundno"].unique())
        print(f"      {yr}: cum_unique_fundnos={len(all_funds):,}", flush=True)
        del h

    global_funds = sorted(all_funds)
    fundno_to_unified = {f: i for i, f in enumerate(global_funds)}
    print(f"[4/4] global_funds: {len(global_funds):,}  (should match graph id_idx universe = 3,459)", flush=True)

    out = SNAP / "canada_fundno_to_unified_id.pkl.gz"
    with gzip.open(out, "wb") as f:
        pickle.dump({
            "fundno_to_unified": fundno_to_unified,
            "global_funds":      global_funds,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
