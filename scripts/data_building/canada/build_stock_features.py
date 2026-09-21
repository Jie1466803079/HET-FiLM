"""Build quarterly stock features for the Canadian benchmark.

Mirrors the US pipeline in CRSP_code/Finding_stock_features_0629.py — pulls
monthly observations, aggregates to quarterly snapshots, computes derived
features. Reads compustat_na_secm_{2014..2025}.pkl.gz from the snapshot dir,
writes compustat_na_features_quarterly.pkl.gz.

Field unit conventions (verified against Compustat NA documentation):
    trt1m  : monthly total return, in PERCENT (e.g. 2.4789 = +2.48%)
    prccm  : end-of-month close, local currency (curcdm)
    cshom  : shares outstanding (raw count)
    cshtrm : monthly trading volume (raw share count)
    prchm  : month high
    prclm  : month low
    dvpspm : monthly dividend per share (cash)

US-equivalent feature → Canadian field map:
    ret  (CRSP)        → trt1m   (compounded across quarter)
    prc                → prccm   (EoQ snapshot)
    shrout             → cshom   (EoQ snapshot)
    vol                → cshtrm  (quarter sum)
    askhi/bidlo        → prchm/prclm  (proxy spread; no daily ask/bid in secm)
    vwretd (mkt ret)   → not available in secm; would need an index pull

Currency note: features stay in local currency (curcdm). A categorical
`curcdm` column is preserved per row so downstream code can FX-normalize
market cap if needed.
"""
from __future__ import annotations
import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SNAP = Path("/srv/scratch/dbgcse/jieliu/canada_funds_snapshot_20260531")


def load(name: str):
    with gzip.open(SNAP / f"{name}.pkl.gz", "rb") as f:
        return pickle.load(f)


def save(obj, name: str):
    p = SNAP / f"{name}.pkl.gz"
    with gzip.open(p, "wb", compresslevel=5) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def main():
    # ---- 1. Load + concat 12 years of monthly Compustat NA observations ----
    print("Loading monthly secm 2014-2025...")
    parts = [load(f"compustat_na_secm_{yr}") for yr in range(2014, 2026)]
    m = pd.concat(parts, ignore_index=True)
    m["datadate"] = pd.to_datetime(m["datadate"])
    print(f"  monthly rows: {len(m):,}  unique (gvkey,iid): "
          f"{m[['gvkey','iid']].drop_duplicates().shape[0]:,}")

    # Convert numeric columns once
    for c in ("prccm", "prchm", "prclm", "cshtrm", "cshom",
              "trt1m", "trfm", "ajexm", "ajpm", "dvpspm", "dvpsxm"):
        m[c] = pd.to_numeric(m[c], errors="coerce")

    # ---- 2. End-of-quarter assignment (matches CRSP code convention) ----
    m["quarter"] = m["datadate"].dt.to_period("Q").dt.to_timestamp("D", how="end").dt.normalize()

    grp = ["gvkey", "iid", "quarter"]

    # ---- 3a. Compound quarterly total return from monthly trt1m ----
    # trt1m is in percent (2.48 = +2.48 %), so divide by 100 before chaining.
    def chain(s: pd.Series) -> float:
        s = s.dropna()
        if s.empty:
            return np.nan
        return float((1.0 + s.values / 100.0).prod() - 1.0)
    ret_q = m.groupby(grp)["trt1m"].apply(chain).reset_index(name="ret_q")

    # ---- 3b. Quarter sum of monthly volume ----
    vol_q = m.groupby(grp)["cshtrm"].sum(min_count=1).reset_index(name="vol_q")

    # ---- 3c. EoQ snapshot of price, shares-out, dividend, currency, exchange ----
    snap = (m.sort_values(["gvkey", "iid", "datadate"])
              .groupby(grp).last().reset_index()
              [["gvkey", "iid", "quarter",
                "datadate", "prccm", "prchm", "prclm",
                "cshom", "dvpspm", "dvrate",
                "curcdm", "fic", "exchg", "secstat", "mkvalincl", "tic", "conm"]])

    # ---- 3d. High-low spread proxy (mean over the 3 months in quarter) ----
    m["hl_spread"] = ((m["prchm"] - m["prclm"]) /
                     (((m["prchm"] + m["prclm"]) / 2.0).replace(0, np.nan)))
    spread_q = m.groupby(grp)["hl_spread"].mean().reset_index(name="spread_q")

    # ---- 4. Merge ----
    q = (snap
         .merge(ret_q, on=grp, how="left")
         .merge(vol_q, on=grp, how="left")
         .merge(spread_q, on=grp, how="left"))

    # ---- 5. Derived market-based features ----
    q["mktcap_q"] = q["prccm"].abs() * q["cshom"]
    q["lme_q"]    = np.log(q["mktcap_q"].where(q["mktcap_q"] > 0))
    q["turn_q"]   = q["vol_q"] / q["cshom"].where(q["cshom"] > 0)
    q["divy_q"]   = (q["dvpspm"] * 12.0) / q["prccm"].where(q["prccm"] > 0)  # annualized dividend yield

    # ---- 6. Trailing-12-month features (need ≥12 prior months in panel) ----
    m_sorted = m.sort_values(["gvkey", "iid", "datadate"])
    # 12-month momentum: skip current month (J-T style: ret over t-12..t-2)
    # Implemented at quarter grain using monthly returns.
    def t12_features(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("datadate")
        r = g["trt1m"].values / 100.0
        dt = g["datadate"].values
        out = []
        for i, d in enumerate(dt):
            window_start = i - 12 + 1   # 12-month window ending at month t
            if window_start < 0:
                out.append((d, np.nan, np.nan))
                continue
            window = r[window_start:i+1]
            window = window[~np.isnan(window)]
            if len(window) < 6:        # need at least 6 of 12 months
                out.append((d, np.nan, np.nan))
                continue
            mom12 = float((1.0 + window[:-1]).prod() - 1.0) if len(window) >= 2 else np.nan  # skip last month
            vol12 = float(np.std(window, ddof=1)) if len(window) >= 2 else np.nan
            out.append((d, mom12, vol12))
        return pd.DataFrame(out, columns=["datadate", "mom12_m", "vol12_m"])

    t12 = (m_sorted.groupby(["gvkey", "iid"], group_keys=True)
                   .apply(t12_features)
                   .reset_index()[["gvkey", "iid", "datadate", "mom12_m", "vol12_m"]])
    t12["quarter"] = t12["datadate"].dt.to_period("Q").dt.to_timestamp("D", how="end").dt.normalize()
    # take the t12 row aligned to end-of-quarter month
    t12_eoq = (t12.sort_values(["gvkey", "iid", "datadate"])
                  .groupby(grp).last().reset_index()
                  [["gvkey", "iid", "quarter", "mom12_m", "vol12_m"]])
    q = q.merge(t12_eoq, on=grp, how="left")

    # ---- 7. Robustness columns (raw values unchanged) ----
    # Winsorized variants: clip at 1st / 99th percentile so downstream training can
    # opt-in to a tail-resilient version without the script having mutated raw data.
    for src, dst in [("ret_q", "ret_q_w99"),
                     ("mom12_m", "mom12_m_w99"),
                     ("vol12_m", "vol12_m_w99")]:
        lo, hi = q[src].quantile([0.01, 0.99])
        q[dst] = q[src].clip(lower=lo, upper=hi)
    # Microcap flag: penny stock heuristic (currency-naive but transparent).
    # Two complementary flags so downstream code can pick its filter.
    q["is_pennystock"] = (q["prccm"].abs() < 5.0)                # local-currency price < 5
    q["is_microcap"]   = (q["mktcap_q"] < 50_000_000)            # local-currency market cap < 50 M

    # ---- 8. Final column ordering + sanity ----
    cols = ["gvkey", "iid", "quarter", "datadate",
            "conm", "tic",
            "prccm", "cshom", "vol_q",
            "ret_q", "ret_q_w99",
            "mom12_m", "mom12_m_w99",
            "vol12_m", "vol12_m_w99",
            "mktcap_q", "lme_q", "turn_q",
            "spread_q", "divy_q",
            "dvpspm", "dvrate",
            "is_pennystock", "is_microcap",
            "curcdm", "fic", "exchg", "secstat", "mkvalincl"]
    q = q[cols].sort_values(["gvkey", "iid", "quarter"]).reset_index(drop=True)

    # Sanity output
    print()
    print(f"Quarterly feature rows: {len(q):,}")
    print(f"Unique (gvkey,iid): {q[['gvkey','iid']].drop_duplicates().shape[0]:,}")
    print(f"Quarter range: {q['quarter'].min()} → {q['quarter'].max()}")
    print()
    print("Field fill rates:")
    for c in ["ret_q","ret_q_w99","mktcap_q","lme_q","turn_q","spread_q",
              "mom12_m","mom12_m_w99","vol12_m","divy_q","curcdm","fic"]:
        print(f"  {c:<15} {100*q[c].notna().mean():>5.1f}%")
    print()
    print("ret_q raw vs winsorized (1st/99th percentile clip):")
    desc = q[["ret_q","ret_q_w99"]].describe(percentiles=[0.01,0.05,0.5,0.95,0.99])
    print(desc.to_string())
    print()
    print("Microcap filter coverage:")
    print(f"  is_pennystock (prccm < 5):   {100*q['is_pennystock'].mean():>5.1f}%")
    print(f"  is_microcap   (mktcap < 50M): {100*q['is_microcap'].mean():>5.1f}%")
    print(f"  either flag set:              {100*(q['is_pennystock']|q['is_microcap']).mean():>5.1f}%")
    print()
    print("Currency mix:")
    print(q["curcdm"].value_counts().head(6).to_string())

    p = save(q, "compustat_na_features_quarterly")
    print(f"\nSaved -> {p.name}  ({p.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
