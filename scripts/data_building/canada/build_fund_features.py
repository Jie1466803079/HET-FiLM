"""Build quarterly fund features for the Canadian benchmark.

Mirrors the US 11feat fund-side numerical block. Targets parity in count
(11 features) and economic intuition, but field origins differ because
S12/Thomson does not carry CRSP-MFDB fields like expense_ratio or
turnover_ratio at the fund level. We construct portfolio-aggregate
analogs from the holdings panel using `weight_norm` as the edge weight.

Edge weight choice: `weight_norm = mkt_val / Σ(mkt_val per fundno-fdate)`,
sums to 1.0 per fund-snapshot. Chosen over S12 `percent_tna` because
the latter is fund-reported, averages ~0.66 per-fund sum, and has
positions reported up to 2076% TNA (data quality issue).

Reads:
    canada_holdings_{2015..2025}.pkl.gz
    canada_funds.pkl.gz
    canada_fundno_to_mgmt_id_v1.pkl.gz
    compustat_na_crosswalk_unified.pkl.gz
    compustat_na_features_quarterly.pkl.gz (for currency, country, sector tags
        that augment holdings when indcode is missing — secondary path)

Writes:
    canada_fund_features_quarterly.pkl.gz

Feature block (11 numerical, in addition to id/context cols):
    log_n_holdings    : log(count of equity positions at fdate)
    log_aum_holdings  : log(Σ mkt_val) — equity AUM (local currency mixed)
    herfindahl        : Σ(weight_norm²) over holdings
    top1_weight       : max(weight_norm)
    top10_weight      : sum of top-10 weight_norm
    new_pos_share     : Σ(weight_norm where fresh==True) — newly added positions
    abs_change_share  : Σ|Δweight_norm| vs prior fdate for same fundno
    sector_herfindahl : Σ(sector_w²) where sector_w aggregates weight_norm by indcode
    usd_share         : Σ(weight_norm where curcdm=='USD') — cross-border exposure
    dom_share         : Σ(weight_norm where fic=='CAN') — home-country bias
    fund_age_years    : (fdate - earliest report) in years, capped at 30

Side columns preserved for downstream gating / inspection:
    fundno, fdate, quarter, mgmt_id, fundname, mgrcoab, ioc, country,
    n_holdings (raw), aum_holdings (raw),
    is_sparse_fund   : n_holdings < 10  (loader-side filter switch)
    is_single_holding: n_holdings == 1  (finer-grained variant)
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


def _top_k_weight(w: pd.Series, k: int) -> float:
    s = w.dropna().sort_values(ascending=False).head(k).sum()
    return float(s) if pd.notna(s) else np.nan


def main():
    # ---- 1. Load + concat holdings 2015-2025 ----
    print("Loading holdings 2015-2025...")
    parts = [load(f"canada_holdings_{yr}") for yr in range(2015, 2026)]
    h = pd.concat(parts, ignore_index=True)
    h["fdate"] = pd.to_datetime(h["fdate"])
    print(f"  holdings rows: {len(h):,}  unique fundno: {h['fundno'].nunique():,}")

    # Drop rows with zero/missing weight or mkt_val — they distort aggregates
    h = h[h["weight_norm"].notna() & (h["weight_norm"] > 0)].copy()
    print(f"  after weight_norm > 0 filter: {len(h):,}")

    # ---- 2. Join stock-side currency/country tags from crosswalk + features ----
    # Crosswalk gives currency-of-incorporation context (excntry).
    # Features file gives end-of-quarter curcdm and fic (preferred when present).
    xw = load("compustat_na_crosswalk_unified")[
        ["cusip8", "gvkey", "iid", "excntry"]
    ].drop_duplicates("cusip8")
    h["cusip8"] = h["cusip"].astype(str).str.upper().str[:8]
    h = h.merge(xw, on="cusip8", how="left")

    # Quarter alignment
    h["quarter"] = h["fdate"].dt.to_period("Q").dt.to_timestamp("D", how="end").dt.normalize()

    stk = load("compustat_na_features_quarterly")[
        ["gvkey", "iid", "quarter", "curcdm", "fic"]
    ]
    h = h.merge(stk, on=["gvkey", "iid", "quarter"], how="left")
    print(f"  after stock join: curcdm present {100*h['curcdm'].notna().mean():.1f}%  "
          f"fic present {100*h['fic'].notna().mean():.1f}%")

    # ---- 3. Per (fundno, fdate) base aggregates ----
    # Use weight_norm throughout for portfolio-share derived features.
    print("Aggregating per (fundno, fdate)...")
    g = h.groupby(["fundno", "fdate"], sort=False)

    base = g.agg(
        n_holdings       =("weight_norm", "size"),
        aum_holdings     =("mkt_val", "sum"),
        herfindahl       =("weight_norm", lambda w: float((w**2).sum())),
        top1_weight      =("weight_norm", "max"),
        new_pos_share    =("weight_norm", lambda w: float(
            w[h.loc[w.index, "fresh"].fillna(False)].sum())),
    ).reset_index()

    # top10 needs a separate pass (sort + head)
    top10 = g["weight_norm"].apply(lambda w: _top_k_weight(w, 10)).reset_index(name="top10_weight")
    base = base.merge(top10, on=["fundno", "fdate"], how="left")

    # ---- 4. Sector concentration (using indcode from S12) ----
    sec_w = (h.groupby(["fundno", "fdate", "indcode"])["weight_norm"].sum()
               .reset_index(name="sector_w"))
    sec_hhi = (sec_w.groupby(["fundno", "fdate"])["sector_w"]
                    .apply(lambda s: float((s**2).sum()))
                    .reset_index(name="sector_herfindahl"))
    base = base.merge(sec_hhi, on=["fundno", "fdate"], how="left")

    # ---- 5. Currency / country shares ----
    # Coalesce: prefer features.curcdm; fall back to excntry from crosswalk
    h["ccy"] = h["curcdm"].fillna(h["excntry"])
    h["cty"] = h["fic"].fillna(h["excntry"])

    ccy_agg = (h.assign(
                  is_usd=(h["ccy"] == "USD").astype(float) * h["weight_norm"],
                  is_can=(h["cty"] == "CAN").astype(float) * h["weight_norm"])
                 .groupby(["fundno", "fdate"])[["is_usd", "is_can"]].sum()
                 .reset_index()
                 .rename(columns={"is_usd": "usd_share", "is_can": "dom_share"}))
    base = base.merge(ccy_agg, on=["fundno", "fdate"], how="left")

    # ---- 6. abs_change_share — turnover proxy via prior-period diff ----
    # For each fund, compute |Δweight_norm| summed across the holding union of
    # consecutive snapshots. Missing in current = sold (delta = -w_prev),
    # missing in prior   = bought (delta =  w_cur). Symmetric.
    print("Computing abs_change_share (per-fund prior-period diff)...")
    pairs = (h[["fundno", "fdate", "cusip8", "weight_norm"]]
               .sort_values(["fundno", "fdate"]))

    fund_dates = (pairs[["fundno", "fdate"]]
                    .drop_duplicates().sort_values(["fundno", "fdate"]))
    fund_dates["prev_fdate"] = fund_dates.groupby("fundno")["fdate"].shift(1)

    cur = pairs.merge(fund_dates, on=["fundno", "fdate"], how="left")
    prev = pairs.rename(columns={"fdate": "prev_fdate", "weight_norm": "w_prev"})
    merged = cur.merge(prev, on=["fundno", "prev_fdate", "cusip8"], how="outer")
    merged["weight_norm"] = merged["weight_norm"].fillna(0.0)
    merged["w_prev"]      = merged["w_prev"].fillna(0.0)
    merged["abs_d"]       = (merged["weight_norm"] - merged["w_prev"]).abs()
    # When current snapshot is missing (no fdate), the row is a stale-sold
    # position belonging to prev_fdate->current transition — back-fill fdate
    # from the fund_dates table using prev_fdate.
    back = fund_dates.rename(columns={"fdate": "cur_fdate"})[["fundno", "prev_fdate", "cur_fdate"]]
    merged = merged.merge(back, on=["fundno", "prev_fdate"], how="left")
    merged["fdate"] = merged["fdate"].fillna(merged["cur_fdate"])

    turn = (merged.dropna(subset=["fdate"])
                  .groupby(["fundno", "fdate"])["abs_d"].sum()
                  .reset_index(name="abs_change_share"))
    base = base.merge(turn, on=["fundno", "fdate"], how="left")
    # Funds without a prior snapshot get NaN -> treat as 0 turnover signal
    base["abs_change_share"] = base["abs_change_share"].fillna(0.0)

    # ---- 7. Fund metadata + age ----
    funds = load("canada_funds")
    funds["rdate1"] = pd.to_datetime(funds["rdate1"])
    base = base.merge(
        funds[["fundno", "fundname", "mgrcoab", "ioc", "country", "rdate1"]],
        on="fundno", how="left"
    )
    base["fund_age_years"] = ((base["fdate"] - base["rdate1"]).dt.days / 365.25)
    base["fund_age_years"] = base["fund_age_years"].clip(lower=0, upper=30)

    # ---- 8. Logs + final feature block ----
    base["log_n_holdings"]   = np.log(base["n_holdings"].clip(lower=1))
    base["log_aum_holdings"] = np.log(base["aum_holdings"].where(base["aum_holdings"] > 0))

    # ---- 9. mgmt_id (89.7% coverage) ----
    mgmt = load("canada_fundno_to_mgmt_id_v1")[
        ["fundno", "mgmt_id", "mgmt_name", "confidence"]
    ]
    base = base.merge(mgmt, on="fundno", how="left")

    # ---- 10. Quarter snapshot column for downstream join with stock features ----
    base["quarter"] = base["fdate"].dt.to_period("Q").dt.to_timestamp("D", how="end").dt.normalize()

    # ---- 10b. Sparse-fund flags (loader-side filter switches, not features) ----
    # Two thresholds so downstream code can pick: strict (n_holdings == 1, ~9%)
    # filters obvious placeholders, broad (n_holdings < 10, ~16%) also removes
    # very-small portfolios where Herfindahl / top-K weight features hit the
    # ceiling and lose discriminative power.
    base["is_single_holding"] = (base["n_holdings"] == 1)
    base["is_sparse_fund"]    = (base["n_holdings"] < 10)

    # ---- 11. Final column ordering ----
    feature_cols = [
        "log_n_holdings", "log_aum_holdings",
        "herfindahl", "top1_weight", "top10_weight",
        "new_pos_share", "abs_change_share",
        "sector_herfindahl",
        "usd_share", "dom_share",
        "fund_age_years",
    ]
    assert len(feature_cols) == 11, f"feature block must be 11, got {len(feature_cols)}"

    ctx_cols = [
        "fundno", "fdate", "quarter",
        "mgmt_id", "mgmt_name", "confidence",
        "fundname", "mgrcoab", "ioc", "country",
        "n_holdings", "aum_holdings",
        "is_single_holding", "is_sparse_fund",
    ]
    out = base[ctx_cols + feature_cols].sort_values(["fundno", "fdate"]).reset_index(drop=True)

    # ---- 12. Sanity output ----
    print()
    print(f"Fund-quarter rows: {len(out):,}")
    print(f"Unique fundno:     {out['fundno'].nunique():,}")
    print(f"Date range:        {out['fdate'].min().date()} -> {out['fdate'].max().date()}")
    print()
    print("Feature fill rates:")
    for c in feature_cols:
        nn = 100 * out[c].notna().mean()
        print(f"  {c:<22} {nn:>5.1f}%")
    print()
    print("Feature distributions (5/50/95 percentiles):")
    desc = out[feature_cols].describe(percentiles=[0.05, 0.5, 0.95])
    print(desc.T[["mean", "std", "5%", "50%", "95%"]].to_string())
    print()
    print(f"mgmt_id coverage: {100*out['mgmt_id'].notna().mean():.1f}%")
    print(f"usd_share > 0:    {100*(out['usd_share']>0).mean():.1f}% of fund-quarters")
    print(f"dom_share > 0:    {100*(out['dom_share']>0).mean():.1f}% of fund-quarters")
    print()
    print("Sparse-fund flags:")
    print(f"  is_single_holding (n=1):  {100*out['is_single_holding'].mean():>5.1f}%  ({int(out['is_single_holding'].sum()):,} fund-quarters)")
    print(f"  is_sparse_fund    (n<10): {100*out['is_sparse_fund'].mean():>5.1f}%  ({int(out['is_sparse_fund'].sum()):,} fund-quarters)")
    print(f"  multi-holding (n>=10):    {100*(~out['is_sparse_fund']).mean():>5.1f}%  ({int((~out['is_sparse_fund']).sum()):,} fund-quarters)")

    p = save(out, "canada_fund_features_quarterly")
    print(f"\nSaved -> {p.name}  ({p.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
