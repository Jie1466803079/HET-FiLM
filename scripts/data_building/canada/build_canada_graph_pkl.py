"""Build Canadian benchmark graph pickle in US-compatible format.

Output: canada_graph_data.pkl  — dict[pd.Timestamp(quarter_end), HeteroData]

Schema matches mutual_fund_prediction/data/graph_data.pkl so the existing
FundsEdgeWeightDataset works via:
    FUNDS_GRAPH_PKL=<this>/canada_graph_data.pkl

USAGE NOTE — leak-free training:
    Stock-feature x in this pickle is pre-z-scored with PANEL-LEVEL (μ, σ)
    over the full 2014-2025 window — same convention as the US pickle. For
    leak-free Canadian runs set in env:
        SPLIT_SAFE_STOCK_SCALING=1
    The loader's re-z-score on train snapshots exactly cancels the panel
    affine transform (see funds_edge_weight.py L313-316), so SAFE+pickle≡
    SAFE+raw mathematically. Fund features ship raw; SPLIT_SAFE_FEATURE_
    SCALING=1 (default ON) handles fund-side scaling.

Snapshot timestamp policy = T2: snapshot key is fdate.to_period('Q'). For 75%
of S12 rows fdate==rdate, matching US (rdate-based) convention; the 5%
late-filing tail correctly lands in its disclosure quarter rather than its
as-of quarter. mean_disclosure_lag_days per fund-quarter shipped as a side
column for downstream sensitivity analysis.

Memory note: this host has a 4 GiB user-slice cgroup cap. The build is
streamed PER-YEAR — never concatenates the full 11-year holdings panel.
Peak working set ≈ one year of holdings (~400 MB) + accumulating graphs dict.
"""
from __future__ import annotations
import gzip
import pickle
import re
import gc
import resource
import sys
from pathlib import Path

import numpy as np
# numpy 2 pickles reference numpy._core; alias to numpy.core for numpy 1 envs
if not hasattr(np, "_core"):
    import numpy.core
    sys.modules["numpy._core"]                = np.core
    sys.modules["numpy._core.numeric"]        = np.core.numeric
    sys.modules["numpy._core.multiarray"]     = np.core.multiarray
    sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath
import pandas as pd
import torch
from torch_geometric.data import HeteroData

SNAP = Path("/srv/scratch/dbgcse/jieliu/canada_funds_snapshot_20260531")

# ---- H3 hard-filter thresholds ----
EQUITY_SHARE_THRESHOLD = 0.50
MIN_HOLDINGS           = 2
NAME_EXCLUDE = re.compile(
    r"INDEX|ETF|PASSIVE|BOND|FIXED INCOME|MORTGAGE|MONEY MARKET|CASH|GIC|TREASURY|MUNICIPAL",
    re.IGNORECASE,
)

# ---- side-flag thresholds ----
STRICT_EQUITY_THRESHOLD   = 0.80
DIVERSIFIED_MIN           = 15
DIVERSIFIED_MAX           = 500
ACTIVE_TURNOVER_THRESHOLD = 0.05
CONCENTRATED_THRESHOLD    = 0.80
INDEX_NAME_REGEX    = re.compile(r"INDEX|ETF|PASSIVE", re.IGNORECASE)
BALANCED_NAME_REGEX = re.compile(
    r"BALANCED|DIVERSIFIED|MULTI[- ]?ASSET|TARGET DATE|LIFECYCLE",
    re.IGNORECASE,
)

# ---- feature column sets ----
FUND_FEATURE_COLS = [
    "log_n_holdings", "log_aum_holdings",
    "herfindahl", "top1_weight", "top10_weight",
    "new_pos_share", "abs_change_share",
    "sector_herfindahl",
    "usd_share", "dom_share",
    "fund_age_years",
]
assert len(FUND_FEATURE_COLS) == 11

STOCK_FEATURE_COLS = [
    "ret_q_w99",
    "mom12_m_w99",
    "vol12_m_w99",
    "lme_q",
    "turn_q",
    "spread_q",
    "divy_q",
    "log_vol_q",
    "log_cshom",
    "log_prccm",
    "dvrate",
]
assert len(STOCK_FEATURE_COLS) == 11

# Cast to float32 to halve memory footprint vs default float64
HOLD_KEEP_COLS = ["fundno", "fdate", "rdate", "cusip", "mkt_val", "weight_norm"]


def load(name: str):
    with gzip.open(SNAP / f"{name}.pkl.gz", "rb") as f:
        return pickle.load(f)


def load_portable(name: str) -> pd.DataFrame:
    """Reconstruct a DataFrame from the dict-of-lists format written by
    /tmp/repickle_portable.py (avoids pandas binary-pickle version skew)."""
    with gzip.open(SNAP / f"{name}_portable.pkl.gz", "rb") as f:
        obj = pickle.load(f)
    out = {}
    for col in obj["_columns"]:
        dtype, vals = obj["_data"][col]
        if dtype == "datetime64[ns]":
            out[col] = pd.to_datetime(np.asarray(vals, dtype="int64"))
        elif dtype == "bool":
            out[col] = np.asarray(vals, dtype=bool)
        elif dtype == "int64":
            out[col] = np.asarray(vals, dtype="int64")
        elif dtype == "float64":
            out[col] = np.asarray(vals, dtype="float64")
        else:
            out[col] = pd.Series(vals, dtype=object)
    return pd.DataFrame(out)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def load_holdings_year(yr: int, xw: pd.DataFrame) -> pd.DataFrame:
    """Load one year, keep only required columns, downcast to save RAM."""
    df = load(f"canada_holdings_{yr}")[HOLD_KEEP_COLS].copy()
    df["fdate"]    = pd.to_datetime(df["fdate"])
    df["rdate"]    = pd.to_datetime(df["rdate"])
    df["cusip8"]   = df["cusip"].astype(str).str.upper().str[:8]
    df = df.drop(columns=["cusip"])
    df["quarter"]  = df["fdate"].dt.to_period("Q").dt.to_timestamp("D", how="end").dt.normalize()
    df["lag_days"] = (df["fdate"] - df["rdate"]).dt.days.astype("int32")
    df["mkt_val"]     = df["mkt_val"].astype("float32")
    df["weight_norm"] = df["weight_norm"].astype("float32")
    df = df[df["weight_norm"].notna() & (df["weight_norm"] > 0)]
    # Crosswalk join for equity classification
    df = df.merge(xw, on="cusip8", how="left")
    df["is_matched_equity"] = df["gvkey"].notna()
    df["mkt_val_eq"] = (df["mkt_val"] * df["is_matched_equity"].astype("float32"))
    return df


def main():
    print(f"[{rss_gb():.2f}GB] === Loading reference tables ===", flush=True)

    print(f"[{rss_gb():.2f}GB] Loading crosswalk...", flush=True)
    xw = load("compustat_na_crosswalk_unified")[["cusip8", "gvkey", "iid"]].drop_duplicates("cusip8")

    print(f"[{rss_gb():.2f}GB] Loading fund metadata...", flush=True)
    fnd = load("canada_funds")[["fundno", "fundname"]].drop_duplicates("fundno")

    print(f"[{rss_gb():.2f}GB] Loading fundno→mgmt_id map (canada_fundno_to_mgmt_id_v1)...", flush=True)
    mgmt_map = load("canada_fundno_to_mgmt_id_v1")
    fundno_to_mgmt: dict[int, str] = {}
    for r in mgmt_map[["fundno", "mgmt_id"]].itertuples(index=False):
        mid = r.mgmt_id
        if pd.notna(mid) and str(mid).strip() and str(mid) != "nan":
            fundno_to_mgmt[int(r.fundno)] = str(mid).strip()
    print(f"  fundno→mgmt_id coverage: {len(fundno_to_mgmt):,} of {len(fnd):,} funds  "
          f"({100*len(fundno_to_mgmt)/len(fnd):.1f}%)", flush=True)

    print(f"[{rss_gb():.2f}GB] Loading fund features (portable)...", flush=True)
    ff = load_portable("canada_fund_features_quarterly")
    # Workaround: 'quarter' was serialized as int64 microseconds but the portable
    # wrapper declared datetime64[ns], so load_portable mis-decodes to ~1970-01-17.
    # Re-interpret the underlying ns int64 as microseconds to recover real dates.
    ff["quarter"] = pd.to_datetime(ff["quarter"].astype("int64"), unit="us").dt.normalize()
    assert ff["quarter"].dt.year.min() >= 2015, (
        f"ff['quarter'] mis-decoded: {ff['quarter'].head().tolist()}"
    )
    ff["is_active"]       = (ff["abs_change_share"] > ACTIVE_TURNOVER_THRESHOLD).fillna(False)
    ff["is_concentrated"] = (ff["top10_weight"] > CONCENTRATED_THRESHOLD).fillna(False)
    for c in FUND_FEATURE_COLS:
        ff[c] = pd.to_numeric(ff[c], errors="coerce").fillna(0.0).astype("float32")

    print(f"[{rss_gb():.2f}GB] Loading stock features (portable)...", flush=True)
    sf = load_portable("compustat_na_features_quarterly")
    sf["quarter"] = pd.to_datetime(sf["quarter"]).dt.normalize()
    sf["log_vol_q"] = np.log1p(sf["vol_q"].clip(lower=0))
    sf["log_cshom"] = np.log(sf["cshom"].where(sf["cshom"] > 0))
    sf["log_prccm"] = np.log(sf["prccm"].abs().where(sf["prccm"].abs() > 0))

    # ──────────────────────────────────────────────────────────────────
    # Phase A: per-year aggregations (small) → combine → apply hard filter
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[{rss_gb():.2f}GB] === Phase A: per-year (fundno, quarter) aggregations ===", flush=True)
    per_year_aggs = []
    for yr in range(2015, 2026):
        h = load_holdings_year(yr, xw)
        a = (h.groupby(["fundno", "quarter"], as_index=False)
               .agg(mkt_val_total            =("mkt_val", "sum"),
                    mkt_val_equity           =("mkt_val_eq", "sum"),
                    n_holdings               =("weight_norm", "size"),
                    mean_disclosure_lag_days =("lag_days", "mean")))
        per_year_aggs.append(a)
        print(f"[{rss_gb():.2f}GB]   {yr}: holding_rows={len(h):,}  agg_rows={len(a):,}", flush=True)
        del h
        gc.collect()

    agg = pd.concat(per_year_aggs, ignore_index=True)
    del per_year_aggs
    gc.collect()
    print(f"[{rss_gb():.2f}GB] Combined agg rows: {len(agg):,}", flush=True)

    agg["equity_share"]        = agg["mkt_val_equity"] / agg["mkt_val_total"].where(agg["mkt_val_total"] > 0)
    agg["equity_share_is_nan"] = agg["equity_share"].isna()
    agg = agg.merge(fnd, on="fundno", how="left")

    fundname_str = agg["fundname"].fillna("").astype(str)
    agg["name_clean"]        = ~fundname_str.str.contains(NAME_EXCLUDE)
    agg["is_index_named"]    = fundname_str.str.contains(INDEX_NAME_REGEX)
    agg["is_balanced_named"] = fundname_str.str.contains(BALANCED_NAME_REGEX)
    agg["pass_hard"] = (
        (agg["equity_share"] >= EQUITY_SHARE_THRESHOLD)
        & (agg["n_holdings"]   >= MIN_HOLDINGS)
        & agg["name_clean"]
    )
    agg["is_equity_strict"] = (agg["equity_share"] >= STRICT_EQUITY_THRESHOLD).fillna(False)
    agg["is_diversified"]   = ((agg["n_holdings"] >= DIVERSIFIED_MIN) &
                               (agg["n_holdings"] <= DIVERSIFIED_MAX))

    print(f"\n[{rss_gb():.2f}GB] === Hard-filter audit ===", flush=True)
    print(f"Total (fundno, quarter) candidates:   {len(agg):,}")
    print(f"  equity_share NaN:                   {agg['equity_share_is_nan'].sum():>7,}  ({100*agg['equity_share_is_nan'].mean():.1f}%)")
    print(f"  fail equity_share >= {EQUITY_SHARE_THRESHOLD}:           {(agg['equity_share'] < EQUITY_SHARE_THRESHOLD).sum():>7,}")
    print(f"  fail n_holdings >= {MIN_HOLDINGS}:                {(agg['n_holdings'] < MIN_HOLDINGS).sum():>7,}")
    print(f"  fail name (matched exclude regex):  {(~agg['name_clean']).sum():>7,}  ({100*(~agg['name_clean']).mean():.1f}%)")
    print(f"  Pass H3 hard filter:                {agg['pass_hard'].sum():>7,}  ({100*agg['pass_hard'].mean():.1f}%)")

    pass_keys = agg.loc[agg["pass_hard"], ["fundno", "quarter"]].copy()
    pass_set = set(zip(pass_keys["fundno"].tolist(), pass_keys["quarter"].tolist()))
    print(f"\n[{rss_gb():.2f}GB] Passing (fundno, quarter) pairs: {len(pass_set):,}", flush=True)

    # ──────────────────────────────────────────────────────────────────
    # Stock z-score (panel-level μ/σ, mirrors US convention)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[{rss_gb():.2f}GB] === Z-scoring stock features (panel-level) ===", flush=True)
    sf_z = sf[["gvkey", "iid", "quarter"] + STOCK_FEATURE_COLS].copy()
    for c in STOCK_FEATURE_COLS:
        v = pd.to_numeric(sf_z[c], errors="coerce")
        mu = v.mean()
        sd = v.std()
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        sf_z[c] = ((v - mu) / sd).astype("float32")
        print(f"  {c:<15} μ={mu:>12.4f}  σ={sd:>12.4f}", flush=True)
    sf_z[STOCK_FEATURE_COLS] = sf_z[STOCK_FEATURE_COLS].fillna(0.0)
    sf_z = sf_z.set_index(["gvkey", "iid", "quarter"])
    del sf
    gc.collect()

    # ──────────────────────────────────────────────────────────────────
    # Phase B: pass over holdings again, collect EDGES per quarter
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[{rss_gb():.2f}GB] === Phase B: per-year edge collection ===", flush=True)
    edges_by_q: dict = {}    # quarter → list of (fundno, gvkey, iid, weight_norm, mkt_val)
    fdate_pick: dict = {}    # (fundno, quarter) → chosen fdate (max within quarter)

    for yr in range(2015, 2026):
        h = load_holdings_year(yr, xw)
        # Restrict to passing keys
        h["_pass"] = list(zip(h["fundno"].tolist(), h["quarter"].tolist()))
        h = h[h["_pass"].map(lambda k: k in pass_set)]
        h = h[h["is_matched_equity"]]
        h = h.drop(columns=["_pass"])

        # Pick latest fdate per (fundno, quarter)
        for (fno, q), grp in h.groupby(["fundno", "quarter"], sort=False):
            mx = grp["fdate"].max()
            cur = fdate_pick.get((fno, q))
            if cur is None or mx > cur:
                fdate_pick[(fno, q)] = mx

        # Aggregate edges (sum across cusip → (gvkey, iid))
        h_agg = (h.groupby(["fundno", "quarter", "gvkey", "iid"], as_index=False)
                   .agg(weight_norm=("weight_norm", "sum"),
                        mkt_val   =("mkt_val", "sum"),
                        max_fdate =("fdate", "max")))

        for q_ts, sub in h_agg.groupby("quarter", sort=False):
            edges_by_q.setdefault(q_ts, []).append(sub)

        print(f"[{rss_gb():.2f}GB]   {yr}: edge_rows_added={len(h_agg):,}  quarters_so_far={len(edges_by_q)}", flush=True)
        del h, h_agg
        gc.collect()

    # Concat per-quarter edges
    for q_ts in list(edges_by_q.keys()):
        edges_by_q[q_ts] = pd.concat(edges_by_q[q_ts], ignore_index=True)
        # Final dedup if same (fundno, gvkey, iid) appeared across overlapping years
        edges_by_q[q_ts] = (edges_by_q[q_ts]
                            .groupby(["fundno", "quarter", "gvkey", "iid"], as_index=False)
                            .agg(weight_norm=("weight_norm", "sum"),
                                 mkt_val   =("mkt_val", "sum")))

    print(f"\n[{rss_gb():.2f}GB] Total quarters with edges: {len(edges_by_q)}", flush=True)

    # ──────────────────────────────────────────────────────────────────
    # Global vocabularies
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[{rss_gb():.2f}GB] === Building global ID vocabularies ===", flush=True)
    all_funds = set()
    all_stocks = set()
    for q_ts, df in edges_by_q.items():
        all_funds.update(df["fundno"].unique().tolist())
        all_stocks.update(zip(df["gvkey"].tolist(), df["iid"].tolist()))
    global_funds = sorted(all_funds)
    fund_to_gidx = {f: i for i, f in enumerate(global_funds)}
    global_stocks = sorted(all_stocks)
    stock_to_gidx = {s: i for i, s in enumerate(global_stocks)}
    print(f"Global fund vocabulary:  {len(global_funds):,}", flush=True)
    print(f"Global stock vocabulary: {len(global_stocks):,}", flush=True)

    # Global mgmt_company vocabulary — mgmt_ids that map to >=1 fund in our graph universe.
    # Funds without a mgmt_id (Option D, US convention) get no fund→mgmt edge.
    mgmts_in_use = sorted({fundno_to_mgmt[f] for f in global_funds if f in fundno_to_mgmt})
    mgmt_to_gidx = {m: i for i, m in enumerate(mgmts_in_use)}
    n_funds_mapped = sum(1 for f in global_funds if f in fundno_to_mgmt)
    print(f"Global mgmt_company vocab: {len(mgmts_in_use):,}  "
          f"(covers {n_funds_mapped}/{len(global_funds)} = {100*n_funds_mapped/len(global_funds):.1f}% of fund nodes; "
          f"others get no fund→mgmt edge — US convention)", flush=True)

    # ──────────────────────────────────────────────────────────────────
    # Phase C prep: shrink working set before the loop
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[{rss_gb():.2f}GB] === Phase C prep: pre-index ff / agg / sf_z by quarter ===", flush=True)
    flag_bool_cols  = ["is_active", "is_equity_strict", "is_diversified",
                       "is_index_named", "is_balanced_named", "is_concentrated"]
    flag_float_cols = ["equity_share", "mean_disclosure_lag_days"]
    flag_extra_bool = ["equity_share_is_nan"]

    # Compact agg to just what we need; group by quarter
    agg_keep = ["fundno", "quarter"] + [c for c in flag_bool_cols if c in agg.columns] + \
               flag_float_cols + flag_extra_bool
    agg_slim = agg.loc[agg["pass_hard"], agg_keep].copy()
    agg_by_q = {q: g.drop(columns=["quarter"]).set_index("fundno")
                for q, g in agg_slim.groupby("quarter")}
    del agg, agg_slim
    gc.collect()
    print(f"[{rss_gb():.2f}GB]   agg_by_q quarters: {len(agg_by_q)}", flush=True)

    # Compact ff: just fund features + needed bool flags
    ff_keep = ["fundno", "quarter"] + FUND_FEATURE_COLS + ["is_active", "is_concentrated"]
    ff_slim = ff[ff_keep].copy()
    # Some fundnos have multiple historical name records → multiple rows per
    # (fundno, quarter). Numeric features are nearly identical across rows;
    # only fund_age_years differs. Keep="first" picks the oldest registration
    # → larger fund_age, consistent with the fund's actual inception age.
    n_pre = len(ff_slim)
    ff_slim = ff_slim.drop_duplicates(subset=["fundno", "quarter"], keep="first")
    print(f"[{rss_gb():.2f}GB]   ff_slim dedup (fundno,quarter): "
          f"{n_pre:,} → {len(ff_slim):,} rows (-{n_pre-len(ff_slim):,})", flush=True)
    ff_by_q = {q: g.drop(columns=["quarter"]).set_index("fundno")
               for q, g in ff_slim.groupby("quarter")}
    del ff, ff_slim
    gc.collect()
    print(f"[{rss_gb():.2f}GB]   ff_by_q quarters: {len(ff_by_q)}", flush=True)

    # Pre-slice stock features by quarter (drops the leading multi-index machinery)
    sf_z_by_q = {q: g.droplevel("quarter")[STOCK_FEATURE_COLS]
                 for q, g in sf_z.groupby(level="quarter")}
    del sf_z
    gc.collect()
    print(f"[{rss_gb():.2f}GB]   sf_z_by_q quarters: {len(sf_z_by_q)}", flush=True)

    # ──────────────────────────────────────────────────────────────────
    # Phase C: per-quarter HeteroData
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[{rss_gb():.2f}GB] === Phase C: per-quarter HeteroData ===", flush=True)
    quarters = sorted(edges_by_q.keys())
    graphs = {}

    for q_ts in quarters:
        h_q = edges_by_q.pop(q_ts)   # drop from dict to release memory after use

        active_funds        = sorted(h_q["fundno"].unique().tolist())
        fund_local          = {f: i for i, f in enumerate(active_funds)}
        active_fund_global  = [fund_to_gidx[f] for f in active_funds]

        active_stock_pairs  = sorted(set(zip(h_q["gvkey"].tolist(), h_q["iid"].tolist())))
        stock_local         = {s: i for i, s in enumerate(active_stock_pairs)}
        active_stock_global = [stock_to_gidx[s] for s in active_stock_pairs]

        N_f = len(active_funds)
        N_s = len(active_stock_pairs)

        # Fund features via pre-sliced ff_by_q
        ff_q = ff_by_q.get(q_ts)
        if ff_q is not None:
            ff_aligned = ff_q.reindex(active_funds)
        else:
            ff_aligned = pd.DataFrame(index=active_funds,
                                      columns=FUND_FEATURE_COLS + ["is_active", "is_concentrated"])
        fund_x = torch.tensor(
            ff_aligned[FUND_FEATURE_COLS].fillna(0.0).values.astype(np.float32),
            dtype=torch.float,
        )

        # Side flags via pre-sliced agg_by_q
        agg_q = agg_by_q.get(q_ts)
        if agg_q is not None:
            agg_aligned = agg_q.reindex(active_funds)
        else:
            agg_aligned = pd.DataFrame(index=active_funds,
                                       columns=flag_bool_cols + flag_float_cols + flag_extra_bool)

        flag_tensors = {}
        for c in flag_bool_cols:
            if c in ("is_active", "is_concentrated"):
                src = ff_aligned
            else:
                src = agg_aligned
            flag_tensors[c] = torch.tensor(src[c].fillna(False).astype(bool).values, dtype=torch.bool)
        for c in flag_float_cols:
            flag_tensors[c] = torch.tensor(agg_aligned[c].fillna(0.0).astype(np.float32).values, dtype=torch.float)
        for c in flag_extra_bool:
            flag_tensors[c] = torch.tensor(agg_aligned[c].fillna(True).astype(bool).values, dtype=torch.bool)

        # Stock features via pre-sliced sf_z_by_q (vectorized reindex)
        sf_q = sf_z_by_q.get(q_ts)
        if sf_q is not None:
            sf_aligned = sf_q.reindex(pd.MultiIndex.from_tuples(active_stock_pairs, names=["gvkey", "iid"]))
            stock_arr = sf_aligned.fillna(0.0).values.astype(np.float32)
        else:
            stock_arr = np.zeros((N_s, len(STOCK_FEATURE_COLS)), dtype=np.float32)
        stock_x = torch.from_numpy(stock_arr).float()

        # Edges — int32 edge_index to halve memory vs int64
        e_src = np.fromiter((fund_local[f]  for f  in h_q["fundno"].tolist()),
                            dtype=np.int32, count=len(h_q))
        e_dst = np.fromiter((stock_local[(g, i)] for g, i in zip(h_q["gvkey"].tolist(), h_q["iid"].tolist())),
                            dtype=np.int32, count=len(h_q))
        edge_index_np = np.stack([e_src, e_dst])
        edge_index = torch.from_numpy(edge_index_np).long()
        edge_attr  = torch.from_numpy((h_q["weight_norm"].values * 100.0).astype(np.float32)).float()
        market_val = torch.from_numpy(h_q["mkt_val"].values.astype(np.float32)).float()

        # mgmt_company nodes for this quarter — only mgmt_ids whose funds are
        # active here, with deterministic random init (US convention; seeded so
        # rebuilds are reproducible).
        active_mgmts = sorted({fundno_to_mgmt[f] for f in active_funds if f in fundno_to_mgmt})
        mgmt_local = {m: i for i, m in enumerate(active_mgmts)}
        active_mgmt_global = [mgmt_to_gidx[m] for m in active_mgmts]
        N_m = len(active_mgmts)
        gen = torch.Generator().manual_seed(hash(("mgmt_x", q_ts)) & 0xFFFFFFFF)
        mgmt_x = torch.randn((N_m, len(FUND_FEATURE_COLS)), generator=gen, dtype=torch.float)

        # fund→mgmt edges (Option D: skip funds without mgmt_id, no placeholder)
        m_src, m_dst = [], []
        for f, fi in fund_local.items():
            mid = fundno_to_mgmt.get(f)
            if mid is not None:
                m_src.append(fi)
                m_dst.append(mgmt_local[mid])
        m_edge_index = torch.tensor([m_src, m_dst], dtype=torch.long)

        data = HeteroData()
        data['fund'].num_nodes = N_f
        data['fund'].x         = fund_x
        data['fund'].id_idx    = torch.tensor(active_fund_global, dtype=torch.long)
        for k, t in flag_tensors.items():
            setattr(data['fund'], k, t)
        data['stock'].num_nodes = N_s
        data['stock'].x         = stock_x
        data['stock'].id_idx    = torch.tensor(active_stock_global, dtype=torch.long)
        data['mgmt_company'].num_nodes = N_m
        data['mgmt_company'].x         = mgmt_x
        data['mgmt_company'].id_idx    = torch.tensor(active_mgmt_global, dtype=torch.long)
        data['fund', 'holds_stock', 'stock'].edge_index = edge_index
        data['fund', 'holds_stock', 'stock'].edge_attr  = edge_attr
        data['fund', 'holds_stock', 'stock'].market_val = market_val
        data['stock', 'rev_holds_stock', 'fund'].edge_index = edge_index.flip(0)
        data['fund', 'by_company', 'mgmt_company'].edge_index = m_edge_index
        data['mgmt_company', 'rev_by_company', 'fund'].edge_index = m_edge_index.flip(0)

        graphs[q_ts] = data

        # Drop quarter-level helpers from the by-quarter caches
        ff_by_q.pop(q_ts, None)
        agg_by_q.pop(q_ts, None)
        sf_z_by_q.pop(q_ts, None)
        del h_q, ff_aligned, agg_aligned

        if len(graphs) % 5 == 0 or len(graphs) == len(quarters):
            gc.collect()
            print(f"[{rss_gb():.2f}GB] built {len(graphs):>3}/{len(quarters)}  "
                  f"{q_ts.date()}  N_f={N_f}  N_s={N_s}  N_m={N_m}  "
                  f"E_fs={edge_index.shape[1]}  E_fm={m_edge_index.shape[1]}", flush=True)

    # ──────────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────────
    out = SNAP / "canada_graph_data.pkl"
    print(f"\n[{rss_gb():.2f}GB] Pickling to {out.name}...", flush=True)
    with open(out, "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved -> {out}  ({out.stat().st_size/1024/1024:.1f} MB)", flush=True)

    print("\n=== Summary ===", flush=True)
    print(f"Snapshots:               {len(graphs):,}")
    print(f"Date range:              {min(graphs.keys()).date()} → {max(graphs.keys()).date()}")
    print(f"Global fund vocab:       {len(global_funds):,}")
    print(f"Global stock vocab:      {len(global_stocks):,}")
    print(f"Global mgmt vocab:       {len(mgmts_in_use):,}")
    total_fs = sum(g['fund','holds_stock','stock'].edge_index.shape[1] for g in graphs.values())
    total_fm = sum(g['fund','by_company','mgmt_company'].edge_index.shape[1] for g in graphs.values())
    print(f"Total fund-stock edges:  {total_fs:,}")
    print(f"Total fund-mgmt edges:   {total_fm:,}")
    print("\nREMINDER: set SPLIT_SAFE_STOCK_SCALING=1 at training time for leak-free runs.")


if __name__ == "__main__":
    main()
