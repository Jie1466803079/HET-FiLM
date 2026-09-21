"""Convert funds-graph snapshots to Heterformer's TSV format — F1-narrow protocol.

F1-narrow protocol (fair comparison vs TAPE/SimTeG):
  - Train: q40 snapshot's fund→stock positive edges ONLY (single static snapshot,
    no time-window aggregation). q40 = 2015-09-30 = last training-window
    quarter per CLAUDE.md's VAL_QUARTERS=6, TEST_QUARTERS=14, SPLIT_GAP=2.
    Verified against core/data/funds_edge_weight_casmln.py.
  - Val: q43..q48 NEW_EDGE_STRICT entry edges (same val window as TAPE/SimTeG;
    early-stopping signal). Cold-start filter: both endpoints must exist in q40.
  - Test: q51..q64 NEW_EDGE_STRICT entry edges (same test window as TAPE/SimTeG).
    Cold-start filter: both endpoints must exist in q40.

NEW_EDGE_STRICT semantics (mirrors core/data/funds_edge_weight_casmln.py:444-446
with NEW_EDGE_STRICT=1, time_window=8):
  For quarter t, an edge (fund, stock) is "strict-new" iff (fund, stock) is
  NOT in any quarter in [max(0, t-time_window), t-1].

Note: time_window is used ONLY for the NEW_EDGE_STRICT lookback, not for
training-data aggregation. The "narrow" variant trains Heterformer on the
q40 snapshot alone — closer to Heterformer's static-single-snapshot design.

Node-type mapping (3 types per user spec — fund + stock + mgmt_company):
  book   (text-rich)            -> fund (with prospectus strategy text)
  shelves (textless K-neighbor) -> stock
  author (textless K-neighbor)  -> unused (--author_neighbour=0)
  publisher (singleton)         -> mgmt_company
  language_code (singleton)     -> dummy (always 0)
  format (singleton)            -> dummy (always 0)

Per-line TSV format (mirrors third_party/heterformer_src/src/data_heter.py:113):
  {query_side}$${key_side}\n
each side has tab-separated cols:
  [self_text, K=book_neighbour text-rich neighbour texts,
   K=shelves_neighbour stock ids,
   K=author_neighbour author ids (= 0 cols when author_neighbour=0),
   publisher_id, language_id, format_id]
Total cols per side = 1 + book_neighbour + shelves_neighbour
                       + author_neighbour + 3.

Sentinel handling: textless ids are shifted by +1 at TSV-build time so 0 is
reserved as the masked sentinel. Embedding tables index in [0, N+1).

Neighbour sampling uses ONLY the q40 snapshot — Heterformer is static and sees
one snapshot during training; at val/test we look up the queried node's q40
neighbourhood. Cold-start endpoints are filtered out before TSV is built.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

GRAPH_DEFAULT = (
    "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v4/DHGAS/graphs/"
    "graphs_v2_portfolio_clean_with_ids_since_2005Q3.pkl"
)
TEXT_DEFAULT = (
    "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/sec_filings_project/extracted/"
    "fund_485bpos_sections_temporal_v2_clean_report_anchored_carry.json"
)
OUT_DEFAULT = (
    "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/sec_filings_project/"
    "heterformer_data/f1_narrow_q40train_q43-48val_q51-64test"
)

TRAIN_QUARTER_IDX = 40   # 2015-09-30 (last training-window quarter)
VAL_QUARTER_MIN = 43     # 2016-06-30
VAL_QUARTER_MAX = 48     # 2017-09-30
TEST_QUARTER_MIN = 51    # 2018-06-30
TEST_QUARTER_MAX = 64    # 2021-09-30 (inclusive)
TIME_WINDOW = 8          # NEW_EDGE_STRICT lookback only


def collect_strict_new_edges(
    graphs,
    quarters,
    q_min: int,
    q_max: int,
    time_window: int,
    fund_gid_to_local: Dict[int, int],
    stock_gid_to_local: Dict[int, int],
) -> Tuple[List[Tuple[int, int, int, str]], Dict[str, int]]:
    """Collect (fund_local_q40, stock_local_q40, quarter_idx, quarter_date)
    for NEW_EDGE_STRICT entry edges in q_min..q_max, filtered to endpoints
    present in q40 (cold-start filter)."""
    counters = {"total": 0, "strict_new": 0, "kept": 0, "skipped_cold": 0}
    edges: List[Tuple[int, int, int, str]] = []
    etype = ("fund", "holds_stock", "stock")
    for t in range(q_min, q_max + 1):
        snap = graphs[quarters[t]]
        if etype not in snap.edge_types:
            continue
        # Support set: (f_gid, s_gid) pairs from quarters [t-time_window, t-1]
        support: Set[Tuple[int, int]] = set()
        for k in range(max(0, t - time_window), t):
            sp = graphs[quarters[k]]
            if etype not in sp.edge_types:
                continue
            e = sp[etype].edge_index
            fg = sp["fund"].id_idx
            sg = sp["stock"].id_idx
            for kk in range(e.shape[1]):
                fl = int(e[0, kk].item())
                sl = int(e[1, kk].item())
                if fl < fg.shape[0] and sl < sg.shape[0]:
                    support.add(
                        (int(fg[fl].item()), int(sg[sl].item()))
                    )
        # Iterate q[t] positives
        e_t = snap[etype].edge_index
        fg_t = snap["fund"].id_idx
        sg_t = snap["stock"].id_idx
        q_date = (quarters[t].strftime("%Y-%m-%d")
                  if hasattr(quarters[t], "strftime")
                  else str(quarters[t]))
        for kk in range(e_t.shape[1]):
            fl_t = int(e_t[0, kk].item())
            sl_t = int(e_t[1, kk].item())
            if fl_t >= fg_t.shape[0] or sl_t >= sg_t.shape[0]:
                continue
            f_gid = int(fg_t[fl_t].item())
            s_gid = int(sg_t[sl_t].item())
            counters["total"] += 1
            if (f_gid, s_gid) in support:
                continue
            counters["strict_new"] += 1
            # Cold-start filter: both endpoints must exist in q40
            if f_gid not in fund_gid_to_local or s_gid not in stock_gid_to_local:
                counters["skipped_cold"] += 1
                continue
            edges.append((
                fund_gid_to_local[f_gid],
                stock_gid_to_local[s_gid],
                t,
                q_date,
            ))
            counters["kept"] += 1
    return edges, counters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_pkl", default=GRAPH_DEFAULT)
    ap.add_argument("--text_json", default=TEXT_DEFAULT)
    ap.add_argument("--train_quarter_idx", type=int, default=TRAIN_QUARTER_IDX)
    ap.add_argument("--val_quarter_min", type=int, default=VAL_QUARTER_MIN)
    ap.add_argument("--val_quarter_max", type=int, default=VAL_QUARTER_MAX)
    ap.add_argument("--test_quarter_min", type=int, default=TEST_QUARTER_MIN)
    ap.add_argument("--test_quarter_max", type=int, default=TEST_QUARTER_MAX)
    ap.add_argument("--time_window", type=int, default=TIME_WINDOW)
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--book_neighbour", type=int, default=5)
    ap.add_argument("--shelves_neighbour", type=int, default=5)
    ap.add_argument("--author_neighbour", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_text_chars", type=int, default=8192)
    args = ap.parse_args()

    random.seed(args.seed)

    # ── Load graph + identify train snapshot ──────────────────────────────
    print(f"[build_funds_data] loading {args.graph_pkl}", flush=True)
    with open(args.graph_pkl, "rb") as f:
        g = pickle.load(f)
    quarters = sorted(g.keys())
    train_dt = quarters[args.train_quarter_idx]
    train_date = (train_dt.strftime("%Y-%m-%d")
                  if hasattr(train_dt, "strftime") else str(train_dt))
    print(f"[build_funds_data] F1-narrow protocol", flush=True)
    print(f"  train snapshot: q{args.train_quarter_idx} = {train_date} (single snapshot only)",
          flush=True)
    print(f"  val window:     q{args.val_quarter_min}..q{args.val_quarter_max} "
          f"({quarters[args.val_quarter_min].date()}..{quarters[args.val_quarter_max].date()})",
          flush=True)
    print(f"  test window:    q{args.test_quarter_min}..q{args.test_quarter_max} "
          f"({quarters[args.test_quarter_min].date()}..{quarters[args.test_quarter_max].date()})",
          flush=True)
    print(f"  time_window:    {args.time_window} (NEW_EDGE_STRICT lookback only)",
          flush=True)
    train_snap = g[train_dt]

    # ── q40 node lookups ──────────────────────────────────────────────────
    fund_id_idx = train_snap["fund"].id_idx
    stock_id_idx = train_snap["stock"].id_idx
    mgmt_id_idx = train_snap["mgmt_company"].id_idx \
        if "mgmt_company" in train_snap.node_types else None
    n_funds = int(fund_id_idx.shape[0])
    n_stocks = int(stock_id_idx.shape[0])
    n_mgmt = int(mgmt_id_idx.shape[0]) if mgmt_id_idx is not None else 0
    print(f"[build_funds_data] q40 nodes: {n_funds} funds, {n_stocks} stocks, "
          f"{n_mgmt} mgmt_companies", flush=True)

    fund_gid_to_local = {int(fund_id_idx[i].item()): i for i in range(n_funds)}
    stock_gid_to_local = {int(stock_id_idx[i].item()): i for i in range(n_stocks)}

    # ── Text lookup for q40 ───────────────────────────────────────────────
    print(f"[build_funds_data] loading {args.text_json}", flush=True)
    with open(args.text_json) as f:
        text_records = json.load(f)
    fund_gid_to_text: Dict[int, str] = {}
    for r in text_records:
        if str(r.get("timestamp")) == train_date:
            fund_gid_to_text[int(r["id_idx"])] = (r.get("strategy") or "").strip()
    print(f"[build_funds_data] {len(fund_gid_to_text)} funds with text for "
          f"{train_date}", flush=True)

    def sanitize_text(t: str) -> str:
        if not t:
            return ""
        t = t[: args.max_text_chars]
        return (t.replace("\t", " ").replace("$$", " ")
                 .replace("\n", " ").replace("\r", " "))

    fund_texts: List[str] = []
    n_text = 0
    for i in range(n_funds):
        gid = int(fund_id_idx[i].item())
        t = fund_gid_to_text.get(gid, "")
        if t:
            n_text += 1
        fund_texts.append(sanitize_text(t))
    print(f"[build_funds_data] {n_text}/{n_funds} funds have non-empty q40 text",
          flush=True)

    # ── q40 fund -> mgmt_company singleton ────────────────────────────────
    fund_to_mgmt: Dict[int, int] = {}
    if ("fund", "by_company", "mgmt_company") in train_snap.edge_types:
        e = train_snap[("fund", "by_company", "mgmt_company")].edge_index
        for k in range(e.shape[1]):
            f, c = int(e[0, k].item()), int(e[1, k].item())
            fund_to_mgmt[f] = c
    print(f"[build_funds_data] {len(fund_to_mgmt)} funds have mgmt_company in q40",
          flush=True)

    # ── q40 fund-stock holdings (positives + neighbour context) ───────────
    q40_fund_to_stocks: Dict[int, List[int]] = defaultdict(list)
    q40_stock_to_funds: Dict[int, List[int]] = defaultdict(list)
    e = train_snap[("fund", "holds_stock", "stock")].edge_index
    for k in range(e.shape[1]):
        f, s = int(e[0, k].item()), int(e[1, k].item())
        q40_fund_to_stocks[f].append(s)
        q40_stock_to_funds[s].append(f)
    n_q40_edges = e.shape[1]
    print(f"[build_funds_data] q40 fund-stock edges: {n_q40_edges}", flush=True)

    # ── Neighbour samplers ────────────────────────────────────────────────
    def pad_to_k(chosen: List[int], k: int) -> List[int]:
        while len(chosen) < k:
            chosen.append(-1)
        return chosen[:k]

    def text_rich_neigh_fund(f_idx: int, k: int) -> List[int]:
        co: Set[int] = set()
        for s in q40_fund_to_stocks.get(f_idx, []):
            for ff in q40_stock_to_funds.get(s, []):
                if ff != f_idx:
                    co.add(ff)
        cand = list(co)
        if len(cand) < k:
            extra = [x for x in range(n_funds) if x != f_idx and x not in co]
            random.shuffle(extra)
            cand += extra[: (k - len(cand))]
        random.shuffle(cand)
        return pad_to_k(cand[:k], k)

    def text_rich_neigh_stock(s_idx: int, k: int) -> List[int]:
        funds = list(q40_stock_to_funds.get(s_idx, []))
        random.shuffle(funds)
        return pad_to_k(funds[:k], k)

    def shelves_fund(f_idx: int, k: int) -> List[int]:
        s_list = list(q40_fund_to_stocks.get(f_idx, []))
        random.shuffle(s_list)
        return pad_to_k(s_list[:k], k)

    def shelves_stock(s_idx: int, k: int) -> List[int]:
        co: Set[int] = set()
        for f in q40_stock_to_funds.get(s_idx, []):
            for ss in q40_fund_to_stocks.get(f, []):
                if ss != s_idx:
                    co.add(ss)
        cand = list(co)
        if len(cand) < k:
            extra = [x for x in range(n_stocks) if x != s_idx and x not in co]
            random.shuffle(extra)
            cand += extra[: (k - len(cand))]
        random.shuffle(cand)
        return pad_to_k(cand[:k], k)

    def shift(x: int) -> int:
        return x + 1 if x >= 0 else 0  # 0 = masked sentinel

    def build_columns(node_type: str, idx: int) -> List[str]:
        if node_type == "fund":
            self_text = fund_texts[idx]
            tn = text_rich_neigh_fund(idx, args.book_neighbour)
            tn_text = [fund_texts[c] if c >= 0 and fund_texts[c] else ""
                       for c in tn]
            sh = shelves_fund(idx, args.shelves_neighbour)
            au: List[int] = []
            pub = fund_to_mgmt.get(idx, -1)
        else:  # stock
            self_text = ""
            tn = text_rich_neigh_stock(idx, args.book_neighbour)
            tn_text = [fund_texts[c] if c >= 0 and fund_texts[c] else ""
                       for c in tn]
            sh = shelves_stock(idx, args.shelves_neighbour)
            au = []
            pub = -1
        lang = -1
        fmt = -1
        cols = (
            [self_text]
            + tn_text
            + [str(shift(x)) for x in sh]
            + [str(shift(x)) for x in au]
            + [str(shift(pub)), str(shift(lang)), str(shift(fmt))]
        )
        expected = 1 + args.book_neighbour + args.shelves_neighbour \
                   + args.author_neighbour + 3
        assert len(cols) == expected, \
            f"col count {len(cols)} != expected {expected} for {node_type}={idx}"
        return cols

    # ── TRAIN edges: all q40 fund-stock positives ─────────────────────────
    q40_pos = [(f, s) for f, ss in q40_fund_to_stocks.items() for s in ss]
    random.shuffle(q40_pos)
    print(f"[build_funds_data] train edges (q40 positives): {len(q40_pos)}",
          flush=True)

    # ── VAL edges: NEW_EDGE_STRICT in q43..q48 ────────────────────────────
    print(f"[build_funds_data] collecting val NEW_EDGE_STRICT entries...",
          flush=True)
    val_edges, val_counts = collect_strict_new_edges(
        g, quarters, args.val_quarter_min, args.val_quarter_max,
        args.time_window, fund_gid_to_local, stock_gid_to_local,
    )
    print(f"[build_funds_data] val: total={val_counts['total']} "
          f"strict_new={val_counts['strict_new']} "
          f"kept={val_counts['kept']} "
          f"cold-skipped={val_counts['skipped_cold']}", flush=True)

    # ── TEST edges: NEW_EDGE_STRICT in q51..q64 ───────────────────────────
    print(f"[build_funds_data] collecting test NEW_EDGE_STRICT entries...",
          flush=True)
    test_edges, test_counts = collect_strict_new_edges(
        g, quarters, args.test_quarter_min, args.test_quarter_max,
        args.time_window, fund_gid_to_local, stock_gid_to_local,
    )
    print(f"[build_funds_data] test: total={test_counts['total']} "
          f"strict_new={test_counts['strict_new']} "
          f"kept={test_counts['kept']} "
          f"cold-skipped={test_counts['skipped_cold']}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_train_tsv(edges_local, path: Path):
        with open(path, "w") as fout:
            for fund_i, stock_i in edges_local:
                q_cols = build_columns("fund", fund_i)
                k_cols = build_columns("stock", stock_i)
                fout.write("\t".join(q_cols) + "$$" + "\t".join(k_cols) + "\n")
        print(f"[build_funds_data] wrote {len(edges_local)} rows -> {path}",
              flush=True)

    def write_eval_tsv(edges_full, path: Path, sidecar_path: Path):
        with open(path, "w") as fout, open(sidecar_path, "w") as scar:
            for f_idx, s_idx, qi, qd in edges_full:
                q_cols = build_columns("fund", f_idx)
                k_cols = build_columns("stock", s_idx)
                fout.write("\t".join(q_cols) + "$$" + "\t".join(k_cols) + "\n")
                scar.write(json.dumps({
                    "fund_local": f_idx,
                    "stock_local": s_idx,
                    "quarter_idx": qi,
                    "quarter_date": qd,
                }) + "\n")
        print(f"[build_funds_data] wrote {len(edges_full)} rows -> {path}",
              flush=True)
        print(f"[build_funds_data] wrote sidecar -> {sidecar_path}",
              flush=True)

    write_train_tsv(q40_pos, out_dir / "train.tsv")
    write_eval_tsv(val_edges, out_dir / "val.tsv", out_dir / "val_index.jsonl")
    write_eval_tsv(test_edges, out_dir / "test.tsv", out_dir / "test_index.jsonl")

    # ── Meta JSON ─────────────────────────────────────────────────────────
    meta = {
        "protocol": "F1-narrow (q40 train + q43..q48 val + q51..q64 test, NEW_EDGE_STRICT)",
        "train_quarter_idx": args.train_quarter_idx,
        "train_quarter_date": train_date,
        "val_quarter_min": args.val_quarter_min,
        "val_quarter_max": args.val_quarter_max,
        "test_quarter_min": args.test_quarter_min,
        "test_quarter_max": args.test_quarter_max,
        "time_window": args.time_window,
        "time_window_used_for": "NEW_EDGE_STRICT lookback only (NOT training aggregation)",
        "n_funds_q40": n_funds,
        "n_stocks_q40": n_stocks,
        "n_mgmt_q40": n_mgmt,
        "shelves_num": n_stocks + 1,
        "author_num": 1,
        "publisher_num": n_mgmt + 1,
        "language_code_num": 1,
        "format_num": 1,
        "book_neighbour": args.book_neighbour,
        "shelves_neighbour": args.shelves_neighbour,
        "author_neighbour": args.author_neighbour,
        "n_train_rows": len(q40_pos),
        "n_val_rows": len(val_edges),
        "n_test_rows": len(test_edges),
        "val_counts": val_counts,
        "test_counts": test_counts,
        "max_text_chars": args.max_text_chars,
        "seed": args.seed,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[build_funds_data] wrote meta -> {out_dir / 'meta.json'}",
          flush=True)
    print(f"[build_funds_data] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
