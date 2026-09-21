"""Build textless-node pretrain data + stub checkpoints for Heterformer F1-narrow.

Mirrors the upstream `pretrain/data.py` schema:
  Each line of train_X.tsv (and val_X.tsv) is:
    {self_text}\t{textless_neighbour_id}

For funds (3 node types):
  - mode 's' (shelves=stock):  pairs (fund_text, stock_id)
                               sampled from q40 fund-stock holdings
  - mode 'p' (publisher=mgmt_company):  pairs (fund_text, mgmt_id)
                                        sampled from q40 fund-mgmt edges
  - modes 'a', 'l', 'f' (author / language_code / format):  size-1 dummies,
                                                            stub checkpoints

Outputs (under --pretrain_dir):
  - train_s.tsv, val_s.tsv  (real)
  - train_p.tsv, val_p.tsv  (real)
  - author_MF_64.pt, language_code_MF_64.pt, format_MF_64.pt  (stub)
    each a pickle dict matching upstream's expected schema:
      {'author_embeddings': torch.zeros(1, heter_embed_size),
       'linear.weight':     torch.zeros(hidden_size, heter_embed_size),
       'linear.bias':       torch.zeros(hidden_size)}
    Loaded by HeterformerD.init_mta_embed when --pretrain_embed True.

The stubs are intentionally zero — those node types are unused (author_neighbour=0
and language/format are 1-of-1 lookups always returning the same vector), so
the embeddings never contribute to a gradient at training time. The Linear
projections are zeroed too so the masked sentinel id contributes nothing.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

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
    "heterformer_data/f1_narrow_q40train_q43-48val_q51-64test/pretrain_dir"
)


def sanitize_text(t: str, max_chars: int) -> str:
    if not t:
        return ""
    t = t[:max_chars]
    return (t.replace("\t", " ").replace("$$", " ")
             .replace("\n", " ").replace("\r", " "))


def write_pairs(pairs, train_path: Path, val_path: Path, val_frac: float):
    random.shuffle(pairs)
    n_val = int(val_frac * len(pairs))
    val, train = pairs[:n_val], pairs[n_val:]
    for path, rows in [(train_path, train), (val_path, val)]:
        with open(path, "w") as f:
            for text, nid in rows:
                f.write(f"{text}\t{nid}\n")
        print(f"[build_pretrain_data] wrote {len(rows)} rows -> {path}",
              flush=True)


def write_stub_ckpt(path: Path, n: int, heter_embed_size: int, hidden_size: int):
    payload = {
        "author_embeddings": torch.zeros(n, heter_embed_size),
        "linear.weight": torch.zeros(hidden_size, heter_embed_size),
        "linear.bias": torch.zeros(hidden_size),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[build_pretrain_data] wrote stub ckpt n={n} d={heter_embed_size} "
          f"-> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_pkl", default=GRAPH_DEFAULT)
    ap.add_argument("--text_json", default=TEXT_DEFAULT)
    ap.add_argument("--train_quarter_idx", type=int, default=40)
    ap.add_argument("--pretrain_dir", default=OUT_DEFAULT)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_text_chars", type=int, default=2048,
                    help="Pretrain max_length=64 tokens ≈ 200 chars; storing "
                         "2K chars gives headroom.")
    ap.add_argument("--heter_embed_size", type=int, default=64,
                    help="Match the value passed to main training "
                         "(--heter_embed_size in main.py).")
    ap.add_argument("--hidden_size", type=int, default=768,
                    help="BERT hidden size (bert-base-uncased = 768).")
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"[build_pretrain_data] loading {args.graph_pkl}", flush=True)
    with open(args.graph_pkl, "rb") as f:
        g = pickle.load(f)
    quarters = sorted(g.keys())
    train_dt = quarters[args.train_quarter_idx]
    train_date = (train_dt.strftime("%Y-%m-%d")
                  if hasattr(train_dt, "strftime") else str(train_dt))
    print(f"[build_pretrain_data] train snapshot q{args.train_quarter_idx} = "
          f"{train_date}", flush=True)
    snap = g[train_dt]

    fund_id_idx = snap["fund"].id_idx
    n_funds = int(fund_id_idx.shape[0])
    stock_id_idx = snap["stock"].id_idx
    n_stocks = int(stock_id_idx.shape[0])
    mgmt_id_idx = snap["mgmt_company"].id_idx \
        if "mgmt_company" in snap.node_types else None
    n_mgmt = int(mgmt_id_idx.shape[0]) if mgmt_id_idx is not None else 0
    print(f"[build_pretrain_data] {n_funds} funds, {n_stocks} stocks, "
          f"{n_mgmt} mgmt_companies", flush=True)

    # Build fund text lookup for the train_date
    with open(args.text_json) as f:
        records = json.load(f)
    fund_gid_to_text: Dict[int, str] = {}
    for r in records:
        if str(r.get("timestamp")) == train_date:
            fund_gid_to_text[int(r["id_idx"])] = (r.get("strategy") or "").strip()
    fund_texts: List[str] = []
    for i in range(n_funds):
        gid = int(fund_id_idx[i].item())
        fund_texts.append(sanitize_text(
            fund_gid_to_text.get(gid, ""), args.max_text_chars
        ))
    n_with = sum(1 for t in fund_texts if t)
    print(f"[build_pretrain_data] {n_with}/{n_funds} funds have non-empty "
          f"q40 text", flush=True)

    out_dir = Path(args.pretrain_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mode 's' (shelves = stock): pairs (fund_text, stock_id+1).
    # Sample one pair per (fund, stock) edge in q40.
    pairs_s = []
    e = snap[("fund", "holds_stock", "stock")].edge_index
    for k in range(e.shape[1]):
        f, s = int(e[0, k].item()), int(e[1, k].item())
        text = fund_texts[f]
        if not text:
            continue
        pairs_s.append((text, s + 1))  # +1 shift: 0 = masked sentinel
    print(f"[build_pretrain_data] mode 's' pairs: {len(pairs_s)}", flush=True)
    write_pairs(pairs_s, out_dir / "train_s.tsv", out_dir / "val_s.tsv",
                args.val_frac)

    # Mode 'p' (publisher = mgmt_company): pairs (fund_text, mgmt_id+1).
    pairs_p = []
    if ("fund", "by_company", "mgmt_company") in snap.edge_types:
        e2 = snap[("fund", "by_company", "mgmt_company")].edge_index
        for k in range(e2.shape[1]):
            f, c = int(e2[0, k].item()), int(e2[1, k].item())
            text = fund_texts[f]
            if not text:
                continue
            pairs_p.append((text, c + 1))
    print(f"[build_pretrain_data] mode 'p' pairs: {len(pairs_p)}", flush=True)
    write_pairs(pairs_p, out_dir / "train_p.tsv", out_dir / "val_p.tsv",
                args.val_frac)

    # Stub .pt checkpoints for the 3 dummy node types (author, language_code,
    # format). Each is a size-1 lookup that returns zeros; the model still
    # expects a file at the path. Naming matches upstream's HeterformerD.
    d = args.heter_embed_size
    h = args.hidden_size
    write_stub_ckpt(out_dir / f"author_MF_{d}.pt",        n=1, heter_embed_size=d, hidden_size=h)
    write_stub_ckpt(out_dir / f"language_code_MF_{d}.pt", n=1, heter_embed_size=d, hidden_size=h)
    write_stub_ckpt(out_dir / f"format_MF_{d}.pt",        n=1, heter_embed_size=d, hidden_size=h)

    # Also dump a meta.json
    meta = {
        "protocol": "F1-narrow pretrain data",
        "train_quarter_idx": args.train_quarter_idx,
        "train_quarter_date": train_date,
        "n_funds": n_funds,
        "n_stocks_shelves_num": n_stocks + 1,
        "n_mgmt_publisher_num": n_mgmt + 1,
        "n_mode_s_pairs": len(pairs_s),
        "n_mode_p_pairs": len(pairs_p),
        "val_frac": args.val_frac,
        "heter_embed_size": args.heter_embed_size,
        "hidden_size": args.hidden_size,
        "max_text_chars": args.max_text_chars,
        "seed": args.seed,
    }
    with open(out_dir / "pretrain_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[build_pretrain_data] wrote pretrain_meta.json", flush=True)
    print(f"[build_pretrain_data] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
