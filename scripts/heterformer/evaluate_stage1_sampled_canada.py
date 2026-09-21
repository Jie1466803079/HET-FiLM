"""Stage-1 sampled-negative AUC/AP eval for Canadian Heterformer.

Loads the trained Canadian Heterformer checkpoint, encodes every fund and
every stock once from the q27 (training) snapshot neighbourhood, then per
test quarter (q36-q43) scores positives + 1:1 sampled negatives using raw
inner-product similarity. AUC/AP are pooled across quarters — directly
comparable to the pipeline's `test_entry_auc` / `test_entry_ap` produced by
`core/trainer/edge_multitask.py` under NEW_EDGE_STRICT=1 with
`--neg_sampling_ratio 1.0`.

Negative sampling matches the pipeline's CasMLN-literal candidate pool
(`core/data/funds_edge_weight_casmln.py:534-540`): for each snapshot, draw
uniformly from `unique(pos_funds) × unique(pos_stocks) \ positives`.

Outputs (under <ckpt_dir>/eval_stage1_sampled/):
  - per_quarter_metrics.json:  list of {quarter_idx, quarter_date, n_pos,
                                        n_neg, auc, ap}
  - overall_metrics.json:      pooled + quarter-averaged AUC/AP

This file is a sibling to evaluate_stage1.py (which does full-cartesian
scoring for the US benchmark); it does NOT modify existing code paths.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


GRAPH_PKL = "/srv/scratch/dbgcse/jieliu/canada_funds_snapshot_20260531/canada_graph_data.pkl"
TEXT_JSON = ("/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/"
             "sec_filings_project/extracted/"
             "fund_485bpos_sections_temporal_v2_clean_report_anchored_carry.json")


def _encode_columns(tok, cols: List[str], max_length: int, n_book: int,
                    n_mta: int, device: str):
    text_cols = cols[: 1 + n_book]
    mta_cols = cols[1 + n_book :]
    assert len(mta_cols) == n_mta
    enc = tok(text_cols, max_length=max_length, padding="max_length",
              truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device)
    mta_ids = torch.tensor([int(x) for x in mta_cols],
                           dtype=torch.long, device=device)
    text_mask = torch.tensor([1 if c.strip() else 0 for c in text_cols],
                             dtype=torch.long, device=device)
    mta_mask = (mta_ids != 0).long()
    node_mask = torch.cat([text_mask, mta_mask], dim=0)
    return input_ids, attention, mta_ids, node_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Canadian data dir (contains meta.json, test_index.jsonl).")
    ap.add_argument("--ckpt_path", required=True,
                    help="Trained Canadian Heterformer .pt checkpoint.")
    ap.add_argument("--pretrain_dir", default=None,
                    help="Defaults to <data_dir>/pretrain_dir.")
    ap.add_argument("--out_dir", default=None,
                    help="Defaults to <ckpt_dir>/eval_stage1_sampled.")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--book_neighbour", type=int, default=5)
    ap.add_argument("--shelves_neighbour", type=int, default=5)
    ap.add_argument("--author_neighbour", type=int, default=0)
    ap.add_argument("--heter_embed_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for negative sampling.")
    ap.add_argument("--neg_ratio", type=float, default=1.0,
                    help="Negatives per positive (default 1.0 matches pipeline).")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.ckpt_path)
    pretrain_dir = Path(args.pretrain_dir) if args.pretrain_dir \
        else data_dir / "pretrain_dir"
    out_dir = Path(args.out_dir) if args.out_dir \
        else ckpt_path.parent / "eval_stage1_sampled"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    n_funds = meta["n_funds_q40"]
    n_stocks = meta["n_stocks_q40"]
    print(f"[eval_s1_sampled] data_dir={data_dir}", flush=True)
    print(f"[eval_s1_sampled] meta: {n_funds} funds, {n_stocks} stocks, "
          f"protocol={meta.get('protocol')}", flush=True)

    n_book = args.book_neighbour
    n_mta = args.shelves_neighbour + args.author_neighbour + 3

    # ── Load Heterformer ──────────────────────────────────────────────────
    sys.path.insert(
        0,
        "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/mutual_fund_prediction/"
        "third_party/heterformer_src",
    )
    from transformers import BertConfig, BertTokenizerFast
    from src.model.HeterformerD import HeterFormerDsForNeighborPredict
    cfg = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model = HeterFormerDsForNeighborPredict.from_pretrained(
        "bert-base-uncased", config=cfg,
    )
    model.shelves_num = meta["shelves_num"]
    model.author_num = meta["author_num"]
    model.publisher_num = meta["publisher_num"]
    model.language_code_num = meta["language_code_num"]
    model.format_num = meta["format_num"]
    model.heter_embed_size = args.heter_embed_size
    model.book_neighbour = args.book_neighbour
    model.shelves_neighbour = args.shelves_neighbour
    model.author_neighbour = args.author_neighbour
    model.init_mta_embed(True, str(pretrain_dir))
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(args.device).eval()
    print(f"[eval_s1_sampled] loaded checkpoint {ckpt_path}", flush=True)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # ── Test-index → per-quarter positives + unique node lists ────────────
    fund_locals: set = set()
    stock_locals: set = set()
    edges_by_quarter: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    quarter_dates: Dict[int, str] = {}
    with open(data_dir / "test_index.jsonl") as f:
        for line in f:
            r = json.loads(line)
            f_loc = int(r["fund_local"])
            s_loc = int(r["stock_local"])
            qi = int(r["quarter_idx"])
            fund_locals.add(f_loc)
            stock_locals.add(s_loc)
            edges_by_quarter[qi].append((f_loc, s_loc))
            quarter_dates[qi] = r["quarter_date"]
    total_pos = sum(len(v) for v in edges_by_quarter.values())
    print(f"[eval_s1_sampled] test: {total_pos} positives, "
          f"{len(fund_locals)} funds, {len(stock_locals)} stocks, "
          f"{len(edges_by_quarter)} quarters", flush=True)

    # ── Canadian q27 snapshot for neighbour context ───────────────────────
    with open(GRAPH_PKL, "rb") as f:
        g = pickle.load(f)
    quarters = sorted(g.keys())
    train_snap = g[quarters[meta["train_quarter_idx"]]]
    train_date = meta["train_quarter_date"]
    print(f"[eval_s1_sampled] q27 snapshot: {train_date}", flush=True)

    fund_id_idx = train_snap["fund"].id_idx
    stock_id_idx = train_snap["stock"].id_idx
    n_funds_q40 = int(fund_id_idx.shape[0])
    n_stocks_q40 = int(stock_id_idx.shape[0])

    # Text (Canadian funds mostly absent → empty strategy; kept identical to
    # evaluate_stage2_canada.py so scoring parity with Stage-2 numbers holds).
    print(f"[eval_s1_sampled] loading text json...", flush=True)
    with open(TEXT_JSON) as f:
        text_records = json.load(f)
    fund_gid_to_text: Dict[int, str] = {}
    for r in text_records:
        if str(r.get("timestamp")) == train_date:
            fund_gid_to_text[int(r["id_idx"])] = (r.get("strategy") or "").strip()
    fund_texts: List[str] = []
    for i in range(n_funds_q40):
        gid = int(fund_id_idx[i].item())
        t = fund_gid_to_text.get(gid, "")
        t = t[:meta.get("max_text_chars", 8192)]
        t = t.replace("\t", " ").replace("$$", " ").replace("\n", " ").replace("\r", " ")
        fund_texts.append(t)

    # Neighbour adjacency in q27
    q40_fund_to_stocks: Dict[int, List[int]] = defaultdict(list)
    q40_stock_to_funds: Dict[int, List[int]] = defaultdict(list)
    e = train_snap[("fund", "holds_stock", "stock")].edge_index
    for k in range(e.shape[1]):
        f_, s_ = int(e[0, k].item()), int(e[1, k].item())
        q40_fund_to_stocks[f_].append(s_)
        q40_stock_to_funds[s_].append(f_)
    fund_to_mgmt: Dict[int, int] = {}
    if ("fund", "by_company", "mgmt_company") in train_snap.edge_types:
        em = train_snap[("fund", "by_company", "mgmt_company")].edge_index
        for k in range(em.shape[1]):
            fl, c = int(em[0, k].item()), int(em[1, k].item())
            fund_to_mgmt[fl] = c

    def pad_to_k(chosen, k):
        while len(chosen) < k:
            chosen.append(-1)
        return chosen[:k]

    def build_fund_columns(f_idx: int) -> List[str]:
        self_text = fund_texts[f_idx]
        co = set()
        for s in q40_fund_to_stocks.get(f_idx, []):
            for ff in q40_stock_to_funds.get(s, []):
                if ff != f_idx:
                    co.add(ff)
        cand = list(co)
        if len(cand) < n_book:
            extra = [x for x in range(n_funds_q40) if x != f_idx and x not in co]
            extra.sort(); cand += extra[: (n_book - len(cand))]
        tn = pad_to_k(cand[:n_book], n_book)
        tn_text = [fund_texts[c] if c >= 0 and fund_texts[c] else "" for c in tn]
        sh = pad_to_k(list(q40_fund_to_stocks.get(f_idx, []))[: args.shelves_neighbour],
                      args.shelves_neighbour)
        pub = fund_to_mgmt.get(f_idx, -1)
        def shift(x): return x + 1 if x >= 0 else 0
        cols = ([self_text] + tn_text
                + [str(shift(x)) for x in sh]
                + [str(shift(pub)), "0", "0"])
        return cols

    def build_stock_columns(s_idx: int) -> List[str]:
        self_text = ""
        funds = list(q40_stock_to_funds.get(s_idx, []))[: n_book]
        tn = pad_to_k(funds, n_book)
        tn_text = [fund_texts[c] if c >= 0 and fund_texts[c] else "" for c in tn]
        co = set()
        for f in q40_stock_to_funds.get(s_idx, []):
            for ss in q40_fund_to_stocks.get(f, []):
                if ss != s_idx:
                    co.add(ss)
        cand = list(co)
        if len(cand) < args.shelves_neighbour:
            extra = [x for x in range(n_stocks_q40) if x != s_idx and x not in co]
            extra.sort(); cand += extra[: (args.shelves_neighbour - len(cand))]
        sh = pad_to_k(cand[: args.shelves_neighbour], args.shelves_neighbour)
        def shift(x): return x + 1 if x >= 0 else 0
        cols = ([self_text] + tn_text
                + [str(shift(x)) for x in sh]
                + ["0", "0", "0"])
        return cols

    @torch.no_grad()
    def encode(cols: List[str]) -> torch.Tensor:
        ids, attn, mta, mask = _encode_columns(
            tok, cols, args.max_length, n_book, n_mta, args.device,
        )
        ids = ids.unsqueeze(0)
        attn = attn.unsqueeze(0)
        mta = mta.unsqueeze(0)
        mask = mask.unsqueeze(0)
        emb = model.infer(ids, attn, mta, mask)
        return emb.squeeze(0).float().cpu()

    # ── Encode every appearing fund + every stock ─────────────────────────
    # Stocks: encode ALL n_stocks_q40 (matches evaluate_stage1.py convention),
    # since negatives could draw any stock in unique(pos_stocks).
    print(f"[eval_s1_sampled] encoding {len(fund_locals)} funds + "
          f"{n_stocks_q40} stocks...", flush=True)
    fund_emb: Dict[int, torch.Tensor] = {}
    for i, f_loc in enumerate(sorted(fund_locals)):
        fund_emb[f_loc] = encode(build_fund_columns(f_loc))
        if i % 50 == 0:
            print(f"  fund {i+1}/{len(fund_locals)}", flush=True)
    hidden = fund_emb[next(iter(fund_emb))].shape[0]
    stock_emb_mat = torch.zeros(n_stocks_q40, hidden)
    for s_loc in range(n_stocks_q40):
        stock_emb_mat[s_loc] = encode(build_stock_columns(s_loc))
        if s_loc % 300 == 0:
            print(f"  stock {s_loc+1}/{n_stocks_q40}", flush=True)

    # ── Per-quarter: sample 1:1 negatives → inner-product AUC/AP ─────────
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = random.Random(args.seed)
    per_quarter = []
    pooled_scores: List[float] = []
    pooled_labels: List[int] = []
    for qi in sorted(edges_by_quarter.keys()):
        pos_pairs = list(dict.fromkeys(edges_by_quarter[qi]))  # dedup, keep order
        pos_set = set(pos_pairs)
        u_funds = sorted({f for f, _ in pos_pairs})
        u_stocks = sorted({s for _, s in pos_pairs})
        n_pos = len(pos_pairs)
        n_neg_target = max(1, int(round(n_pos * args.neg_ratio)))

        neg_pairs: List[Tuple[int, int]] = []
        neg_set: set = set()
        max_tries = n_neg_target * 20 + 100
        pool_size = len(u_funds) * len(u_stocks) - n_pos
        while len(neg_pairs) < n_neg_target and max_tries > 0 and pool_size > 0:
            f_ = rng.choice(u_funds)
            s_ = rng.choice(u_stocks)
            pair = (f_, s_)
            if pair in pos_set or pair in neg_set:
                max_tries -= 1
                continue
            neg_set.add(pair)
            neg_pairs.append(pair)
        if len(neg_pairs) < n_neg_target:
            print(f"  q{qi}: WARN sampled only {len(neg_pairs)}/{n_neg_target} "
                  f"negatives (pool exhausted or tries capped)", flush=True)

        # Score pos + neg
        all_pairs = pos_pairs + neg_pairs
        labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
        f_idx = torch.tensor([p[0] for p in all_pairs], dtype=torch.long)
        s_idx = torch.tensor([p[1] for p in all_pairs], dtype=torch.long)
        f_mat = torch.stack([fund_emb[int(i)] for i in f_idx])   # (N, D)
        s_mat = stock_emb_mat[s_idx]                              # (N, D)
        scores = (f_mat * s_mat).sum(dim=1).numpy().tolist()

        try:
            qauc = float(roc_auc_score(labels, scores))
            qap = float(average_precision_score(labels, scores))
        except Exception:
            qauc, qap = float("nan"), float("nan")
        per_quarter.append({
            "quarter_idx": qi,
            "quarter_date": quarter_dates[qi],
            "n_pos": n_pos,
            "n_neg": len(neg_pairs),
            "n_unique_funds": len(u_funds),
            "n_unique_stocks": len(u_stocks),
            "auc": qauc,
            "ap": qap,
        })
        print(f"  q{qi} ({quarter_dates[qi]}): n_pos={n_pos} n_neg={len(neg_pairs)} "
              f"AUC={qauc:.4f} AP={qap:.4f}", flush=True)
        pooled_scores.extend(scores)
        pooled_labels.extend(labels)

    pooled_auc = float(roc_auc_score(pooled_labels, pooled_scores))
    pooled_ap = float(average_precision_score(pooled_labels, pooled_scores))
    q_mean_auc = float(np.mean([q["auc"] for q in per_quarter
                                if not np.isnan(q["auc"])]))
    q_mean_ap = float(np.mean([q["ap"] for q in per_quarter
                               if not np.isnan(q["ap"])]))
    overall = {
        "pooled_auc": pooled_auc,
        "pooled_ap": pooled_ap,
        "quarter_mean_auc": q_mean_auc,
        "quarter_mean_ap": q_mean_ap,
        "n_quarters": len(per_quarter),
        "n_pos_total": sum(q["n_pos"] for q in per_quarter),
        "n_neg_total": sum(q["n_neg"] for q in per_quarter),
        "seed": args.seed,
        "neg_ratio": args.neg_ratio,
        "scoring": "inner_product",
        "note": ("Sampled-negative AUC/AP matching pipeline's test_entry_auc/ap "
                 "under NEW_EDGE_STRICT=1 + neg_sampling_ratio=1.0. Negatives "
                 "drawn from unique(pos_funds) × unique(pos_stocks) \\ positives."),
    }
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    with open(out_dir / "per_quarter_metrics.json", "w") as f:
        json.dump(per_quarter, f, indent=2)
    print(f"[eval_s1_sampled] overall:", flush=True)
    print(f"  pooled_auc = {pooled_auc:.4f}", flush=True)
    print(f"  pooled_ap  = {pooled_ap:.4f}", flush=True)
    print(f"  quarter_mean_auc = {q_mean_auc:.4f}", flush=True)
    print(f"  quarter_mean_ap  = {q_mean_ap:.4f}", flush=True)
    print(f"[eval_s1_sampled] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
