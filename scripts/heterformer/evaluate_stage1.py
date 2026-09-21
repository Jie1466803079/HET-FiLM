"""Stage-1 eval for the F1-narrow Heterformer comparison.

Loads a trained Heterformer checkpoint, encodes every q40 fund and every q40
stock once, then for each test quarter scores the cartesian (fund × stock)
product and computes AUC + AP against the NEW_EDGE_STRICT entry-edge labels
(matches the existing pipeline's EXHAUSTIVE_TEST=1 protocol).

Outputs (under <ckpt_dir>/eval_stage1/):
  - per_quarter_metrics.json:  list of {quarter_idx, quarter_date, n_funds,
                                        n_stocks, n_positives, auc, ap}
  - overall_metrics.json:      pooled AUC + AP across all test pairs, plus
                               quarter-averaged values
  - per_fund_metrics.jsonl:    one line per (fund, quarter) with its own
                               AUC/AP (where the fund has >=1 positive +
                               >=1 negative in that quarter)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def _load_q40_columns(data_dir: Path, max_length: int, book_neighbour: int,
                      shelves_neighbour: int, author_neighbour: int):
    """Re-tokenise the train.tsv to recover the q40 self-row for every fund and
    stock that appears as a query/key. We need (input_ids, attention_mask,
    mta_ids, mask) per node — same shape the model consumes during training.
    Returns dicts:
        fund_features[local_idx] = (token_q, attn_q, mta_q, mask_q)
        stock_features[local_idx] = (token_k, attn_k, mta_k, mask_k)
    """
    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_tsv = data_dir / "train.tsv"
    fund_feat: Dict[int, Tuple] = {}
    stock_feat: Dict[int, Tuple] = {}

    n_cols = 1 + book_neighbour + shelves_neighbour + author_neighbour + 3
    sidecar = data_dir / "train_index.jsonl"
    # train.tsv has no sidecar; we infer fund/stock local idx from the meta + a
    # second sidecar that build_funds_data.py does NOT currently emit. To
    # work around without changing the adapter, we instead re-encode each
    # fund/stock when the model needs it via the eval-side sidecar files
    # (val_index.jsonl, test_index.jsonl) → see _encode_node().
    return tok, n_cols  # caller will use tok directly with per-row TSV cols


def _read_tsv_rows(path: Path) -> List[Tuple[List[str], List[str]]]:
    rows: List[Tuple[List[str], List[str]]] = []
    with open(path) as f:
        for line in f:
            q_all, k_all = line.rstrip("\n").split("$$")
            rows.append((q_all.split("\t"), k_all.split("\t")))
    return rows


def _encode_columns(tok, cols: List[str], max_length: int, n_book: int,
                    n_mta: int, device: str):
    """Turn one TSV-side's columns into the four tensors HeterformerD expects."""
    text_cols = cols[: 1 + n_book]
    mta_cols = cols[1 + n_book :]
    assert len(mta_cols) == n_mta, f"mta col count {len(mta_cols)} != {n_mta}"
    enc = tok(
        text_cols, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)               # (1+n_book, L)
    attention = enc["attention_mask"].to(device)
    mta_ids = torch.tensor([int(x) for x in mta_cols],
                           dtype=torch.long, device=device)  # (n_mta,)
    # Mask = 1 where text non-empty for text part, 1 where id != 0 (= shifted
    # sentinel) for mta part. Matches read_process_data_heter's tmp_mask logic.
    text_mask = torch.tensor(
        [1 if c.strip() else 0 for c in text_cols],
        dtype=torch.long, device=device,
    )
    mta_mask = (mta_ids != 0).long()
    node_mask = torch.cat([text_mask, mta_mask], dim=0)  # (1+n_book+n_mta,)
    return input_ids, attention, mta_ids, node_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        default="/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/"
                "sec_filings_project/heterformer_data/"
                "f1_narrow_q40train_q43-48val_q51-64test",
    )
    ap.add_argument("--ckpt_path", required=True,
                    help="Path to the trained Heterformer .pt checkpoint")
    ap.add_argument("--pretrain_dir", default=None,
                    help="Path to pretrain_dir (for textless embeddings). "
                         "Defaults to <data_dir>/pretrain_dir.")
    ap.add_argument("--out_dir", default=None,
                    help="Defaults to <ckpt_dir>/eval_stage1.")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--book_neighbour", type=int, default=5)
    ap.add_argument("--shelves_neighbour", type=int, default=5)
    ap.add_argument("--author_neighbour", type=int, default=0)
    ap.add_argument("--heter_embed_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.ckpt_path)
    pretrain_dir = Path(args.pretrain_dir) if args.pretrain_dir \
        else data_dir / "pretrain_dir"
    out_dir = Path(args.out_dir) if args.out_dir \
        else ckpt_path.parent / "eval_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    n_funds = meta["n_funds_q40"]
    n_stocks = meta["n_stocks_q40"]
    print(f"[eval_stage1] data_dir={data_dir}", flush=True)
    print(f"[eval_stage1] meta: {n_funds} funds, {n_stocks} stocks, "
          f"protocol={meta.get('protocol')}", flush=True)

    n_book = args.book_neighbour
    n_mta = args.shelves_neighbour + args.author_neighbour + 3

    # ── Load model (upstream HeterformerD via vendored src/) ──────────────
    sys.path.insert(
        0,
        "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/mutual_fund_prediction/"
        "third_party/heterformer_src",
    )
    from transformers import BertConfig
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
    print(f"[eval_stage1] loaded checkpoint {ckpt_path}", flush=True)

    # ── Build per-fund and per-stock TSV columns by re-encoding ───────────
    # The train.tsv has each (fund, stock) edge as a separate row. To get a
    # unique encoding per fund (resp. stock), we use the q40 query-side
    # columns from the FIRST row containing each fund-local-idx, and the
    # key-side columns from the FIRST row containing each stock-local-idx.
    # This works because neighbour sampling is random but consistent within a
    # row, and the seed is fixed.
    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # We DON'T have a train_index.jsonl, so we recover fund/stock local idx
    # from the val_index.jsonl + test_index.jsonl (these list the active
    # fund/stock locals appearing in eval). For any fund/stock NOT in eval,
    # we don't need an embedding (it can't appear in eval scoring).
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
    print(f"[eval_stage1] test contains {sum(len(v) for v in edges_by_quarter.values())} edges, "
          f"{len(fund_locals)} unique funds, {len(stock_locals)} unique stocks, "
          f"{len(edges_by_quarter)} quarters", flush=True)

    # Build fund_local -> first train.tsv row that uses it as query
    # And stock_local -> first row that uses it as key
    # We need a sidecar mapping. Since we don't have it, regenerate by
    # re-reading train.tsv and matching shelves-list of fund column 11
    # (= first shelves id, +1 shifted). Hack: we instead derive directly from
    # the snapshot pickle. That's cleaner.
    import pickle
    with open(meta_graph_pkl(args.data_dir), "rb") as f:
        g = pickle.load(f)
    quarters = sorted(g.keys())
    train_snap = g[quarters[meta["train_quarter_idx"]]]
    train_date = meta["train_quarter_date"]

    # Reproduce build_columns() for any fund_local or stock_local using the
    # q40 snapshot. Import build_funds_data.py's helpers.
    from importlib import util as iu
    spec = iu.spec_from_file_location(
        "bfd",
        "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/mutual_fund_prediction/"
        "scripts/heterformer/build_funds_data.py",
    )
    bfd = iu.module_from_spec(spec)
    # Force deterministic neighbour sampling matching the adapter's run
    import random as _random
    _random.seed(meta.get("seed", 42))
    spec.loader.exec_module(bfd)

    # Re-use the adapter's machinery — call main() with same args but we
    # only need build_columns. Easier: re-derive it inline here.
    fund_id_idx = train_snap["fund"].id_idx
    stock_id_idx = train_snap["stock"].id_idx
    n_funds_q40 = int(fund_id_idx.shape[0])
    n_stocks_q40 = int(stock_id_idx.shape[0])
    fund_gid_to_local = {int(fund_id_idx[i].item()): i for i in range(n_funds_q40)}
    stock_gid_to_local = {int(stock_id_idx[i].item()): i for i in range(n_stocks_q40)}

    # Load text for q40
    print(f"[eval_stage1] loading text json...", flush=True)
    with open(
        "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/"
        "sec_filings_project/extracted/"
        "fund_485bpos_sections_temporal_v2_clean_report_anchored_carry.json"
    ) as f:
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

    # Build q40 fund→stocks / stock→funds / fund→mgmt for neighbour sampling
    q40_fund_to_stocks: Dict[int, List[int]] = defaultdict(list)
    q40_stock_to_funds: Dict[int, List[int]] = defaultdict(list)
    e = train_snap[("fund", "holds_stock", "stock")].edge_index
    for k in range(e.shape[1]):
        f, s = int(e[0, k].item()), int(e[1, k].item())
        q40_fund_to_stocks[f].append(s)
        q40_stock_to_funds[s].append(f)
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
        sh = pad_to_k(list(q40_fund_to_stocks.get(f_idx, []))[: args.shelves_neighbour], args.shelves_neighbour)
        pub = fund_to_mgmt.get(f_idx, -1)
        def shift(x): return x + 1 if x >= 0 else 0
        cols = (
            [self_text] + tn_text
            + [str(shift(x)) for x in sh]
            + [str(shift(pub)), "0", "0"]
        )
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
        cols = (
            [self_text] + tn_text
            + [str(shift(x)) for x in sh]
            + ["0", "0", "0"]
        )
        return cols

    @torch.no_grad()
    def encode(cols: List[str]) -> torch.Tensor:
        ids, attn, mta, mask = _encode_columns(
            tok, cols, args.max_length, n_book, n_mta, args.device,
        )
        ids = ids.unsqueeze(0)            # (1, 1+n_book, L)
        attn = attn.unsqueeze(0)
        mta = mta.unsqueeze(0)            # (1, n_mta)
        mask = mask.unsqueeze(0)          # (1, 1+n_book+n_mta)
        emb = model.infer(ids, attn, mta, mask)  # (1, hidden)
        return emb.squeeze(0).float().cpu()

    # ── Encode all unique funds + all stocks ──────────────────────────────
    print(f"[eval_stage1] encoding {len(fund_locals)} funds + {n_stocks_q40} stocks...",
          flush=True)
    fund_emb: Dict[int, torch.Tensor] = {}
    for i, f_loc in enumerate(sorted(fund_locals)):
        fund_emb[f_loc] = encode(build_fund_columns(f_loc))
        if i % 50 == 0:
            print(f"  fund {i+1}/{len(fund_locals)}", flush=True)
    stock_emb_mat = torch.zeros(n_stocks_q40, fund_emb[next(iter(fund_emb))].shape[0])
    for s_loc in range(n_stocks_q40):
        stock_emb_mat[s_loc] = encode(build_stock_columns(s_loc))
        if s_loc % 200 == 0:
            print(f"  stock {s_loc+1}/{n_stocks_q40}", flush=True)

    # ── Per-quarter exhaustive scoring + AUC/AP ───────────────────────────
    from sklearn.metrics import roc_auc_score, average_precision_score
    per_quarter = []
    per_fund_rows = []
    pooled_scores: List[float] = []
    pooled_labels: List[int] = []
    for qi in sorted(edges_by_quarter.keys()):
        pos_pairs = set(edges_by_quarter[qi])
        funds_in_q = sorted({f for f, _ in pos_pairs})
        n_pos = len(pos_pairs)
        q_scores: List[float] = []
        q_labels: List[int] = []
        n_fund_metrics = 0
        for f_loc in funds_in_q:
            femb = fund_emb[f_loc].unsqueeze(0)            # (1, D)
            scores = (femb @ stock_emb_mat.T).squeeze(0).numpy()  # (n_stocks,)
            labels = np.zeros(n_stocks_q40, dtype=np.int64)
            for s_loc in range(n_stocks_q40):
                if (f_loc, s_loc) in pos_pairs:
                    labels[s_loc] = 1
            if labels.sum() >= 1 and labels.sum() < n_stocks_q40:
                try:
                    fauc = float(roc_auc_score(labels, scores))
                    fap = float(average_precision_score(labels, scores))
                    per_fund_rows.append({
                        "quarter_idx": qi, "fund_local": f_loc,
                        "n_pos": int(labels.sum()), "auc": fauc, "ap": fap,
                    })
                    n_fund_metrics += 1
                except Exception:
                    pass
            q_scores.extend(scores.tolist())
            q_labels.extend(labels.tolist())
        try:
            qauc = float(roc_auc_score(q_labels, q_scores))
            qap = float(average_precision_score(q_labels, q_scores))
        except Exception:
            qauc, qap = float("nan"), float("nan")
        per_quarter.append({
            "quarter_idx": qi,
            "quarter_date": quarter_dates[qi],
            "n_funds": len(funds_in_q),
            "n_stocks": n_stocks_q40,
            "n_positives": n_pos,
            "n_fund_metrics": n_fund_metrics,
            "auc": qauc,
            "ap": qap,
        })
        print(f"  q{qi} ({quarter_dates[qi]}): "
              f"n_funds={len(funds_in_q)} n_pos={n_pos} AUC={qauc:.4f} AP={qap:.4f}",
              flush=True)
        pooled_scores.extend(q_scores)
        pooled_labels.extend(q_labels)

    # ── Pooled + averaged ─────────────────────────────────────────────────
    pooled_auc = float(roc_auc_score(pooled_labels, pooled_scores))
    pooled_ap = float(average_precision_score(pooled_labels, pooled_scores))
    q_mean_auc = float(np.mean([q["auc"] for q in per_quarter
                                if not np.isnan(q["auc"])]))
    q_mean_ap = float(np.mean([q["ap"] for q in per_quarter
                               if not np.isnan(q["ap"])]))
    fund_mean_auc = float(np.mean([r["auc"] for r in per_fund_rows]))
    fund_mean_ap = float(np.mean([r["ap"] for r in per_fund_rows]))

    overall = {
        "pooled_auc": pooled_auc,
        "pooled_ap": pooled_ap,
        "quarter_mean_auc": q_mean_auc,
        "quarter_mean_ap": q_mean_ap,
        "fund_mean_auc": fund_mean_auc,
        "fund_mean_ap": fund_mean_ap,
        "n_per_fund_metrics": len(per_fund_rows),
        "n_quarters": len(per_quarter),
    }
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(overall, f, indent=2)
    with open(out_dir / "per_quarter_metrics.json", "w") as f:
        json.dump(per_quarter, f, indent=2)
    with open(out_dir / "per_fund_metrics.jsonl", "w") as f:
        for r in per_fund_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[eval_stage1] overall:", flush=True)
    print(f"  pooled_auc = {pooled_auc:.4f}", flush=True)
    print(f"  pooled_ap  = {pooled_ap:.4f}", flush=True)
    print(f"  quarter_mean_auc = {q_mean_auc:.4f}", flush=True)
    print(f"  quarter_mean_ap  = {q_mean_ap:.4f}", flush=True)
    print(f"  fund_mean_auc    = {fund_mean_auc:.4f}", flush=True)
    print(f"  fund_mean_ap     = {fund_mean_ap:.4f}", flush=True)
    print(f"[eval_stage1] wrote {out_dir}", flush=True)
    return 0


def meta_graph_pkl(_data_dir_unused: str) -> str:
    # Hard-coded to match build_funds_data.py default.
    return ("/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v4/"
            "DHGAS/graphs/graphs_v2_portfolio_clean_with_ids_since_2005Q3.pkl")


if __name__ == "__main__":
    sys.exit(main())
