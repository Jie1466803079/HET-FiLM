"""Stage-2 eval for the F1-narrow Heterformer comparison.

Trains a small MLP regression head on top of a frozen Stage-1 Heterformer
encoder, predicting edge weight (percent of portfolio = EDGE_WEIGHT_MODE=percent
to match the existing pipeline). Evaluates on the val (q43-q48) and test
(q51-q64) NEW_EDGE_STRICT entry edges. Reports MAE / RMSE / R².

Workflow:
  1. Load Heterformer Stage-1 checkpoint, freeze all weights.
  2. For each unique (fund, stock) in q40 train + val + test edges, encode
     both endpoints once using q40 neighbour context → 768-d each.
  3. Form edge features as concat[h_fund, h_stock] = 1536-d.
  4. Train a small MLP (1536 → 256 → 1) on q40 positive edges with their
     true `percent_tna`-style weights as target, MSE loss.
  5. Predict on val + test edges, report MAE / RMSE / R² per quarter and
     aggregated.
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
import torch.nn as nn

GRAPH_PKL = ("/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v4/"
             "DHGAS/graphs/graphs_v2_portfolio_clean_with_ids_since_2005Q3.pkl")
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
    ap.add_argument(
        "--data_dir",
        default="/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/"
                "sec_filings_project/heterformer_data/"
                "f1_narrow_q40train_q43-48val_q51-64test",
    )
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--pretrain_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--book_neighbour", type=int, default=5)
    ap.add_argument("--shelves_neighbour", type=int, default=5)
    ap.add_argument("--author_neighbour", type=int, default=0)
    ap.add_argument("--heter_embed_size", type=int, default=64)
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--head_epochs", type=int, default=20)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--head_batch_size", type=int, default=512)
    ap.add_argument("--head_dropout", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.ckpt_path)
    pretrain_dir = Path(args.pretrain_dir) if args.pretrain_dir \
        else data_dir / "pretrain_dir"
    out_dir = Path(args.out_dir) if args.out_dir \
        else ckpt_path.parent / "eval_stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    print(f"[stage2] meta protocol={meta.get('protocol')}", flush=True)

    n_book = args.book_neighbour
    n_mta = args.shelves_neighbour + args.author_neighbour + 3

    # ── Load Stage-1 Heterformer encoder (frozen) ─────────────────────────
    sys.path.insert(
        0,
        "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/mutual_fund_prediction/"
        "third_party/heterformer_src",
    )
    from transformers import BertConfig, BertTokenizerFast
    from src.model.HeterformerD import HeterFormerDsForNeighborPredict
    cfg = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model = HeterFormerDsForNeighborPredict.from_pretrained(
        "bert-base-uncased", config=cfg)
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
    for p in model.parameters():
        p.requires_grad_(False)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    hidden = model.config.hidden_size
    print(f"[stage2] loaded encoder, hidden={hidden}", flush=True)

    # ── q40 snapshot + neighbour context ──────────────────────────────────
    print(f"[stage2] loading graph...", flush=True)
    with open(GRAPH_PKL, "rb") as f:
        g = pickle.load(f)
    quarters = sorted(g.keys())
    train_snap = g[quarters[meta["train_quarter_idx"]]]
    train_date = meta["train_quarter_date"]
    fund_id_idx = train_snap["fund"].id_idx
    stock_id_idx = train_snap["stock"].id_idx
    n_funds_q40 = int(fund_id_idx.shape[0])
    n_stocks_q40 = int(stock_id_idx.shape[0])
    fund_gid_to_local = {int(fund_id_idx[i].item()): i for i in range(n_funds_q40)}
    stock_gid_to_local = {int(stock_id_idx[i].item()): i for i in range(n_stocks_q40)}

    # q40 fund text
    print(f"[stage2] loading text json...", flush=True)
    with open(TEXT_JSON) as f:
        text_records = json.load(f)
    fund_gid_to_text: Dict[int, str] = {}
    for r in text_records:
        if str(r.get("timestamp")) == train_date:
            fund_gid_to_text[int(r["id_idx"])] = (r.get("strategy") or "").strip()
    fund_texts: List[str] = []
    for i in range(n_funds_q40):
        gid = int(fund_id_idx[i].item())
        t = fund_gid_to_text.get(gid, "")[:meta.get("max_text_chars", 8192)]
        t = t.replace("\t", " ").replace("$$", " ").replace("\n", " ").replace("\r", " ")
        fund_texts.append(t)

    # q40 holdings + mgmt
    q40_fund_to_stocks: Dict[int, List[int]] = defaultdict(list)
    q40_stock_to_funds: Dict[int, List[int]] = defaultdict(list)
    e = train_snap[("fund", "holds_stock", "stock")].edge_index
    e_attr = train_snap[("fund", "holds_stock", "stock")].edge_attr
    train_edge_weight: Dict[Tuple[int, int], float] = {}
    for k in range(e.shape[1]):
        f, s = int(e[0, k].item()), int(e[1, k].item())
        q40_fund_to_stocks[f].append(s)
        q40_stock_to_funds[s].append(f)
        train_edge_weight[(f, s)] = float(e_attr[k].item())
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

    def shift(x): return x + 1 if x >= 0 else 0

    def build_fund_cols(f_idx: int) -> List[str]:
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
        return ([self_text] + tn_text
                + [str(shift(x)) for x in sh]
                + [str(shift(pub)), "0", "0"])

    def build_stock_cols(s_idx: int) -> List[str]:
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
        return (["" ] + tn_text
                + [str(shift(x)) for x in sh]
                + ["0", "0", "0"])

    @torch.no_grad()
    def encode(cols: List[str]) -> torch.Tensor:
        ids, attn, mta, mask = _encode_columns(
            tok, cols, args.max_length, n_book, n_mta, args.device)
        emb = model.infer(ids.unsqueeze(0), attn.unsqueeze(0),
                          mta.unsqueeze(0), mask.unsqueeze(0))
        return emb.squeeze(0).float().cpu()

    # ── Collect all (fund, stock, weight, quarter, split) tuples ──────────
    # Train: q40 positives (use train_edge_weight)
    # Val/Test: NEW_EDGE_STRICT entries — need to fetch weights from each
    # quarter's edge_attr.

    train_rows: List[Tuple[int, int, float]] = []
    for (f, s), w in train_edge_weight.items():
        train_rows.append((f, s, w))
    print(f"[stage2] train (q40) rows: {len(train_rows)}", flush=True)

    def read_eval_sidecar(path: Path):
        rows: List[Tuple[int, int, int, str]] = []
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                rows.append((int(r["fund_local"]), int(r["stock_local"]),
                             int(r["quarter_idx"]), r["quarter_date"]))
        return rows

    val_rows = read_eval_sidecar(data_dir / "val_index.jsonl")
    test_rows = read_eval_sidecar(data_dir / "test_index.jsonl")
    print(f"[stage2] val/test sidecar rows: {len(val_rows)} val, "
          f"{len(test_rows)} test", flush=True)

    # Fetch true weights for val/test by walking each test quarter once.
    # Map (f_loc_q40, s_loc_q40, qi) -> weight.
    by_quarter: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for f, s, qi, _ in val_rows + test_rows:
        by_quarter[qi].append((f, s))
    weight_lookup: Dict[Tuple[int, int, int], float] = {}
    etype = ("fund", "holds_stock", "stock")
    for qi, _pairs in by_quarter.items():
        snap = g[quarters[qi]]
        if etype not in snap.edge_types:
            continue
        e_i = snap[etype].edge_index
        e_w = snap[etype].edge_attr
        fg = snap["fund"].id_idx
        sg = snap["stock"].id_idx
        # Build (f_gid, s_gid) -> weight for this quarter
        pair_w: Dict[Tuple[int, int], float] = {}
        for k in range(e_i.shape[1]):
            fl_t = int(e_i[0, k].item())
            sl_t = int(e_i[1, k].item())
            if fl_t < fg.shape[0] and sl_t < sg.shape[0]:
                pair_w[(int(fg[fl_t].item()), int(sg[sl_t].item()))] = \
                    float(e_w[k].item())
        # Map q40-local pairs in this quarter to weights
        gid_inv_f = {v: k for k, v in fund_gid_to_local.items()}
        gid_inv_s = {v: k for k, v in stock_gid_to_local.items()}
        for f_loc, s_loc in _pairs:
            f_gid = gid_inv_f.get(f_loc)
            s_gid = gid_inv_s.get(s_loc)
            if f_gid is None or s_gid is None:
                continue
            w = pair_w.get((f_gid, s_gid))
            if w is not None:
                weight_lookup[(f_loc, s_loc, qi)] = w
    print(f"[stage2] resolved weights for {len(weight_lookup)} val+test "
          f"edges (of {len(val_rows) + len(test_rows)} sidecar rows)",
          flush=True)

    # ── Encode unique funds + stocks ──────────────────────────────────────
    fund_locals: set = set()
    stock_locals: set = set()
    for f, s, _ in train_rows:
        fund_locals.add(f); stock_locals.add(s)
    for f, s, _, _ in val_rows + test_rows:
        fund_locals.add(f); stock_locals.add(s)
    print(f"[stage2] encoding {len(fund_locals)} unique funds + "
          f"{len(stock_locals)} unique stocks...", flush=True)
    fund_emb: Dict[int, torch.Tensor] = {}
    stock_emb: Dict[int, torch.Tensor] = {}
    for i, f in enumerate(sorted(fund_locals)):
        fund_emb[f] = encode(build_fund_cols(f))
        if i % 100 == 0:
            print(f"  fund {i+1}/{len(fund_locals)}", flush=True)
    for i, s in enumerate(sorted(stock_locals)):
        stock_emb[s] = encode(build_stock_cols(s))
        if i % 300 == 0:
            print(f"  stock {i+1}/{len(stock_locals)}", flush=True)

    # ── MLP regression head ───────────────────────────────────────────────
    class Head(nn.Module):
        def __init__(self, d_in, d_hid, p):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(p),
                nn.Linear(d_hid, d_hid), nn.GELU(), nn.Dropout(p),
                nn.Linear(d_hid, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    head = Head(2 * hidden, args.head_hidden, args.head_dropout).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr,
                            weight_decay=1e-3)

    def make_xy(rows, weight_map=None):
        X = []
        Y = []
        for r in rows:
            if len(r) == 3:
                f, s, w = r; qi = None
            else:
                f, s, qi, _ = r
                w = weight_map.get((f, s, qi)) if weight_map else None
                if w is None:
                    continue
            if f not in fund_emb or s not in stock_emb:
                continue
            X.append(torch.cat([fund_emb[f], stock_emb[s]]))
            Y.append(w)
        return torch.stack(X), torch.tensor(Y, dtype=torch.float32)

    print(f"[stage2] preparing tensors...", flush=True)
    X_tr, Y_tr = make_xy(train_rows)
    X_val, Y_val = make_xy(val_rows, weight_lookup)
    X_te, Y_te = make_xy(test_rows, weight_lookup)
    print(f"  X_tr={X_tr.shape} Y_tr range=[{Y_tr.min():.4f},{Y_tr.max():.4f}]",
          flush=True)
    print(f"  X_val={X_val.shape} X_te={X_te.shape}", flush=True)

    # ── Train head ────────────────────────────────────────────────────────
    n = X_tr.shape[0]
    best_val = float("inf")
    best_state = None
    for epoch in range(args.head_epochs):
        head.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.head_batch_size):
            idx = perm[i:i + args.head_batch_size]
            x = X_tr[idx].to(args.device)
            y = Y_tr[idx].to(args.device)
            yh = head(x)
            loss = ((yh - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        tr_mse = tot / n
        head.eval()
        with torch.no_grad():
            v_yh = head(X_val.to(args.device)).cpu()
        v_mae = float((v_yh - Y_val).abs().mean())
        v_mse = float(((v_yh - Y_val) ** 2).mean())
        print(f"  epoch {epoch}: train_mse={tr_mse:.6f} val_mae={v_mae:.6f} "
              f"val_mse={v_mse:.6f}", flush=True)
        if v_mse < best_val:
            best_val = v_mse
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)

    # ── Eval ──────────────────────────────────────────────────────────────
    head.eval()
    def metrics(X, Y):
        with torch.no_grad():
            yh = head(X.to(args.device)).cpu().numpy()
        y = Y.numpy()
        mae = float(np.abs(yh - y).mean())
        rmse = float(np.sqrt(((yh - y) ** 2).mean()))
        ss_res = float(((y - yh) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(y))}

    val_m = metrics(X_val, Y_val)
    test_m = metrics(X_te, Y_te)

    # Per-quarter test metrics
    qm: Dict[int, List[int]] = defaultdict(list)
    for i, (f, s, qi, qd) in enumerate(test_rows):
        if (f, s, qi) in weight_lookup and f in fund_emb and s in stock_emb:
            qm[qi].append(i)
    per_q_metrics = []
    # build per-row indices into X_te
    # Reconstruct row index by repeating make_xy logic:
    valid_te_idx = []
    for i, (f, s, qi, qd) in enumerate(test_rows):
        if (f, s, qi) in weight_lookup and f in fund_emb and s in stock_emb:
            valid_te_idx.append((i, qi, qd))
    # Map list-position -> (test_row_idx, qi, qd). The X_te tensor lists them
    # in the same order. So position k of X_te corresponds to valid_te_idx[k].
    qi_of_xte = [v[1] for v in valid_te_idx]
    qd_of_xte = [v[2] for v in valid_te_idx]
    by_q: Dict[int, List[int]] = defaultdict(list)
    for k, qi in enumerate(qi_of_xte):
        by_q[qi].append(k)
    for qi in sorted(by_q.keys()):
        idx = torch.tensor(by_q[qi])
        m = metrics(X_te[idx], Y_te[idx])
        m["quarter_idx"] = qi
        # qd is same for all entries with same qi
        m["quarter_date"] = qd_of_xte[by_q[qi][0]]
        per_q_metrics.append(m)
        print(f"  q{qi} ({m['quarter_date']}): "
              f"n={m['n']} MAE={m['mae']:.6f} RMSE={m['rmse']:.6f} R2={m['r2']:.4f}",
              flush=True)

    out_overall = {
        "val": val_m, "test": test_m,
        "best_val_mse": best_val,
        "head_epochs": args.head_epochs,
        "head_lr": args.head_lr,
        "head_hidden": args.head_hidden,
        "head_dropout": args.head_dropout,
    }
    with open(out_dir / "overall_metrics.json", "w") as f:
        json.dump(out_overall, f, indent=2)
    with open(out_dir / "per_quarter_metrics.json", "w") as f:
        json.dump(per_q_metrics, f, indent=2)
    torch.save(head.state_dict(), out_dir / "stage2_head.pt")
    print(f"[stage2] val: MAE={val_m['mae']:.6f} RMSE={val_m['rmse']:.6f} "
          f"R2={val_m['r2']:.4f} (n={val_m['n']})", flush=True)
    print(f"[stage2] test: MAE={test_m['mae']:.6f} RMSE={test_m['rmse']:.6f} "
          f"R2={test_m['r2']:.4f} (n={test_m['n']})", flush=True)
    print(f"[stage2] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
