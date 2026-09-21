"""Build Canadian text-embedding H5 mirroring the US v2-RA-carry schema.

Pipeline:
    1. Load fundno -> unified_id (from canada_fundno_to_unified_id.pkl.gz)
    2. Load sedar_fund_strategy.jsonl, translate to (unified_id, quarter_idx)
    3. Embed unique strategy_text via OpenAI text-embedding-3-large @ 1024-d
    4. Forward-fill across all 44 quarters for each fund (carry policy, unbounded)
    5. Write H5 with US-compatible schema: strategy_emb, risk_emb (zeros, per
       job 8315821 which uses --risk_weight 0.0), has_text, delta_t, cosine_sim,
       fund_ids, snapshot_quarters.

OpenAI API key MUST be in env (OPENAI_API_KEY). Outbound HTTPS required.

Output: embeddings/prospectus_embeddings_canada_carry.h5  (~1.3 GB)
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SNAP = Path(__file__).resolve().parent
JSONL = SNAP / "sedar_probe/sedar_coverage/sedar_fund_strategy.jsonl"
MAP   = SNAP / "canada_fundno_to_unified_id.pkl.gz"
OUT_DIR = SNAP / "embeddings"
OUT_DIR.mkdir(exist_ok=True)
OUT   = OUT_DIR / "prospectus_embeddings_canada_carry.h5"
CACHE = OUT_DIR / "_emb_cache.jsonl"   # appended on each batch — resume on crash

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM   = 1024
MAX_TOKENS_PER_TEXT  = 8000    # API hard limit 8191; trim for safety
APPROX_CHARS_PER_TOKEN = 4
MAX_CHARS_PER_TEXT   = MAX_TOKENS_PER_TEXT * APPROX_CHARS_PER_TOKEN
BATCH_SIZE = 64                # texts per request


def quarter_idx_from_eff_q(eff_q: str, q_to_idx: dict) -> int:
    """ '2015-03-31' -> '2015Q1' -> 0 """
    y, m, _ = eff_q.split("-")
    qn = (int(m) - 1) // 3 + 1
    return q_to_idx.get(f"{y}Q{qn}", -1)


def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def load_cache() -> dict[str, list[float]]:
    """Return {text_hash: embedding_vector} from on-disk cache."""
    cache: dict[str, list[float]] = {}
    if CACHE.exists():
        with open(CACHE, "r") as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["h"]] = rec["v"]
    return cache


def append_cache(items: list[tuple[str, list[float]]]) -> None:
    with open(CACHE, "a") as f:
        for h, v in items:
            f.write(json.dumps({"h": h, "v": v}) + "\n")


def embed_batch(texts: list[str], retries: int = 5) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch. Backoff on 429/5xx."""
    import urllib.request, urllib.error
    key = os.environ["OPENAI_API_KEY"]
    body = json.dumps({
        "model": EMBED_MODEL,
        "input": texts,
        "dimensions": EMBED_DIM,
    }).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/embeddings",
                data=body, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            return [item["embedding"] for item in d["data"]]
        except urllib.error.HTTPError as e:
            wait = min(60, 2 ** attempt)
            print(f"  HTTP {e.code} attempt {attempt+1}: sleeping {wait}s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            wait = min(60, 2 ** attempt)
            print(f"  URLError attempt {attempt+1}: {e}: sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("OpenAI embeddings failed after retries")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true",
                    help="Run universe match + H5 skeleton, skip API calls")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="Override the input strategy JSONL path. Default: "
                         "sedar_probe/sedar_coverage/sedar_fund_strategy.jsonl.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Override the output H5 path. Default: "
                         "embeddings/prospectus_embeddings_canada_carry.h5. "
                         "The meta JSON is derived as OUT.parent/{OUT.stem}_meta.json.")
    args = ap.parse_args()
    # Overrides — leave module-level defaults intact when unset (preserves the
    # baseline artifact path exactly).
    global JSONL, OUT
    if args.jsonl is not None:
        JSONL = args.jsonl
    if args.output is not None:
        OUT = args.output

    print("[1/6] Loading fundno->unified mapping...", flush=True)
    with gzip.open(MAP, "rb") as f:
        m = pickle.load(f)
    fundno_to_unified: dict[int, int] = m["fundno_to_unified"]
    global_funds: list[int] = m["global_funds"]
    N = len(global_funds)
    print(f"      universe N={N:,}", flush=True)

    # 44 snapshot quarters: 2015Q1..2025Q4
    quarters = []
    for y in range(2015, 2026):
        for q in (1, 2, 3, 4):
            quarters.append(f"{y}Q{q}")
    assert len(quarters) == 44
    Q = len(quarters)
    q_to_idx = {q: i for i, q in enumerate(quarters)}

    print("[2/6] Loading sedar_fund_strategy.jsonl + translating keys...", flush=True)
    records: list[dict] = []
    skipped_no_fundno = skipped_no_quarter = skipped_empty = 0
    with open(JSONL) as f:
        for line in f:
            r = json.loads(line)
            fn = r.get("fundno")
            if fn is None or fn not in fundno_to_unified:
                skipped_no_fundno += 1
                continue
            qi = quarter_idx_from_eff_q(r["effective_q"], q_to_idx)
            if qi < 0:
                skipped_no_quarter += 1
                continue
            text = (r.get("strategy_text") or "").strip()
            if not text or r.get("strategy_chars", 0) == 0:
                skipped_empty += 1
                continue
            records.append({
                "uid": fundno_to_unified[fn],
                "qi":  qi,
                "text": text[:MAX_CHARS_PER_TEXT],
                "hash": text_hash(text[:MAX_CHARS_PER_TEXT]),
            })
    print(f"      kept={len(records):,}  skip_unfundno={skipped_no_fundno:,}  "
          f"skip_quarter={skipped_no_quarter:,}  skip_empty={skipped_empty:,}", flush=True)

    # Dedup texts for embedding cost
    uniq_hashes = {r["hash"]: r["text"] for r in records}
    print(f"      unique strategy texts to embed: {len(uniq_hashes):,}", flush=True)

    # Resume from cache
    cache = load_cache()
    print(f"[3/6] Embedding cache: {len(cache):,} hits already", flush=True)
    to_embed = [(h, t) for h, t in uniq_hashes.items() if h not in cache]
    print(f"      remaining to embed: {len(to_embed):,}", flush=True)

    if args.dry_run:
        print("      DRY RUN — skipping API calls", flush=True)
        for h, _ in to_embed:
            cache[h] = [0.0] * EMBED_DIM
    else:
        n_done = 0
        t0 = time.time()
        for i in range(0, len(to_embed), BATCH_SIZE):
            batch = to_embed[i : i + BATCH_SIZE]
            embs = embed_batch([t for _, t in batch])
            assert len(embs) == len(batch)
            items = [(h, e) for (h, _), e in zip(batch, embs)]
            append_cache(items)
            for h, e in items:
                cache[h] = e
            n_done += len(batch)
            rate = n_done / max(1, time.time() - t0)
            eta = (len(to_embed) - n_done) / max(rate, 1e-6)
            print(f"      embedded {n_done:,}/{len(to_embed):,}  "
                  f"rate={rate:.1f}/s  eta={eta/60:.1f}m", flush=True)

    # Covered-only universe: drop the 2,536 funds with zero text. Loader's
    # _lookup_rows treats fund_ids missing from the H5 LUT as cold-start, so
    # this is behaviour-equivalent but doesn't dilute coverage optics.
    covered_uids = sorted({r["uid"] for r in records})
    uid_to_row   = {uid: i for i, uid in enumerate(covered_uids)}
    N_covered    = len(covered_uids)
    print(f"[4/6] Materializing N_covered x Q x D  (covered N={N_covered:,} of universe {N:,})...", flush=True)
    strategy_emb = np.zeros((N_covered, Q, EMBED_DIM), dtype=np.float32)
    has_text     = np.zeros((N_covered, Q), dtype=bool)
    # Stamp fresh embeddings
    for r in records:
        e = cache.get(r["hash"])
        if e is None:
            continue
        v = np.asarray(e, dtype=np.float32)
        row = uid_to_row[r["uid"]]
        # If the same (uid, qi) had multiple records (shouldn't, but safe), last wins
        strategy_emb[row, r["qi"]] = v
        has_text[row, r["qi"]] = True

    print("[5/6] Carry-forward (unbounded max_lag)...", flush=True)
    delta_t = np.full((N_covered, Q), -1, dtype=np.int8)
    cosine_sim = np.zeros((N_covered, Q), dtype=np.float32)
    for u in range(N_covered):
        last_emb = None
        last_q   = None
        for q in range(Q):
            if has_text[u, q]:
                cur = strategy_emb[u, q]
                if last_emb is not None:
                    a = cur / max(1e-12, np.linalg.norm(cur))
                    b = last_emb / max(1e-12, np.linalg.norm(last_emb))
                    cosine_sim[u, q] = float(np.dot(a, b))
                delta_t[u, q] = 0
                last_emb = cur
                last_q   = q
            elif last_emb is not None:
                strategy_emb[u, q] = last_emb
                delta_t[u, q] = q - last_q
                # carried cells: cosine_sim left at 0
            # else: cold-start; strategy_emb stays zero, delta_t stays -1

    # Coverage stats (over the covered subset of N_covered × Q cells)
    n_fresh = int(has_text.sum())
    n_carry = int(((delta_t > 0) & (delta_t != -1)).sum())
    n_cold  = int((delta_t == -1).sum())
    total   = N_covered * Q
    universe_total = N * Q
    universe_cold  = (N - N_covered) * Q  # the uncovered funds the loader treats as cold
    print(f"      [covered subset {N_covered:,} × {Q}] fresh={n_fresh:,}  carry={n_carry:,}  cold={n_cold:,}  total={total:,}", flush=True)
    print(f"      pct_fresh={100*n_fresh/total:.2f}  "
          f"pct_carry={100*n_carry/total:.2f}  "
          f"pct_cold={100*n_cold/total:.2f}", flush=True)
    print(f"      [full universe {N:,} × {Q}] uncovered {N-N_covered:,} funds become cold via LUT fallthrough at load time", flush=True)

    print(f"[6/6] Writing H5 -> {OUT}", flush=True)
    risk_emb = np.zeros_like(strategy_emb)  # job 8315821 sets --risk_weight 0.0
    fund_ids_bytes = np.array([str(u).encode("ascii") for u in covered_uids], dtype="O")
    snap_q_bytes   = np.array([q.encode("ascii") for q in quarters], dtype="O")

    tmp = OUT.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("strategy_emb", data=strategy_emb, compression="gzip", compression_opts=4)
        f.create_dataset("risk_emb",     data=risk_emb,     compression="gzip", compression_opts=4)
        f.create_dataset("has_text",     data=has_text)
        f.create_dataset("delta_t",      data=delta_t)
        f.create_dataset("cosine_sim",   data=cosine_sim)
        dt_s = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("fund_ids",          data=fund_ids_bytes, dtype=dt_s)
        f.create_dataset("snapshot_quarters", data=snap_q_bytes,   dtype=dt_s)
        f.attrs["anchor"]        = "report_dt"
        f.attrs["anchor_policy"] = "carry"
        f.attrs["fallback"]      = "carry_forward"
        f.attrs["max_lag"]       = Q
        f.attrs["sections"]      = "strategy"
        f.attrs["source_jsonl"]  = str(JSONL.relative_to(SNAP))
        f.attrs["encoder"]       = EMBED_MODEL
        f.attrs["embedding_dim"] = EMBED_DIM
        f.attrs["n_funds"]       = N_covered
        f.attrs["universe_n_funds"] = N
        f.attrs["n_quarters"]    = Q
        f.attrs["coverage_policy"] = "covered_only (uncovered universe funds fall through loader LUT as cold-start)"
        f.attrs["risk_emb_policy"] = "zeros_only (Canada has no risk text; use --risk_weight 0.0)"
    tmp.rename(OUT)
    print(f"Wrote {OUT}  size={OUT.stat().st_size / 1024**2:.0f} MB", flush=True)

    # Side metadata
    meta = {
        "encoder_name": EMBED_MODEL,
        "embedding_dim": EMBED_DIM,
        "n_funds_h5":           N_covered,
        "n_funds_universe":     N,
        "n_funds_uncovered":    N - N_covered,
        "n_quarters":           Q,
        "n_fresh":              n_fresh,
        "n_carry":              n_carry,
        "n_cold_in_h5":         n_cold,
        "n_cold_via_lut_fallthrough": universe_cold,
        "covered_subset_pct_fresh": 100*n_fresh/total,
        "covered_subset_pct_carry": 100*n_carry/total,
        "covered_subset_pct_cold":  100*n_cold/total,
        "universe_pct_fresh":   100*n_fresh/universe_total,
        "universe_pct_carry":   100*n_carry/universe_total,
        "universe_pct_cold":    100*(n_cold + universe_cold)/universe_total,
        "policy": "carry",
        "max_lag_effective": Q,
        "coverage_policy": "covered_only",
        "source_jsonl": str(JSONL.relative_to(SNAP)),
        "fundno_map":   str(MAP.relative_to(SNAP)),
    }
    # Derive meta path from OUT — identical to the literal for the default
    # (baseline) run since OUT.stem == 'prospectus_embeddings_canada_carry'.
    meta_path = OUT.parent / f"{OUT.stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
