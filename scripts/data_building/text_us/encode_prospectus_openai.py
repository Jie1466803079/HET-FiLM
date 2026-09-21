#!/usr/bin/env python3
"""Encode 485BPOS prospectus sections using OpenAI text-embedding-3-large.

Sibling to encode_prospectus_json.py (which uses sec-bert). Reads the same
cleaned JSON, applies the same LOCF + per-source-quarter cache, but calls
the OpenAI embeddings API with retry/backoff and writes a 1024-dim HDF5 to
embeddings_openai/.

Usage:
    export OPENAI_API_KEY="sk-..."
    python mutual_fund_prediction/scripts/encode_prospectus_openai.py \\
        --json_path sec_filings_project/extracted/fund_485bpos_sections_temporal.json \\
        --output_dir sec_filings_project/embeddings_openai \\
        --yes
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import tiktoken
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    BadRequestError,
    InternalServerError,
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "text-embedding-3-large"
EMB_DIM = 1024
MAX_TOKENS = 8000        # safety margin under 8191 hard limit
CHUNK_OVERLAP = 200
COST_PER_1M_TOKENS = 0.13  # USD

SECTIONS: tuple[str, ...] = ("strategy", "risk")
SECTION_WEIGHTS: dict[str, float] = {"strategy": 0.625, "risk": 0.375}
# Sections written to HDF5 schema (objective is kept as zero tensor for backward
# compatibility with the existing sec-bert pipeline's H5 layout, but it is NOT
# encoded — analysis showed objective is mostly redundant boilerplate, see
# docs/plans/2026-04-06-openai-prospectus-embeddings-design.md).
H5_SECTIONS: tuple[str, ...] = ("strategy", "risk", "objective")

API_BATCH_MAX_INPUTS = 100
API_BATCH_MAX_TOKENS = 250_000


def _build_quarter_list() -> list[str]:
    quarters: list[str] = []
    for year in range(2005, 2022):
        for q in range(1, 5):
            if (year == 2005 and q < 3) or (year == 2021 and q > 3):
                continue
            quarters.append(f"{year}Q{q}")
    return quarters


ALL_QUARTERS: list[str] = _build_quarter_list()
QUARTER_TO_IDX: dict[str, int] = {q: i for i, q in enumerate(ALL_QUARTERS)}
N_QUARTERS: int = len(ALL_QUARTERS)
assert N_QUARTERS == 65


def date_to_quarter(date_str: str) -> str:
    y, m, _ = date_str.split("-")
    q = (int(m) - 1) // 3 + 1
    return f"{y}Q{q}"


@dataclass
class Config:
    json_path: Path
    output_dir: Path
    max_lag: int = 6
    resume: bool = False
    max_funds: int = 0
    dry_run_tokens: bool = False
    yes: bool = False

    @property
    def temp_dir(self) -> Path:
        return self.output_dir / "temp_npy"


# ---------------------------------------------------------------------------
# OpenAI encoder
# ---------------------------------------------------------------------------

@dataclass
class TokenStats:
    total_tokens: int = 0
    n_requests: int = 0
    n_chunked_sections: int = 0


class OpenAIProspectusEncoder:
    def __init__(self, api_key: str, logger: logging.Logger) -> None:
        self.client = OpenAI(api_key=api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.logger = logger
        self.stats = TokenStats()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text or ""))

    def chunk_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= MAX_TOKENS:
            return [text]
        stride = MAX_TOKENS - CHUNK_OVERLAP
        chunks: list[str] = []
        for i in range(0, len(tokens), stride):
            piece = tokens[i : i + MAX_TOKENS]
            decoded = self.tokenizer.decode(piece)
            # BPE round-trip can produce a slightly different token count after decode.
            # Defensively re-encode and trim to stay under the 8191 hard limit.
            re_encoded = self.tokenizer.encode(decoded)
            if len(re_encoded) > 8191:
                # Trim by token, leaving safety margin
                decoded = self.tokenizer.decode(re_encoded[:8000])
            chunks.append(decoded)
            if i + MAX_TOKENS >= len(tokens):
                break
        return chunks

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)),
        wait=wait_exponential(multiplier=1, min=1, max=32),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _api_call(self, inputs: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(
            model=MODEL_NAME,
            input=inputs,
            dimensions=EMB_DIM,
        )
        self.stats.n_requests += 1
        self.stats.total_tokens += resp.usage.total_tokens
        return [d.embedding for d in resp.data]

    def encode_inputs_batched(self, inputs: list[str]) -> list[np.ndarray]:
        if not inputs:
            return []
        results: list[np.ndarray] = []
        batch: list[str] = []
        batch_tokens = 0
        for text in inputs:
            n = self.count_tokens(text)
            # Defense in depth: a single input should never exceed the per-request token cap,
            # because chunk_text caps individual chunks at MAX_TOKENS=8000 and the cap is 250K.
            # If this fires, something upstream is broken.
            assert n <= API_BATCH_MAX_TOKENS, (
                f"Single input has {n} tokens > API_BATCH_MAX_TOKENS={API_BATCH_MAX_TOKENS}; "
                f"chunk_text invariant violated"
            )
            if batch and (
                len(batch) >= API_BATCH_MAX_INPUTS or batch_tokens + n > API_BATCH_MAX_TOKENS
            ):
                embs = self._api_call(batch)
                results.extend(np.asarray(e, dtype=np.float32) for e in embs)
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += n
        if batch:
            embs = self._api_call(batch)
            results.extend(np.asarray(e, dtype=np.float32) for e in embs)
        return results

    def encode_section(self, text: Optional[str]) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(EMB_DIM, dtype=np.float32)
        try:
            chunks = self.chunk_text(text)
            if len(chunks) > 1:
                self.stats.n_chunked_sections += 1
            embs = self.encode_inputs_batched(chunks)
            if not embs:
                return np.zeros(EMB_DIM, dtype=np.float32)
            if len(embs) == 1:
                return embs[0]
            # Length-weighted mean: each chunk weighted by its token count.
            # Without weighting, a short tail chunk (e.g. 300 tokens after a full
            # 8000-token chunk) would get equal weight, biasing toward the end.
            weights = np.asarray(
                [self.count_tokens(c) for c in chunks], dtype=np.float32
            )
            weights = weights / weights.sum()
            stacked = np.stack(embs)  # (n_chunks, EMB_DIM)
            return (weights[:, None] * stacked).sum(axis=0).astype(np.float32)
        except BadRequestError as exc:
            self.logger.error(f"BadRequestError on section: {exc}")
            return np.zeros(EMB_DIM, dtype=np.float32)
        except Exception as exc:
            self.logger.error(f"Unrecoverable encode_section error: {exc}", exc_info=True)
            return np.zeros(EMB_DIM, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class FundTextData:
    fund_texts: dict[int, dict[int, dict[str, Optional[str]]]]
    all_fund_ids: list[int]
    funds_with_text: set[int]

    @staticmethod
    def from_json(json_path: Path, logger: logging.Logger) -> "FundTextData":
        logger.info(f"Loading JSON: {json_path}")
        with open(json_path) as f:
            raw: list[dict] = json.load(f)
        logger.info(f"  {len(raw)} entries")

        fund_texts: dict[int, dict[int, dict[str, Optional[str]]]] = {}
        max_id_idx = 0
        for entry in raw:
            fid: int = entry["id_idx"]
            if fid > max_id_idx:
                max_id_idx = fid
            quarter = date_to_quarter(entry["timestamp"])
            if quarter not in QUARTER_TO_IDX:
                continue
            qi = QUARTER_TO_IDX[quarter]
            if fid not in fund_texts:
                fund_texts[fid] = {}
            if qi in fund_texts[fid]:
                old_len = sum(len(fund_texts[fid][qi].get(s) or "") for s in SECTIONS)
                new_len = sum(len(entry.get(s) or "") for s in SECTIONS)
                if new_len <= old_len:
                    continue
            fund_texts[fid][qi] = {s: entry.get(s) for s in SECTIONS}

        n_fund_nodes = max_id_idx + 1
        return FundTextData(
            fund_texts=fund_texts,
            all_fund_ids=list(range(n_fund_nodes)),
            funds_with_text=set(fund_texts.keys()),
        )


def apply_locf(fund_texts, fund_id, max_lag):
    raw = fund_texts.get(fund_id, {})
    result = []
    empty = {s: None for s in SECTIONS}
    for qi in range(N_QUARTERS):
        if qi in raw:
            result.append((raw[qi], 0, True))
        else:
            found = False
            for lag in range(1, max_lag + 1):
                prev = qi - lag
                if prev < 0:
                    break
                if prev in raw:
                    result.append((raw[prev], lag, False))
                    found = True
                    break
            if not found:
                result.append((empty, -1, False))
    return result


def process_fund(
    fund_id: int,
    fund_data: FundTextData,
    encoder: OpenAIProspectusEncoder,
    cfg: Config,
) -> dict[str, np.ndarray]:
    locf = apply_locf(fund_data.fund_texts, fund_id, cfg.max_lag)

    # H5_SECTIONS includes objective (kept as zeros for schema compat); SECTIONS
    # is the subset that's actually encoded.
    sec_emb = {s: np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32) for s in H5_SECTIONS}
    abs_emb = np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32)
    delta_emb = np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32)
    cosine_sim = np.zeros(N_QUARTERS, dtype=np.float32)
    delta_t = np.zeros(N_QUARTERS, dtype=np.int8)
    has_text = np.zeros(N_QUARTERS, dtype=bool)

    cache: dict[tuple[int, str], np.ndarray] = {}

    for qi, (sections, dt, ht) in enumerate(locf):
        delta_t[qi] = dt
        has_text[qi] = ht
        if dt == -1:
            continue
        src_q = qi - dt
        for sec in SECTIONS:
            key = (src_q, sec)
            if key not in cache:
                cache[key] = encoder.encode_section(sections.get(sec))
            sec_emb[sec][qi] = cache[key]
        abs_emb[qi] = sum(
            SECTION_WEIGHTS[s] * sec_emb[s][qi] for s in SECTIONS
        )

    for qi in range(1, N_QUARTERS):
        curr, prev = abs_emb[qi], abs_emb[qi - 1]
        delta_emb[qi] = curr - prev
        nc, np_ = np.linalg.norm(curr), np.linalg.norm(prev)
        if nc > 0 and np_ > 0:
            cosine_sim[qi] = float(np.dot(curr, prev) / (nc * np_))

    return {
        "strategy_emb": sec_emb["strategy"],
        "risk_emb": sec_emb["risk"],
        "objective_emb": sec_emb["objective"],
        "abs_emb": abs_emb,
        "delta_emb": delta_emb,
        "cosine_sim": cosine_sim,
        "delta_t": delta_t,
        "has_text": has_text,
    }


# ---------------------------------------------------------------------------
# Temp file I/O
# ---------------------------------------------------------------------------

def save_fund_temp(fund_id: int, result: dict[str, np.ndarray], temp_dir: Path) -> None:
    """Atomic write: tmp file -> os.replace, so a kill mid-write can't corrupt the resume cache."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    final_path = temp_dir / f"fund_{fund_id}.npz"
    tmp_path = temp_dir / f"fund_{fund_id}.tmp.npz"
    np.savez_compressed(tmp_path, **result)
    os.replace(tmp_path, final_path)


def load_fund_temp(fund_id: int, temp_dir: Path) -> Optional[dict[str, np.ndarray]]:
    path = temp_dir / f"fund_{fund_id}.npz"
    if path.exists():
        data = np.load(path)
        return {k: data[k] for k in data.files}
    return None


ZERO_RESULT: dict[str, np.ndarray] = {
    "strategy_emb": np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32),
    "risk_emb": np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32),
    "objective_emb": np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32),
    "abs_emb": np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32),
    "delta_emb": np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32),
    "cosine_sim": np.zeros(N_QUARTERS, dtype=np.float32),
    "delta_t": np.full(N_QUARTERS, -1, dtype=np.int8),
    "has_text": np.zeros(N_QUARTERS, dtype=bool),
}


# ---------------------------------------------------------------------------
# HDF5 + metadata
# ---------------------------------------------------------------------------

def merge_to_h5(all_fund_ids, temp_dir, output_dir, logger):
    n = len(all_fund_ids)
    h5_path = output_dir / "prospectus_embeddings.h5"
    str_dt = h5py.string_dtype()
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("fund_ids", data=[str(i) for i in all_fund_ids], dtype=str_dt)
        h5.create_dataset("snapshot_quarters", data=ALL_QUARTERS, dtype=str_dt)
        ds = {
            "abs_emb":       h5.create_dataset("abs_emb",       (n, N_QUARTERS, EMB_DIM), dtype="float32"),
            "strategy_emb":  h5.create_dataset("strategy_emb",  (n, N_QUARTERS, EMB_DIM), dtype="float32"),
            "risk_emb":      h5.create_dataset("risk_emb",      (n, N_QUARTERS, EMB_DIM), dtype="float32"),
            "objective_emb": h5.create_dataset("objective_emb", (n, N_QUARTERS, EMB_DIM), dtype="float32"),
            "delta_emb":     h5.create_dataset("delta_emb",     (n, N_QUARTERS, EMB_DIM), dtype="float32"),
            "cosine_sim":    h5.create_dataset("cosine_sim",    (n, N_QUARTERS), dtype="float32"),
            "delta_t":       h5.create_dataset("delta_t",       (n, N_QUARTERS), dtype="int8"),
            "has_text":      h5.create_dataset("has_text",      (n, N_QUARTERS), dtype="bool"),
        }
        for i, fid in enumerate(tqdm(all_fund_ids, desc="Writing HDF5")):
            r = load_fund_temp(fid, temp_dir)
            if r is None:
                ds["delta_t"][i, :] = -1
                continue
            for key in ds:
                ds[key][i] = r[key]
    logger.info(f"Saved {h5_path} ({h5_path.stat().st_size / 1e9:.2f} GB)")


def save_metadata(all_fund_ids, fund_data, temp_dir, output_dir, encoder, cfg, logger):
    total = real = locf = cold = 0
    cos_consec: list[float] = []
    for fid in all_fund_ids:
        r = load_fund_temp(fid, temp_dir)
        if r is None:
            cold += N_QUARTERS
            total += N_QUARTERS
            continue
        dt, ht, cos = r["delta_t"], r["has_text"], r["cosine_sim"]
        for q in range(N_QUARTERS):
            total += 1
            if dt[q] == -1:
                cold += 1
            elif ht[q]:
                real += 1
            else:
                locf += 1
        for q in range(1, N_QUARTERS):
            if ht[q] and ht[q - 1] and cos[q] != 0.0:
                cos_consec.append(float(cos[q]))

    mean_cos = float(np.mean(cos_consec)) if cos_consec else 0.0
    marker = cfg.json_path.parent / "_clean_verified.txt"
    clean_verified_at = "unknown"
    if marker.exists():
        lines = marker.read_text().splitlines()
        if lines:
            clean_verified_at = lines[0]

    meta = {
        "encoder_name": MODEL_NAME,
        "embedding_dim": EMB_DIM,
        "encoding_date": datetime.datetime.now().isoformat(),
        "section_weights": SECTION_WEIGHTS,
        "max_lag": cfg.max_lag,
        "tokens_billed": encoder.stats.total_tokens,
        "cost_usd_estimated": round(encoder.stats.total_tokens / 1e6 * COST_PER_1M_TOKENS, 4),
        "n_api_requests": encoder.stats.n_requests,
        "n_chunked_sections": encoder.stats.n_chunked_sections,
        "clean_verified_at": clean_verified_at,
        "total_pairs": total,
        "funds_with_zero_text": len(all_fund_ids) - len(fund_data.funds_with_text),
        "pairs_with_real_text": real,
        "pairs_with_locf": locf,
        "pairs_with_cold_start": cold,
        "pct_real_text": round(100 * real / total, 2) if total else 0,
        "pct_locf": round(100 * locf / total, 2) if total else 0,
        "pct_cold_start": round(100 * cold / total, 2) if total else 0,
        "mean_consecutive_cosine_sim": round(mean_cos, 6),
        "n_consecutive_pairs": len(cos_consec),
    }
    meta_path = output_dir / "prospectus_embeddings_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved {meta_path}")
    logger.info(
        f"Tokens: {meta['tokens_billed']:,}  Cost: ${meta['cost_usd_estimated']:.2f}  "
        f"Requests: {meta['n_api_requests']}  Chunked sections: {meta['n_chunked_sections']}"
    )


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("encode_prospectus_openai")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(output_dir / "encode_prospectus_openai.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def preflight(cfg: Config, fund_data: FundTextData, encoder: OpenAIProspectusEncoder, logger: logging.Logger) -> None:
    fund_ids = fund_data.all_fund_ids
    if cfg.max_funds > 0:
        fund_ids = fund_ids[:cfg.max_funds]

    logger.info("Dry-run: counting tokens ...")
    total_tokens = 0
    n_unique = 0
    n_oversized = 0
    for fid in tqdm(fund_ids, desc="Counting"):
        seen: set[tuple[int, str]] = set()
        for qi, sections in fund_data.fund_texts.get(fid, {}).items():
            for sec in SECTIONS:
                key = (qi, sec)
                if key in seen:
                    continue
                seen.add(key)
                t = sections.get(sec)
                if not t:
                    continue
                n = encoder.count_tokens(t)
                total_tokens += n
                n_unique += 1
                if n > MAX_TOKENS:
                    n_oversized += 1

    cost = total_tokens / 1e6 * COST_PER_1M_TOKENS
    logger.info(f"  Unique sections to encode: {n_unique:,}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Sections > {MAX_TOKENS} tokens (need chunking): {n_oversized}")
    logger.info(f"  Estimated cost: ${cost:.2f}")

    if cfg.dry_run_tokens:
        logger.info("Dry run only — exiting.")
        sys.exit(0)

    if not cfg.yes:
        ans = input("\nProceed with encoding? Type YES to continue: ")
        if ans.strip() != "YES":
            logger.info("Aborted by user.")
            sys.exit(0)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--json_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--max_lag", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_funds", type=int, default=0)
    p.add_argument("--dry_run_tokens", action="store_true",
                   help="Count tokens and exit; no API calls")
    p.add_argument("--yes", action="store_true",
                   help="Skip interactive confirmation (for PBS jobs)")
    a = p.parse_args()
    return Config(**vars(a))


def check_clean_marker(json_path: Path, logger: logging.Logger) -> None:
    marker = json_path.parent / "_clean_verified.txt"
    if not marker.exists():
        logger.error(f"Clean marker not found: {marker}")
        logger.error("Run: python sec_filings_project/verify_clean.py")
        sys.exit(2)
    if marker.stat().st_mtime < json_path.stat().st_mtime:
        logger.error(f"Clean marker is older than JSON. Re-run verify_clean.py.")
        sys.exit(2)
    logger.info(f"Clean marker OK: {marker}")


def main() -> None:
    cfg = parse_args()
    logger = setup_logging(cfg.output_dir)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        sys.exit(2)

    check_clean_marker(cfg.json_path, logger)

    logger.info(f"model={MODEL_NAME}  dim={EMB_DIM}  max_tokens={MAX_TOKENS}  overlap={CHUNK_OVERLAP}")

    fund_data = FundTextData.from_json(cfg.json_path, logger)
    encoder = OpenAIProspectusEncoder(api_key, logger)

    preflight(cfg, fund_data, encoder, logger)

    fund_ids = fund_data.all_fund_ids
    if cfg.max_funds > 0:
        fund_ids = fund_ids[:cfg.max_funds]
        logger.info(f"Limiting to first {cfg.max_funds} funds")

    skipped = errors = 0
    t0 = time.time()
    for i, fid in enumerate(tqdm(fund_ids, desc="Encoding funds")):
        if cfg.resume and load_fund_temp(fid, cfg.temp_dir) is not None:
            skipped += 1
            continue
        try:
            result = process_fund(fid, fund_data, encoder, cfg)
            save_fund_temp(fid, result, cfg.temp_dir)
        except Exception as exc:
            logger.error(f"Fund {fid} failed: {exc}", exc_info=True)
            errors += 1
            save_fund_temp(fid, ZERO_RESULT, cfg.temp_dir)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            cost = encoder.stats.total_tokens / 1e6 * COST_PER_1M_TOKENS
            logger.info(
                f"  [{i+1}/{len(fund_ids)}] tokens={encoder.stats.total_tokens:,}  "
                f"cost=${cost:.2f}  reqs={encoder.stats.n_requests}  "
                f"chunked={encoder.stats.n_chunked_sections}  "
                f"elapsed={elapsed/60:.1f}min"
            )

    logger.info(f"Skipped: {skipped}  Errors: {errors}")
    logger.info("Merging to HDF5 ...")
    merge_to_h5(fund_ids, cfg.temp_dir, cfg.output_dir, logger)
    logger.info("Computing metadata ...")
    save_metadata(fund_ids, fund_data, cfg.temp_dir, cfg.output_dir, encoder, cfg, logger)
    logger.info("Done.")


if __name__ == "__main__":
    main()
