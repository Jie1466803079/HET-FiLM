#!/usr/bin/env python3
"""Encode pre-extracted 485BPOS prospectus sections into fund-level temporal embeddings.

Reads a JSON file of fund prospectus text (objective, strategy, risk sections),
encodes each section with a transformer (sec-bert-base or finbert), applies
LOCF propagation across quarters, and writes embeddings + metadata to HDF5.

Usage:
    python scripts/encode_prospectus_json.py \
        --json_path sec_filings_project/extracted/fund_485bpos_sections_temporal.json \
        --output_dir sec_filings_project/embeddings \
        --encoder sec-bert \
        --batch_size 32 \
        --device cuda
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODER_MAP: dict[str, str] = {
    "sec-bert": "nlpaueb/sec-bert-base",
    "finbert": "ProsusAI/finbert",
}

SECTIONS: tuple[str, ...] = ("strategy", "risk", "objective")
SECTION_WEIGHTS: dict[str, float] = {"strategy": 0.5, "risk": 0.3, "objective": 0.2}

EMB_DIM = 768
MAX_SEQ_LEN = 512
CHUNK_OVERLAP = 64


def _build_quarter_list() -> list[str]:
    """Build sorted list of fiscal quarters from 2005Q3 to 2021Q3."""
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
assert N_QUARTERS == 65, f"Expected 65 quarters, got {N_QUARTERS}"


def date_to_quarter(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' to 'YYYYQn'."""
    y, m, _ = date_str.split("-")
    q = (int(m) - 1) // 3 + 1
    return f"{y}Q{q}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    json_path: Path
    output_dir: Path
    encoder: str = "sec-bert"
    batch_size: int = 32
    max_lag: int = 6
    device: str = "cuda"
    resume: bool = False
    max_funds: int = 0  # 0 = all

    @property
    def encoder_name(self) -> str:
        return ENCODER_MAP.get(self.encoder, self.encoder)

    @property
    def temp_dir(self) -> Path:
        return self.output_dir / "temp_npy"


# ---------------------------------------------------------------------------
# Attention aggregator for chunk pooling
# ---------------------------------------------------------------------------

class ChunkAttentionAggregator(nn.Module):
    """Learned attention over variable-length chunk embeddings."""

    def __init__(self, dim: int = EMB_DIM) -> None:
        super().__init__()
        self.attn_linear = nn.Linear(dim, 1, bias=False)

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        """Aggregate chunk embeddings via softmax attention.

        Args:
            chunks: (n_chunks, dim) tensor.
        Returns:
            (dim,) aggregated embedding.
        """
        scores = self.attn_linear(chunks)        # (n_chunks, 1)
        weights = F.softmax(scores, dim=0)       # (n_chunks, 1)
        return (weights * chunks).sum(dim=0)     # (dim,)


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------

class ProspectusEncoder:
    """Encodes prospectus text sections using a transformer with chunk attention."""

    def __init__(self, cfg: Config, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        if cfg.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")

        logger.info(f"Loading encoder: {cfg.encoder_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
            self.model = AutoModel.from_pretrained(cfg.encoder_name).to(self.device)
        except Exception:
            fallback = ENCODER_MAP["finbert"]
            logger.warning(f"Failed to load {cfg.encoder_name}, falling back to {fallback}")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
            self.model = AutoModel.from_pretrained(fallback).to(self.device)
        self.model.eval()

        self.aggregator = ChunkAttentionAggregator(EMB_DIM).to(self.device)
        nn.init.zeros_(self.aggregator.attn_linear.weight)

        self.batch_size = cfg.batch_size
        self._original_batch_size = cfg.batch_size

    # ---- chunking --------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks of ≤ MAX_SEQ_LEN tokens."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return []
        stride = MAX_SEQ_LEN - CHUNK_OVERLAP - 2  # room for [CLS]+[SEP]
        chunks: list[str] = []
        for start in range(0, len(tokens), stride):
            chunk_tok = tokens[start : start + MAX_SEQ_LEN - 2]
            chunks.append(self.tokenizer.decode(chunk_tok, skip_special_tokens=True))
            if start + MAX_SEQ_LEN - 2 >= len(tokens):
                break
        return chunks

    # ---- batch encoding --------------------------------------------------

    @torch.no_grad()
    def _encode_chunks_batched(self, chunks: list[str]) -> torch.Tensor:
        """Encode text chunks, mean-pool non-padding tokens.

        Returns: (n_chunks, EMB_DIM) tensor on CPU.
        """
        all_embs: list[torch.Tensor] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_SEQ_LEN, return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**encoded).last_hidden_state   # (B, seq, dim)
            mask = encoded["attention_mask"].unsqueeze(-1)     # (B, seq, 1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            all_embs.append(pooled.cpu())
        return torch.cat(all_embs, dim=0)

    # ---- section-level encoding ------------------------------------------

    def encode_section(self, text: Optional[str]) -> np.ndarray:
        """Encode a single section into a (768,) float32 array. None → zeros."""
        if not text or not text.strip():
            return np.zeros(EMB_DIM, dtype=np.float32)
        chunks = self._chunk_text(text)
        if not chunks:
            return np.zeros(EMB_DIM, dtype=np.float32)

        chunk_embs = self._encode_chunks_batched(chunks)  # (n, dim)
        if chunk_embs.shape[0] == 1:
            return chunk_embs[0].numpy().astype(np.float32)

        agg = self.aggregator(chunk_embs.to(self.device))
        return agg.detach().cpu().numpy().astype(np.float32)

    def encode_section_safe(self, text: Optional[str]) -> np.ndarray:
        """encode_section with OOM recovery: halve batch → CPU fallback."""
        try:
            return self.encode_section(text)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.logger.warning(f"OOM at batch_size={self.batch_size}, halving")
            self.batch_size = max(1, self.batch_size // 2)
            try:
                return self.encode_section(text)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.logger.warning("OOM again — falling back to CPU")
                orig_dev = self.device
                self.device = torch.device("cpu")
                self.model.cpu()
                self.aggregator.cpu()
                result = self.encode_section(text)
                self.device = orig_dev
                self.model.to(orig_dev)
                self.aggregator.to(orig_dev)
                self.batch_size = self._original_batch_size
                return result


# ---------------------------------------------------------------------------
# Data loading & structuring
# ---------------------------------------------------------------------------

@dataclass
class FundTextData:
    """Parsed text data organised by fund and quarter."""

    # fund_id → quarter_idx → {section: text}
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

            # keep entry with more text if duplicate
            if qi in fund_texts[fid]:
                old_len = sum(len(fund_texts[fid][qi].get(s) or "") for s in SECTIONS)
                new_len = sum(len(entry.get(s) or "") for s in SECTIONS)
                if new_len <= old_len:
                    continue

            fund_texts[fid][qi] = {s: entry.get(s) for s in SECTIONS}

        n_fund_nodes = max_id_idx + 1
        all_fund_ids = list(range(n_fund_nodes))
        funds_with_text = set(fund_texts.keys())
        pct = 100 * len(funds_with_text) / n_fund_nodes
        logger.info(f"  {len(funds_with_text)}/{n_fund_nodes} funds with text ({pct:.1f}%)")

        return FundTextData(fund_texts=fund_texts, all_fund_ids=all_fund_ids,
                            funds_with_text=funds_with_text)


# ---------------------------------------------------------------------------
# LOCF propagation
# ---------------------------------------------------------------------------

def apply_locf(
    fund_texts: dict[int, dict[int, dict[str, Optional[str]]]],
    fund_id: int,
    max_lag: int,
) -> list[tuple[dict[str, Optional[str]], int, bool]]:
    """Apply Last-Observation-Carried-Forward for one fund.

    Returns list of (section_dict, delta_t, has_text) for each of 65 quarters.
        delta_t = 0  → real text this quarter
        delta_t = 1…max_lag → LOCF, quarters since last real text
        delta_t = -1 → cold start (zero embedding)
    """
    raw = fund_texts.get(fund_id, {})
    result: list[tuple[dict[str, Optional[str]], int, bool]] = []
    empty: dict[str, Optional[str]] = {s: None for s in SECTIONS}

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


# ---------------------------------------------------------------------------
# Per-fund processing
# ---------------------------------------------------------------------------

def process_fund(
    fund_id: int,
    fund_data: FundTextData,
    encoder: ProspectusEncoder,
    cfg: Config,
    logger: logging.Logger,
) -> dict[str, np.ndarray]:
    """Encode all quarters for one fund.

    Returns dict with keys:
        strategy_emb, risk_emb, objective_emb  (65, 768)
        abs_emb, delta_emb                     (65, 768)
        cosine_sim                             (65,)
        delta_t                                (65,)  int8
        has_text                               (65,)  bool
    """
    locf = apply_locf(fund_data.fund_texts, fund_id, cfg.max_lag)

    sec_emb = {s: np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32) for s in SECTIONS}
    abs_emb = np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32)
    delta_emb = np.zeros((N_QUARTERS, EMB_DIM), dtype=np.float32)
    cosine_sim = np.zeros(N_QUARTERS, dtype=np.float32)
    delta_t = np.zeros(N_QUARTERS, dtype=np.int8)
    has_text = np.zeros(N_QUARTERS, dtype=bool)

    # Cache by source quarter to avoid re-encoding carried-forward text
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
                cache[key] = encoder.encode_section_safe(sections.get(sec))
            sec_emb[sec][qi] = cache[key]

        abs_emb[qi] = (
            SECTION_WEIGHTS["strategy"]  * sec_emb["strategy"][qi]
            + SECTION_WEIGHTS["risk"]    * sec_emb["risk"][qi]
            + SECTION_WEIGHTS["objective"] * sec_emb["objective"][qi]
        )

    # Delta embeddings & cosine similarity
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
# Temp file I/O (resume support)
# ---------------------------------------------------------------------------

def save_fund_temp(fund_id: int, result: dict[str, np.ndarray], temp_dir: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temp_dir / f"fund_{fund_id}.npz", **result)


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
# HDF5 output
# ---------------------------------------------------------------------------

def merge_to_h5(
    all_fund_ids: list[int], temp_dir: Path, output_dir: Path,
    logger: logging.Logger,
) -> None:
    n = len(all_fund_ids)
    h5_path = output_dir / "prospectus_embeddings.h5"
    logger.info(f"Merging {n} funds → {h5_path}")

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


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_metadata(
    all_fund_ids: list[int], fund_data: FundTextData,
    temp_dir: Path, output_dir: Path, cfg: Config, logger: logging.Logger,
) -> None:
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
    zero_text = len(all_fund_ids) - len(fund_data.funds_with_text)

    meta = {
        "encoder_name": cfg.encoder_name,
        "encoding_date": datetime.datetime.now().isoformat(),
        "section_weights": SECTION_WEIGHTS,
        "max_lag": cfg.max_lag,
        "total_pairs": total,
        "funds_with_zero_text": zero_text,
        "pairs_with_real_text": real,
        "pairs_with_locf": locf,
        "pairs_with_cold_start": cold,
        "pct_real_text": round(100 * real / total, 2),
        "pct_locf": round(100 * locf / total, 2),
        "pct_cold_start": round(100 * cold / total, 2),
        "mean_consecutive_cosine_sim": round(mean_cos, 6),
        "n_consecutive_pairs": len(cos_consec),
    }

    meta_path = output_dir / "prospectus_embeddings_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved {meta_path}")
    logger.info(
        f"Summary: {meta['pct_real_text']:.1f}% real / "
        f"{meta['pct_locf']:.1f}% LOCF / "
        f"{meta['pct_cold_start']:.1f}% cold-start | "
        f"mean cos_sim={mean_cos:.4f}"
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("encode_prospectus")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(output_dir / "encode_prospectus.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Encode 485BPOS prospectus sections → HDF5 embeddings")
    p.add_argument("--json_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--encoder", default="sec-bert", choices=list(ENCODER_MAP.keys()))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_lag", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_funds", type=int, default=0, help="Limit to first N funds (0=all, for testing)")
    a = p.parse_args()
    return Config(json_path=a.json_path, output_dir=a.output_dir, encoder=a.encoder,
                  batch_size=a.batch_size, max_lag=a.max_lag, device=a.device, resume=a.resume,
                  max_funds=a.max_funds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()
    logger = setup_logging(cfg.output_dir)
    logger.info(f"encoder={cfg.encoder_name}  batch={cfg.batch_size}  "
                f"max_lag={cfg.max_lag}  device={cfg.device}  resume={cfg.resume}")

    # ── Print structure ──────────────────────────────────────────────────
    with open(cfg.json_path) as f:
        raw: list[dict] = json.load(f)
    logger.info(f"JSON: {type(raw).__name__}, {len(raw)} entries")
    if raw:
        s = raw[0]
        logger.info(f"  Keys: {list(s.keys())}")
        logger.info(f"  Sample: id_idx={s['id_idx']}  ts={s['timestamp']}")
        for sec in SECTIONS:
            v = s.get(sec)
            logger.info(f"    {sec}: {len(v)} chars" if v else f"    {sec}: None")
    del raw

    # ── Load data ────────────────────────────────────────────────────────
    fund_data = FundTextData.from_json(cfg.json_path, logger)

    # ── Init encoder ─────────────────────────────────────────────────────
    logger.info("Initialising encoder...")
    encoder = ProspectusEncoder(cfg, logger)

    # ── Encode ───────────────────────────────────────────────────────────
    fund_ids_to_encode = fund_data.all_fund_ids
    if cfg.max_funds > 0:
        fund_ids_to_encode = fund_ids_to_encode[:cfg.max_funds]
        logger.info(f"Sample mode: encoding first {cfg.max_funds} funds only")

    logger.info("Encoding funds...")
    skipped = errors = 0

    for fid in tqdm(fund_ids_to_encode, desc="Encoding funds"):
        if cfg.resume and load_fund_temp(fid, cfg.temp_dir) is not None:
            skipped += 1
            continue
        try:
            result = process_fund(fid, fund_data, encoder, cfg, logger)
            save_fund_temp(fid, result, cfg.temp_dir)
        except Exception as exc:
            logger.error(f"Fund {fid} failed: {exc}", exc_info=True)
            errors += 1
            save_fund_temp(fid, ZERO_RESULT, cfg.temp_dir)

    if skipped:
        logger.info(f"Resumed: skipped {skipped} already-encoded funds")
    if errors:
        logger.warning(f"Errors: {errors} funds (saved as zeros)")

    # ── Merge → HDF5 ────────────────────────────────────────────────────
    logger.info("Merging to HDF5...")
    merge_to_h5(fund_ids_to_encode, cfg.temp_dir, cfg.output_dir, logger)

    # ── Metadata ─────────────────────────────────────────────────────────
    logger.info("Computing metadata...")
    save_metadata(fund_ids_to_encode, fund_data, cfg.temp_dir, cfg.output_dir, cfg, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
