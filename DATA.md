# Data payload

The training pipelines expect the following files under `$DATA_ROOT/`.
They are hosted on **OneDrive UNSW** (rather than in this repository)
because they exceed GitHub's 100 MB per-file limit and, in the case of
the US graph pickle, carry WRDS/CRSP redistribution constraints that
make a public code host inappropriate.

**Access:** the OneDrive share URLs are listed in
`scripts/data_manifest.txt`. Download either through the OneDrive web
UI (click each URL, click Download) or programmatically via
`scripts/download_data.sh`.

## File manifest

| Path (under `$DATA_ROOT`)                       | Size    | Contents                                                                 |
|-------------------------------------------------|---------|--------------------------------------------------------------------------|
| `us/graph.pkl`                                  | 250 MB  | US heterogeneous graph, 65 quarters 2005Q3–2021Q3, from CRSP MFDB + Compustat + SEC EDGAR |
| `us/prospectus_embeddings.h5`                   | 4.0 GB  | 3,166 funds × up to 65 quarters × 1024-d, OpenAI text-embedding-3-large on 485BPOS strategy sections, report-anchored with carry-forward |
| `us/se_htgnn_llm_features.pt`                   | 36 KB   | LLaMA-3-8B per-node-type embeddings for SE-HTGNN                        |
| `us/tape_ta.h5`, `us/tape_p.h5`, `us/tape_e.h5` | 3 × ~1 GB | TAPE 3-channel DeBERTa-base + LoRA embeddings, 768-d per channel      |
| `canada/graph.pkl`                              | 264 MB  | Canada graph, 44 quarters 2015Q1–2025Q4, from SEDAR holdings + Compustat NA |
| `canada/prospectus_embeddings.h5`               | 28 MB   | 755 SEDAR-covered funds × 44 quarters × 1024-d text-embedding-3-large   |
| `canada/tape_ta.h5`, `tape_p.h5`, `tape_e.h5`   | 3 × ~30 MB | Canada TAPE 3-channel embeddings                                     |
| `heterformer/ckpt_book/`                        | 7.4 GB  | Heterformer pretrained checkpoints (18 files, ~420 MB each)             |

Total payload: **~19 GB**, well within the UNSW OneDrive 1 TB
allocation.

## Provenance

### US graph pickle

Built from the **CRSP Survivor-Bias-Free US Mutual Fund Holdings**
database (WRDS-licensed) and the **CRSP Monthly Stock File**, with SEC
EDGAR 485BPOS filings supplying the fund–CIK mapping. Each fund–quarter
in the pickle carries an 11-dim fund characteristic vector (fund age,
expense ratio, TNA, portfolio herfindahl, etc.) and each stock–quarter
carries a 15-dim vector (momentum, reversal, idiosyncratic volatility
and Yeo–Johnson-transformed size/turnover). See Section 3 of the thesis
for the full feature list.

Fund-quarter cells with any missing feature are dropped during graph
construction. The universe is domestic diversified actively managed
equity funds; management-company nodes (fund–by-company edges) are
included as a third node type.

**Redistribution note.** The graph pickle is a derived artefact of
WRDS-licensed CRSP data. It is hosted alongside this code purely for
academic reproducibility of the thesis results. Downstream users who
do not have a WRDS/CRSP entitlement should treat the pickle as
research-use-only and must not redistribute it further.

### US prospectus embeddings

Sourced from SEC EDGAR **485BPOS** post-effective-amendment filings
(2005Q1 through 2021Q4). For each fund–quarter, the *Principal
Investment Strategies* section is extracted with a multi-stage
regex + heading pattern extractor (see
`scripts/data_building/text_us/extract_485bpos_sections_v2.py`), then
encoded with OpenAI's `text-embedding-3-large` (1024-d output).

Report anchoring: each fund–quarter cell inherits the most recent
prospectus whose effective date ≤ quarter end. Cells fall into three
coverage states — **fresh** (3.22 %), **carry-forward** (49.91 %), and
**cold-start** (46.87 %). See Table 3.5 of the thesis.

Cold-start funds are encoded as a zero vector and their per-fund
availability mask is `False`; the trainer's prospectus fusion module
drives the gate towards the numerical channel for these funds.

### Canada graph pickle

Built from the **SEDAR** holdings panel (post-2015 disclosures under
NI 81-101 / NI 81-102) joined against **Compustat North America** for
security-level metadata. Quarterly snapshots are keyed by the
**filing date** (`fdate`) rather than the report date, which is more
conservative than the US `rdate`/`caldt` convention and eliminates the
25 % long-lag tail from the report-anchoring problem. The universe is
funds with equity share ≥ 0.50 and ≥ 2 stock positions per quarter,
after excluding ETFs / index / passive / bond / money-market names by
regex.

Each fund–quarter carries an 11-dim vector (portfolio concentration,
top-1/top-10 weights, new-position share, sector herfindahl, USD share,
Canadian share, fund age) and each stock–quarter an 11-dim vector
(returns, momentum, volatility, size, turnover, spread, dividend yield,
volume, shares outstanding, price, dividend rate) with 1st/99th
percentile winsorisation on skewed columns. See Tables 3.2, 3.4.

### Canada prospectus embeddings

Sourced from SEDAR simplified prospectuses (SP) plus amendments and
restated SPs; Material Change reports, Fund Facts, AIF and MRFP
documents are deliberately excluded. Fund–quarter alignment is
report-anchored with carry-forward. The covered pool is 755 funds out
of the 3,459-fund universe (Table 3.5 covered-pool column). Uncovered
funds are cold-start with a zero embedding.

### SE-HTGNN LLM features

The 36 KB `.pt` file stores type-level LLaMA-3-8B embeddings that the
SE-HTGNN backbone uses in its `LLM4init` layer. It was generated once
from short type descriptions and does not depend on any panel state.

### Heterformer checkpoints

The upstream Heterformer implementation ships pretrained models at
`third_party/heterformer_src/pretrain/ckpt_book/` (`ckpt_publisher/`
and `ckpt_shelves/`), each roughly 420 MB. These are the frozen
BERT-based backbones that the funds-specific Stage 2 evaluator loads.
They are hosted on OneDrive, not GitHub, because each single file
exceeds GitHub's 100 MB per-file hard limit.

## Rebuilding from raw sources

Users with WRDS + SEC EDGAR + SEDAR + OpenAI API access can
regenerate every artefact above from `scripts/data_building/`. The
commands are in `HOW_TO_REPRODUCE.md`.

## Long-term archival note

OneDrive UNSW links depend on the author's UNSW account remaining
active. If you need a citation-stable copy — for example when citing
this data from a later paper — mirror the same files to Zenodo
(free, 50 GB per record, permanent DOI) or UNSWorks (UNSW's
institutional repository). No code changes are required; just
update `scripts/data_manifest.txt` with the mirror URLs.

## Ethics and licensing

- **US graph + embeddings** — derivative of WRDS/CRSP subscription data
  and public SEC EDGAR filings, distributed for academic reproducibility.
- **Canada graph + embeddings** — derivative of public SEDAR/SEDAR+ and
  Compustat NA data. Distributed under Creative Commons Attribution 4.0.
- **Third-party checkpoints** — Heterformer checkpoints inherit their
  upstream licence (see the upstream README).
- **No PII, no confidential holdings** — every source is either
  regulator-mandated public disclosure (SEDAR, SEC EDGAR) or
  WRDS-mediated academic research access (CRSP).
