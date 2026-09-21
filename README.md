# HET-FiLM

Code and data pointers for the Master of Research thesis:

> **HET-FiLM: Edge-Trajectory-Conditioned Modulation for Initiation-Weight
> Prediction on Dynamic Heterogeneous Mutual Fund Graphs.**
> Jie Liu, UNSW School of Computer Science and Engineering, September 2026.

The task is *initiation-weight prediction*: given a fund–stock pair that has
never appeared in the graph before, forecast the portfolio weight the fund
will assign to it in the query quarter. The method (HET-FiLM) conditions
message passing on the trajectory of each holding edge and fuses a
prospectus-derived text channel into fund representations. It is evaluated
against twelve baselines on two independent panels: **US CRSP + SEC EDGAR**
(2005Q3–2021Q3, 65 quarters) and **SEDAR + Compustat NA** (2015Q1–2025Q4,
44 quarters).

## Quick start

```bash
# 1. Environment (CUDA 11.8 host, Python 3.9+)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install -r requirements.txt
pip install -e .

# 2. Data — fetch from OneDrive (URLs in scripts/data_manifest.txt; see DATA.md)
export DATA_ROOT=/scratch/hetfilm-data
bash scripts/download_data.sh

# 3. Reproduce the HET-FiLM headline row on the US panel
export REPO_ROOT=$PWD
qsub configs/us/us_hetfilm.pbs             # on a PBS cluster
# or, without a scheduler:
bash configs/us/us_hetfilm.pbs             # loops seeds 42..51 serially
```

Every configuration in Tables 5.2, 5.3 and Appendix A has a matching
PBS script under `configs/us/` or `configs/canada/`.
See `HOW_TO_REPRODUCE.md` for the row-to-script mapping.

## Repository layout

```
hetfilm/
├── core/                    # Python package: models, data loaders, trainer, losses
│   ├── args_model.py        # CLI flags
│   ├── data/                # FundsDataset, FundsEdgeWeightDataset (+ CasMLN variant)
│   ├── models/              # 13 baselines + HET-FiLM (start at hetfilm.py)
│   ├── trainer/edge_multitask.py    # two-stage trainer
│   └── evaluation_metrics.py
├── scripts/
│   ├── run/                 # Entry points (run_model.py + wrappers)
│   ├── data_building/       # Preprocessing: US + Canada graph, text pipelines
│   ├── analysis/            # parse_both_stages.py + aggregators + paper CSVs
│   ├── heterformer/         # Heterformer-specific Stage 1 / Stage 2 evaluators
│   └── download_data.sh     # Fetch OneDrive payload → $DATA_ROOT
├── configs/
│   ├── us/                  # Table 5.2 — 13 PBS jobs
│   └── canada/              # Table 5.3 — 13 PBS jobs
├── third_party/
│   ├── heterformer_src/     # KDD 2023 Heterformer, pruned
│   └── SE-HTGNN/            # SE-HTGNN model/ + LLM feature tensors
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Method summary

HET-FiLM is a dynamic heterogeneous graph model for initiation-weight
prediction. It has three parts:

1. **Prospectus-aware fund encoding** — a frozen 1024-d text embedding is
   projected to a 128-d strategy code, gated per-dimension against the
   linear projection of the 11-d numerical fund features, and replaces the
   raw fund vector in every snapshot. Cold-start funds receive a zero
   text vector and the gate falls back to the numerical channel.
2. **Type-indexed attention encoder** — each quarterly snapshot is encoded
   by multi-head attention whose query, key and value projections are
   indexed by the meta-relation of the edge being traversed (the
   formulation of Hu et al. 2020), and a temporal self-attention layer
   pools the per-snapshot representations over the support window.
3. **Edge-trajectory-conditioned message passing** — every fund–stock edge
   in a registry of past holdings is encoded through a causal GRU on the
   three-channel sequence `(w_t, Δw_t, π_t)`. The final hidden state
   produces per-edge γ and β that feature-wise linearly modulate each
   message before attention weighting. Modulation heads are
   zero-initialised, so training starts from unmodulated messages.

Two-stage training: Stage 1 fits the encoder by BCE link prediction on
the same new-edge population; Stage 2 freezes the encoder and fits an MLP
weight head under Huber loss in `log(1+y)` space.

## Where HET-FiLM is implemented

`core/models/hetfilm.py` is the entry point to the model: it imports every
HET-FiLM module, and the table shows where each one lives.

| Component | Code |
|---|---|
| Prospectus text fusion into fund features | `ProspectusTextFusion` in `core/models/prospectus_fusion.py` |
| Type-indexed attention encoder and temporal pooling | `HGT`, `HGTConv` in `core/models/HGT.py` |
| Edge-trajectory encoder (causal GRU over each persistent edge's weight history) | `EdgeTrajectoryEncoder` in `core/models/edge_trajectory.py` |
| Per-edge FiLM heads producing γ and β (plus the text-conditioned branch enabled by `--use_tcetf`) | `EdgeTrajectoryFiLM` in `core/models/edge_trajectory.py` |
| FiLM applied to the attention values during message passing | `HGTConv.message` in `core/models/HGT.py` |
| Wiring and two-stage training | `MultiTaskEdgePredictor` in `core/models/multitask_edge.py` (`configure_edge_trajectory`, `_apply_prospectus_fusion`) |
| Model assembly from the command-line flags | `scripts/run/run_model.py` |

In the run scripts, `--model HGT+` is the internal key that selects the
attention encoder, and `--use_prospectus` and `--use_edge_trajectory`
switch on the fusion and trajectory modules. The exact flags for each
panel are in `configs/us/us_hetfilm.pbs` and
`configs/canada/canada_hetfilm.pbs`; the Canadian configuration also
passes `--use_tcetf` and `--grad_clip 1.0`.

## Reproducibility notes

- **Data payload lives on OneDrive, not GitHub**, because the four largest
  files (graph pickles + US prospectus embeddings) exceed GitHub's 100 MB
  per-file hard limit and the WRDS-derived US graph pickle carries
  redistribution constraints. See `DATA.md` for the file manifest.
- **Ten random seeds per configuration** (42–51). Every PBS script loops
  the same range; `analysis/paper_table_5_{2,3}_*.csv` reports means and
  ddof=1 sample standard deviations over these seeds. Two Canada rows
  (HGT+, DySAT) are flagged in the paper for a single divergent seed
  and reported with a robust "drop worst R²" variant — see the CSVs.
- **Heterformer is one seed** on both panels (see paper caveat) because
  its Stage-1 candidate scoring uses fund-mean pooling on a 1:1 negatively
  subsampled set, and the pretrained backbone is a public checkpoint
  (BERT-base) whose stochasticity would not be resolved by more seeds.

## Third-party code

- `third_party/heterformer_src/` — lightly pruned copy of the official
  Heterformer implementation (Jin et al., "Heterformer: Transformer-based
  Deep Node Representation Learning on Heterogeneous Text-Rich Networks,"
  KDD 2023).
- `third_party/SE-HTGNN/` — LLM-enhanced SE-HTGNN backbone (`model/`) and
  the LLaMA-3-8B per-node-type feature tensors used by the SEHTGNN_LLM
  branch of `core/models/load_model.py`.

Pretrained Heterformer checkpoints (`pretrain/ckpt_book/`, ~7.4 GB) are
hosted with the data payload on OneDrive, not in this repository.

## Citing

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{liu2026hetfilm,
  author = {Liu, Jie},
  title  = {HET-FiLM: Edge-Trajectory-Conditioned Modulation for
            Initiation-Weight Prediction on Dynamic Heterogeneous
            Mutual Fund Graphs},
  school = {UNSW Sydney},
  year   = {2026},
  type   = {Master of Research thesis},
}
```

## License

MIT for original code. Third-party code retains its upstream license
(see `third_party/heterformer_src/README.md` and
`third_party/SE-HTGNN/README.md`). Data payload on OneDrive is
Creative Commons Attribution 4.0 for the derived Canadian graph and
text embeddings; the US graph pickle is provided under the terms of
the researcher's WRDS/CRSP subscription and is redistributed here for
academic reproducibility only.
