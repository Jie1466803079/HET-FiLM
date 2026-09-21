# How to reproduce the thesis tables

Every row of Tables 5.2, 5.3 and Appendix A has an exact PBS script
in `configs/`. Each script loops seeds 42–51 and writes per-seed JSON
results under `logs/${PBS_JOBNAME}/seed_<N>/hgt_plus_edge_weight_results.json`.
Aggregation into the paper-formatted table is done by the analysis
scripts in `scripts/analysis/`.

**Prerequisites for every run:**

```bash
export REPO_ROOT=$PWD
export DATA_ROOT=/path/to/downloaded/data          # from scripts/download_data.sh
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
```

## Table 5.2 — US, Entry subset, 10-seed mean ± ddof=1 std

| Paper row              | Script                                    |
|------------------------|-------------------------------------------|
| GCN                    | `configs/us/us_gcn.pbs`                   |
| GAT                    | `configs/us/us_gat.pbs`                   |
| R-GCN                  | `configs/us/us_rgcn.pbs`                  |
| HAN                    | `configs/us/us_han.pbs`                   |
| DySAT                  | `configs/us/us_dysat.pbs`                 |
| HTGNN                  | `configs/us/us_htgnn.pbs` (9/10 seeds)    |
| SE-HTGNN               | `configs/us/us_se_htgnn.pbs`              |
| DHGAS                  | `configs/us/us_dhspace.pbs`               |
| CasMLN                 | `configs/us/us_casmln_noew.pbs`           |
| HGT+                   | `configs/us/us_hgt_plus.pbs`              |
| TAPE (DeBERTa + LoRA)  | `configs/us/us_tape.pbs`                  |
| Heterformer (BERT)     | `configs/us/us_heterformer.pbs` (1 seed)  |
| **HET-FiLM**           | `configs/us/us_hetfilm.pbs`               |

Aggregate:

```bash
python scripts/analysis/build_results_csv_casmln_neg.py \
    --logs-root logs/ \
    --out scripts/analysis/paper_table_5_2_us.csv \
    --panel us --ddof 1
```

The frozen paper CSV of record is
`scripts/analysis/paper_table_5_2_us.csv`; a fresh run should match its
numbers up to seed-level noise from CUDA nondeterminism.

## Table 5.3 — Canada, Entry subset

Same 13 rows, in `configs/canada/`:

| Paper row              | Script                                    |
|------------------------|-------------------------------------------|
| GCN                    | `configs/canada/canada_gcn.pbs`           |
| GAT                    | `configs/canada/canada_gat.pbs`           |
| R-GCN                  | `configs/canada/canada_rgcn.pbs`          |
| HAN                    | `configs/canada/canada_han.pbs`           |
| DySAT (8 seeds)        | `configs/canada/canada_dysat.pbs`         |
| HTGNN                  | `configs/canada/canada_htgnn.pbs`         |
| SE-HTGNN               | `configs/canada/canada_se_htgnn.pbs`      |
| DHGAS                  | `configs/canada/canada_dhspace.pbs`       |
| CasMLN                 | `configs/canada/canada_casmln_noew.pbs`   |
| HGT+ (divergent seed)  | `configs/canada/canada_hgt_plus.pbs`      |
| TAPE                   | `configs/canada/canada_tape.pbs`          |
| Heterformer (1 seed)   | `configs/canada/canada_heterformer.pbs`   |
| **HET-FiLM**           | `configs/canada/canada_hetfilm.pbs`       |

Canada uses a 4-quarter validation window and 8-quarter test window (vs.
US 6 / 14) — this is preserved verbatim in the copied PBS scripts.

Aggregation with the "drop worst R² seed" robust variant used in the
paper table:

```bash
python scripts/analysis/build_results_csv_casmln_neg.py \
    --logs-root logs/ \
    --out scripts/analysis/paper_table_5_3_canada.csv \
    --panel canada --ddof 1 --drop-worst-r2
```

## Appendix A — Stage-1 link-prediction results

Each PBS script already produces Stage-1 AUC/AP in the same JSON as
Stage-2. To aggregate the Appendix table:

```bash
python scripts/analysis/build_results_csv_casmln_neg.py \
    --logs-root logs/ --stage 1 \
    --out scripts/analysis/appendix_a_stage1.csv
```

## Running a single seed for a smoke test

```bash
python scripts/run/run_model_casmln_neg.py \
    --model HGT+ --dataset Funds \
    --task link_weight_multitask_new_twostage \
    --device cuda --seed 42 \
    --time_window 8 --neg_sampling_ratio 1.0 \
    --hid_dim 64 --n_heads 4 --n_layers 2 \
    --lr 0.001 --wd 0.0 \
    --max_epochs 20 --patience 10 \
    --use_joint_losses --use_amp 0 \
    --stage1_monitor auc \
    --use_prospectus \
    --prospectus_emb_path "$DATA_ROOT/us/prospectus_embeddings.h5" \
    --no_modality_dropout --risk_weight 0.0 \
    --numerical_dim 11 --no_phase_scheduler --no_staleness \
    --use_edge_trajectory --edge_trajectory_dim 32 \
    --log_dir logs/smoke_hetfilm_us_s42
```

`FUNDS_GRAPH_PKL=$DATA_ROOT/us/graph.pkl` must be exported before this
command (or add it inline). The full PBS scripts export every env-var
the training loop reads.

## Data reconstruction from raw sources

If you have WRDS/CRSP + SEC EDGAR access and want to rebuild the US
graph from scratch instead of downloading the pkl:

```bash
bash scripts/data_building/us/build_us_graph.sh          # placeholder — see README
python scripts/data_building/us/Building_graph_v3_preprocessed.py
python scripts/data_building/us/add_market_val_to_graphs_fixed.py
```

For the Canadian panel:

```bash
python scripts/data_building/canada/build_stock_features.py
python scripts/data_building/canada/build_fund_features.py
python scripts/data_building/canada/build_canada_fundno_unified_map.py
python scripts/data_building/canada/build_canada_graph_pkl.py
```

Text pipelines:

```bash
# US 485BPOS → OpenAI text-embedding-3-large → carry-forward H5
python scripts/data_building/text_us/download_all_485bpos.py
python scripts/data_building/text_us/extract_485bpos_sections_v2.py
python scripts/data_building/text_us/postprocess_v2_extraction.py
python scripts/data_building/text_us/encode_prospectus_openai.py
python scripts/data_building/text_us/rebuild_prospectus_h5_report_anchored.py --anchor_policy carry

# Canada SEDAR → per-manager pattern extraction → OpenAI → H5
python scripts/data_building/text_canada/extract_all_pdfs.py
python scripts/data_building/text_canada/build_text_canada_carry.py
```

All of these read `${DATA_ROOT}` and produce artefacts in
`${DATA_ROOT}/{us,canada}/`.
