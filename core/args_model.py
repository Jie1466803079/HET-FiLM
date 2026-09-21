import argparse
from core.models import Sta_MODEL, Homo_MODEL
import os


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Aminer")
    parser.add_argument("--model", type=str)
    parser.add_argument("--dhconfig", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="logs/tmp")
    parser.add_argument("--device", default="6")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--dynamic", type=int, default=-1)
    parser.add_argument("--homo", type=int, default=-1)
    parser.add_argument("--twin", type=int, default=-1)
    parser.add_argument("--time_window", type=int, default=-1,
                        help="Optional override for temporal window (falls back to dataset default when -1)")
    parser.add_argument("--test_full", type=int, default=-1)
    parser.add_argument("--predict_type", type=str, default="")
    parser.add_argument("--in_dim", type=int, default=-1)
    parser.add_argument("--hid_dim", type=int, default=-1)
    parser.add_argument("--out_dim", type=int, default=-1)
    parser.add_argument("--num_classes", type=int, default=-1)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--cul", type=int, default=1)
    parser.add_argument("--norm", type=int, default=1)
    parser.add_argument("--hlinear_act", type=str, default="tanh")
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--grad_clip", type=float, default=0)
    parser.add_argument("--stage2_warmup_epochs", type=int, default=0,
                        help="Linear LR warmup epochs for Stage 2 only (0 = off)")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_learnable_feats", action="store_true")
    parser.add_argument("--target_mode", type=str, default="value",
                        choices=["value", "rank"],
                        help="Target mode: value (regression on returns) or rank (percentile rank per snapshot)")
    parser.add_argument("--loss_mode", type=str, default="mse",
                        choices=["mse", "huber", "ic", "bce"],
                        help="Loss to optimize for Funds: mse, huber, ic (negative Pearson correlation), or bce for link tasks")
    parser.add_argument("--task", type=str, default="regression",
                        choices=['regression', 'link', 'link_weight', 'link_weight_delta_cls',
                                 'link_weight_multitask', 'link_weight_multitask_new',
                                 'link_weight_multitask_new_twostage', 'link_weight_multitask_new_joint'],
                        help="Task type")
    parser.add_argument("--neg_sampling_ratio", type=float, default=1.0)
    parser.add_argument("--edge_delta_pct", type=float, default=80.0)
    parser.add_argument("--edge_delta_abs", type=float, default=-1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=1.0)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument(
        "--tail_reweight_alpha",
        type=float,
        default=0.0,
        help="Stage-2 loss reweighting: each positive edge with weight_true > "
             "q67_train gets a sample weight of (1 + alpha). 0.0 disables (default).",
    )
    parser.add_argument("--pos_weight", type=float, default=-1.0)
    parser.add_argument("--use_joint_losses", action="store_true",
                        help="Use BCEWithLogitsLoss + HuberLoss(log1p) in two-stage pipeline")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip Stage 1 and load --stage1_ckpt")
    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--skip_stage2", action="store_true",
                        help="Skip Stage 2 training; load --stage2_ckpt and run eval only "
                             "(mirrors --skip_stage1).")
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--skip_joint", action="store_true",
                        help="Joint-task analogue of --skip_stage2: skip Joint training; "
                             "load --joint_ckpt and run final eval only.")
    parser.add_argument("--joint_ckpt", type=str, default="")
    parser.add_argument("--gate_hidden", type=int, default=16)
    parser.add_argument("--sampling_ratio", type=float, default=0.8)
    parser.add_argument("--gate_clip", type=float, nargs=2, metavar=('MIN', 'MAX'),
                        default=(0.02, 0.98))
    parser.add_argument("--gate_keep_q", type=float, default=0.3)
    parser.add_argument("--aux_lambda", type=float, default=0.2)
    parser.add_argument("--monitor_metric", type=str, default="",
                        help="Override monitor metric for early stopping (e.g., mae or auc)")
    parser.add_argument("--stage1_monitor", type=str, default="",
                        help="Override Stage 1 (link prediction) monitor. Default 'val_loss'. "
                             "Try 'entry_auc' to pick checkpoints by entry-discrimination.")
    parser.add_argument("--stage2_monitor", type=str, default="",
                        help="Override Stage 2 (weight regression) monitor. Default 'val_loss'. "
                             "Try 'mae_entry' (lower-better) for entry-weight prediction quality.")
    # XGBoost
    parser.add_argument("--xgb_max_depth", type=int, default=5)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.1)
    parser.add_argument("--xgb_n_estimators", type=int, default=100)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_min_child_weight", type=int, default=1)
    parser.add_argument("--xgb_gamma", type=float, default=0.0)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    # LLM
    parser.add_argument("--llm_embedding_path", type=str,
                        default=os.environ.get("LLM_EMBEDDING_PATH", "data/funds_llm_features.pt"))
    # CasMLN
    parser.add_argument("--amplifier", type=float, default=5.0,
                        help="CasMLN degree amplifier base (default 5.0)")
    # Prospectus
    parser.add_argument("--use_prospectus", action="store_true")
    parser.add_argument("--use_tcmp", action="store_true",
                        help="Enable Text-Conditioned Message Passing (TCMP): "
                             "per-layer FiLM modulation of fund hidden states by "
                             "fund-side text. Asymmetric — stock features unchanged. "
                             "Requires --use_prospectus. Default: off.")
    parser.add_argument("--tcmp_text_dim", type=int, default=128,
                        help="Text dim for TCMP FiLM input (default 128, "
                             "matches ProspectusTextFusion.ABS_PROJ_DIM).")
    parser.add_argument("--no_text_fund_feature", action="store_true",
                        help="Bypass text injection into fund node features "
                             "(h_fund := num_proj(numerical) only). Aux losses "
                             "(TBA, SpAB, SpatialAlign, ToA) still consume text "
                             "via abs_proj_out. Default: off.")
    parser.add_argument("--use_prospectus_node", action="store_true")
    parser.add_argument("--use_prospectus_node_cold_token", action="store_true")
    parser.add_argument("--prospectus_emb_path", type=str,
                        default="/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/sec_filings_project/embeddings_openai/prospectus_embeddings.h5")
    parser.add_argument("--numerical_dim", type=int, default=16)
    parser.add_argument("--risk_weight", type=float, default=0.375)
    parser.add_argument(
        "--mask_invalid_strategy", type=int, choices=(0, 1), default=1,
        help="Treat (fund, quarter) cells with zero strategy_emb as cold-start "
             "(Case-3 masking). 1=on (default); 0=off (pre-2026-05-17 behavior, "
             "for direct baseline comparison).",
    )
    parser.add_argument("--no_staleness", action="store_true")
    parser.add_argument("--staleness_scale", type=float, default=6.0,
                        help="Divisor in ProspectusTextFusion's staleness sigmoid: "
                             "sigmoid(w * delta_t/staleness_scale + b). Default 6.0 "
                             "is calibrated for the bounded report-anchored H5 "
                             "(max_lag=6). Set to ~24 for the unbounded carry-forward "
                             "H5 so the sigmoid spans deeper in-distribution Δ.")
    parser.add_argument("--use_intent_attention", action="store_true",
                        help="M1 (IBF probe): enable IntentConditionedAttention. "
                             "Off by default. When on, ProspectusTextFusion-into-encoder "
                             "is bypassed (option B); text enters only at the head via "
                             "g_f_eff and via mu inside the attention. Requires "
                             "--use_prospectus + --prospectus_emb_path.")
    parser.add_argument("--intent_lambda", type=float, default=1.0,
                        help="Weight on the eligibility prior mu in the M1 attention "
                             "logit: a = QK/sqrt(d) + lambda * mu. Default 1.0.")
    parser.add_argument("--intent_safety_cap", type=int, default=512,
                        help="Max context entries per scored edge in M1 attention. "
                             "When |context| > cap, keep top-cap by mu. Default 512 "
                             "(DO-FIRST #1 / option A: non-binding for ~62% of "
                             "fund-quarters in the 2005Q3 graph).")
    parser.add_argument("--intent_holdings_cap", type=int, default=512,
                        help="Per-fund cap on holdings entries in C(f). Default 512 "
                             "(covers p98 of |holdings|; index funds with >512 "
                             "positions are truncated before the safety_cap μ-trim).")
    parser.add_argument("--intent_siblings_cap", type=int, default=128,
                        help="Per-fund cap on sibling-stock entries in C(f), ranked "
                             "by family co-holding frequency. Default 128 — top-128 "
                             "captures the dominant family theme without exploding "
                             "context for big mgmt-company families.")
    parser.add_argument("--intent_staleness_scale", type=float, default=24.0,
                        help="Divisor for the staleness scalar fed into phi(delta_t) "
                             "inside the M1 attention. Default 24.0 (carry-H5 calibrated). "
                             "Independent of --staleness_scale (which applies only to "
                             "the legacy ProspectusTextFusion staleness gate).")
    parser.add_argument("--prospectus_fusion_mode", type=str, default="gated",
                        choices=["gated", "concat", "sum"],
                        help="Top-level fusion selector. 'gated' (default): "
                             "ProspectusTextFusion (existing behavior, driven by "
                             "--fusion_mode/--use_jacobian_ratio_gate/etc). "
                             "'concat': ProspectusTextConcatFusion — direct concat of "
                             "abs_proj(text) with num_proj(numerical) into a Linear "
                             "projection (no gate, no staleness sigmoid, cold-start "
                             "hard-zero). Meant for word2vec-style low-dim text runs; "
                             "H5 EMB_DIM is auto-detected via the flex-dim loader. "
                             "'sum': ProspectusTextSumFusion — elementwise add of "
                             "abs_proj(text) and num_proj(numerical), no learned "
                             "mixing matrix. num_proj has LayerNorm so scale matches "
                             "text branch. Third-cell ablation vs concat/gated.")
    parser.add_argument("--fusion_mode", type=str, default="convex",
                        choices=["convex", "residual", "film"],
                        help="ProspectusTextFusion combination mode. "
                             "'convex' (default, existing behavior): "
                             "h_fund = gate*text_norm(text_input) + "
                             "(1-gate)*num_proj(numerical) with gate bias init 1.0. "
                             "'residual': h_fund = num_proj(numerical) + "
                             "gate*text_norm(text_input) with gate bias init -5.0 "
                             "(sigmoid ≈ 0 at init). Text becomes a strictly additive, "
                             "zero-init add-on: model behaves like a text-free baseline "
                             "at epoch 0 and can only add signal from text. Cold-start "
                             "funds never lose numerical signal. Only affects "
                             "--use_prospectus runs.")
    parser.add_argument("--no_modality_dropout", action="store_true")
    parser.add_argument("--no_phase_scheduler", action="store_true",
                        help="Bypass the three-phase training scheduler even when "
                             "--use_prospectus is on. Routes through the regular "
                             "train_till_end with all params unfrozen from epoch 0 "
                             "(no Phase A freeze, no LR ramping). Forward fusion "
                             "still runs identically — only the optimization schedule changes.")
    # SimTeG baseline (Task 8): inject fine-tuned LM embeddings into fund features only.
    # Off by default to preserve baseline behavior.
    parser.add_argument("--simteg_node_features", type=str, default="none",
                        choices=["none", "replace", "concat"],
                        help="SimTeG injection mode for fund features (default: none).")
    parser.add_argument("--simteg_h5_path", type=str,
                        default="/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/"
                                "sec_filings_project/embeddings_simteg/"
                                "prospectus_embeddings_simteg.h5",
                        help="Path to SimTeG embeddings H5.")
    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--text_stock_prior", action="store_true")
    parser.add_argument("--text_mlp_decoder", action="store_true")
    parser.add_argument("--use_fund_stock_contrastive", action="store_true")
    parser.add_argument("--fund_stock_contrastive_lambda", type=float, default=0.1)
    # Option 1: text-behavior trajectory alignment (additive, off by default).
    parser.add_argument("--use_text_behavior_alignment", action="store_true",
                        help="Add per-fund text↔portfolio trajectory alignment "
                             "(strategy-only ΔE vs portfolio Δh). Auxiliary InfoNCE "
                             "loss; optional gate via --text_behavior_alignment_gate.")
    parser.add_argument("--text_behavior_alignment_lambda", type=float, default=0.1,
                        help="Weight for the alignment InfoNCE auxiliary loss.")
    parser.add_argument("--text_behavior_alignment_gate", action="store_true",
                        help="Also multiply the learned per-fund alignment gate into "
                             "ProspectusTextFusion's text branch (in addition to the "
                             "InfoNCE aux loss). Off → aux loss only.")
    # Option 2: spatial text-stock alignment (additive, off by default).
    parser.add_argument("--use_spatial_alignment", action="store_true",
                        help="Add per-edge text-stock spatial alignment (Option 2). "
                             "Trained with InfoNCE on positive fund→stock edges.")
    parser.add_argument("--spatial_alignment_lambda", type=float, default=0.1,
                        help="Weight for the spatial-alignment InfoNCE auxiliary loss.")
    parser.add_argument("--spatial_alignment_temperature", type=float, default=0.1,
                        help="InfoNCE temperature for SpatialAlignment (softmax scale). "
                             "Default 0.1 (unchanged from original hardcode). Lower → "
                             "sharper softmax, more emphasis on hard negatives; higher → "
                             "softer softmax, more distributed gradient. Only active with "
                             "--use_spatial_alignment.")
    parser.add_argument("--spatial_logit_beta", type=float, default=0.0,
                        help="Initial scale for adding spatial alignment score to "
                             "Stage-1 link logits. 0.0 = no bias added (just train "
                             "the projections via InfoNCE).")
    parser.add_argument("--use_combined_gate", action="store_true",
                        help="Activate the combined posterior α = σ(γ_T·temporal + "
                             "γ_S·spatial + δ). Requires both --use_text_behavior_alignment "
                             "and --use_spatial_alignment.")
    parser.add_argument("--alignment_kl_lambda", type=float, default=0.0,
                        help="Weight for Beta(2,2) prior penalty on combined-gate α. "
                             "0.0 disables. Recommended 0.01-0.1 if --use_combined_gate.")
    # Jacobian-ratio gate (additive, off by default):
    parser.add_argument("--use_jacobian_ratio_gate", action="store_true",
                        help="Augment ProspectusTextFusion gate with per-fund log-ratio "
                             "log(|ΔE'|) - log(|ΔE|). High value means MLP locally "
                             "amplifying small text changes (likely noise) → gate can "
                             "learn to suppress text. Strategy-only embedding.")
    parser.add_argument("--use_spatial_attention_bias", action="store_true",
                        help="Modulate HGT+ attention scores with β·cos(text_f, stock_s) "
                             "for fund→stock and stock→fund edges. Requires "
                             "--use_spatial_alignment to provide the projections. β is a "
                             "learnable scalar in the spatial_alignment module, init 0.")
    parser.add_argument("--spa_negative_scope", type=str, default="global",
                        choices=["global", "per_fund"],
                        help="InfoNCE negative-pool scope for SpatialAlignment. "
                             "'global' (default, original behaviour): each anchor "
                             "contrasts against ALL stocks in the support-window "
                             "union. 'per_fund': restrict negatives to fund f's "
                             "investable universe = stocks f held in any support "
                             "snapshot (∪ the current positive, so cross-entropy "
                             "always has a valid target). Only active with "
                             "--use_spatial_alignment.")
    # C2-4 Text-only auxiliary head (ablation appendix, off by default)
    parser.add_argument("--use_text_only_aux_loss", action="store_true",
                        help="Add a text-only edge prediction head trained with BCE on "
                             "Stage-1 candidate edges. Provides 'text-only AUC' diagnostic "
                             "and a direct BCE gradient to abs_proj (complements Spatial's "
                             "contrastive supervision). Nice-to-have ablation, off by default.")
    parser.add_argument("--text_only_aux_lambda", type=float, default=0.1,
                        help="Weight on text-only BCE auxiliary loss.")
    # Edge memory
    parser.add_argument("--use_edge_memory", action="store_true",
                        help="Enable edge memory GRU + peer attention for Stage 2 weight prediction")
    parser.add_argument("--memory_dim", type=int, default=32,
                        help="GRU hidden dimension for edge memory")
    parser.add_argument("--use_weight_profile", action="store_true",
                        help="Add per-fund weight distribution features to Stage 2 decoder")
    parser.add_argument("--edge_memory_trainable", action="store_true",
                        help="[Abl1] Run GRU in forward pass with gradients (trainable edge memory)")
    parser.add_argument("--edge_weight_profile_mode", type=str, default="fund",
                        choices=["fund", "edge"],
                        help="[Abl2] 'fund'=fund-level profiles (default), 'edge'=per-edge historical weights")
    parser.add_argument("--peer_holders_all_snapshots", action="store_true",
                        help="[Abl3] Build stock_to_holders from ALL snapshots, not just the last one")
    parser.add_argument("--gru_rich_input", action="store_true",
                        help="[Abl4] Enrich GRU input with node features (fund+stock embeddings)")
    parser.add_argument("--peer_residual_gate", action="store_true",
                        help="[Abl5] Use gated residual connection for m_peer instead of concatenation")
    parser.add_argument("--edge_memory_both_stages", action="store_true",
                        help="Activate edge memory in Stage 1 too (trains GRU/attention for link prediction)")
    parser.add_argument("--dual_pathway", action="store_true",
                        help="[Strategy 3] Dual-pathway prediction: separate GNN head and trajectory head "
                             "with learned gate, instead of concatenating m_peer into a single MLP")
    # Edge memory v2 improvements
    parser.add_argument("--stratified_peers", action="store_true",
                        help="[v2] Stratified peer selection: sample evenly across weight-quantiles "
                             "instead of random when >MAX_PEERS holders")
    parser.add_argument("--profile_query", action="store_true",
                        help="[v2] Enrich peer attention query Q with fund portfolio profile features")
    parser.add_argument("--film_decoder", action="store_true",
                        help="[v2] FiLM conditioning: peer memory produces scale/shift that modulates "
                             "GNN prediction instead of concat into MLP")
    parser.add_argument("--max_peers", type=int, default=50,
                        help="Max peer holders per stock for attention (default: 50)")
    parser.add_argument("--memory_dim_v2", type=int, default=None,
                        help="[v2] Override memory_dim for v2 experiments (default: use --memory_dim)")
    # Edge weight trajectory (FiLM into HGT+ value path). Independent of --use_edge_memory.
    parser.add_argument("--use_edge_trajectory", action="store_true",
                        help="Enable per-edge weight-trajectory encoder (GRU over "
                             "[w(t), Δw(t), present_t]) that emits per-snapshot edge "
                             "features γ, β into HGT+ message-passing value path via FiLM. "
                             "Zero-initialized — model is byte-identical to baseline at "
                             "init. Independent of --use_edge_memory (decoder-side trajectory).")
    parser.add_argument("--edge_trajectory_dim", type=int, default=32,
                        help="Hidden dim of edge-trajectory GRU. Default 32 matches --memory_dim.")
    parser.add_argument("--edge_trajectory_log1p", action="store_true",
                        help="Apply log1p to edge weights before feeding the trajectory GRU. "
                             "When on, the GRU sees [log1p(w(t)), log1p(w(t))-log1p(w(t-1)), present_t]. "
                             "Default off (raw percent weights, matching --use_edge_trajectory baseline).")
    parser.add_argument("--edge_trajectory_no_present", action="store_true",
                        help="Drop the present_t channel from the trajectory GRU input. "
                             "When on, GRU input is 2-channel [w(t), Δw(t)] (or their log1p variants "
                             "if --edge_trajectory_log1p is also set). Ablation: tests whether the "
                             "presence indicator carries information beyond what (w, Δw) already convey.")
    parser.add_argument("--edge_trajectory_static", action="store_true",
                        help="Replace the recurrent GRU with a static per-snapshot MLP "
                             "(Linear→GELU→Linear). Ablation: tests whether the recurrence is "
                             "load-bearing or whether instantaneous edge features are enough. "
                             "h(t) then depends only on snapshot t's input — no information flows "
                             "between snapshots through learnable parameters.")
    parser.add_argument("--edge_trajectory_no_delta_w", action="store_true",
                        help="Drop the Δw(t) channel from the edge-trajectory input. "
                             "When combined with --edge_trajectory_static, the static encoder "
                             "emits a 1-channel [w(t)] feature instead of [w(t), Δw(t)]. "
                             "Ablation: tests whether the lookback (Δw) carries signal beyond "
                             "the instantaneous weight. Default off — existing PBSes unaffected.")
    parser.add_argument("--use_tcetf", action="store_true",
                        help="Enable TCETF (Text-Conditioned Edge Trajectory FiLM): "
                             "Δγ(text_f) + Δβ(text_f) zero-init offsets added to the existing "
                             "edge-trajectory FiLM γ, β. Per-edge text is the fund-endpoint's "
                             "abs_proj. Requires --use_edge_trajectory + --use_prospectus. "
                             "Asymmetric — fund-side text only. Default off.")
    parser.add_argument("--use_ttt", action="store_true",
                        help="Enable TTT (Temporal Text Trajectory): GRU over per-fund "
                             "abs_proj across the support window. The GRU output replaces "
                             "instantaneous abs_proj as the input to TCMP's per-layer FiLM. "
                             "Requires --use_tcmp + --use_prospectus. Default off.")
    parser.add_argument("--ttt_dim", type=int, default=128,
                        help="Hidden dim of the TTT GRU output (must match --tcmp_text_dim "
                             "since TTT output feeds TCMP's FiLM input). Default 128.")
    parser.add_argument("--no_text_in_stage2", action="store_true",
                        help="Disable text mechanisms during Stage 2 (weight regression). "
                             "Switches TCMP FiLM off at runtime AND zeros all text aux-loss "
                             "lambdas (TBA, SpA, ToA, KL prior) for Stage 2 training only. "
                             "Stage 1 link prediction is unaffected; the Stage 1 checkpoint "
                             "carries text-shaped representations into Stage 2 implicitly. "
                             "Default off — existing PBSes unaffected.")
    parser.add_argument("--tcetf_text_dim", type=int, default=128,
                        help="Text dim for TCETF input (matches ProspectusTextFusion.ABS_PROJ_DIM=128).")
    parser.add_argument("--tcetf_mode", type=str, default="additive",
                        choices=["additive", "gated"],
                        help="TCETF FiLM combination mode. "
                             "'additive' (default): γ = γ_base + Δγ_text (existing behavior). "
                             "'gated': γ = γ_base · σ(W_gate·text) + Δγ_text — multiplicative gate "
                             "lets text suppress the trajectory-driven γ_base. "
                             "Init: gate.bias=5 so σ≈0.993, γ ≈ γ_base + Δγ_text at start. "
                             "Requires --use_tcetf.")
    parser.add_argument("--tcetf_text_init", action="store_true",
                        help="Text-conditions-trajectory: use text as GRU initial hidden state. "
                             "h_0[e] = Linear(text_init_dim -> traj_dim)(text_per_edge). "
                             "Zero-init so h_0 = 0 at start (matches current behavior). "
                             "Requires --use_edge_trajectory + --use_prospectus. Default off.")
    parser.add_argument("--tcetf_text_init_dim", type=int, default=128,
                        help="Text dim for TCETF text_init projection input (matches ABS_PROJ_DIM=128).")
    parser.add_argument("--tcetf_text_step_input", action="store_true",
                        help="Text-conditions-trajectory (Option 2b): concatenate text_per_edge "
                             "to the (w, dw, present) scalar inputs at every GRU step. "
                             "GRU input_size becomes (2 or 3) + text_step_input_dim (bigger GRU). "
                             "Uses per-snapshot text (_abs_proj_per_snapshot[t]) so text can "
                             "vary across the window. Requires --use_edge_trajectory + --use_prospectus. "
                             "Default off.")
    parser.add_argument("--tcetf_text_step_input_dim", type=int, default=128,
                        help="Text dim for TCETF text_step_input concatenation (matches ABS_PROJ_DIM=128).")
    # Weight-aware contrastive loss (Stage 1)
    parser.add_argument("--weight_contrastive", action="store_true",
                        help="Enable weight-aware contrastive auxiliary loss in Stage 1")
    parser.add_argument("--weight_contrastive_lambda", type=float, default=0.1,
                        help="Weight for contrastive auxiliary loss (default: 0.1)")
    parser.add_argument("--weight_contrastive_temp", type=float, default=0.1,
                        help="Temperature for SupCon loss (default: 0.1)")
    parser.add_argument("--weight_contrastive_bins", type=int, default=4,
                        help="Number of weight quantile bins for SupCon (default: 4)")
    # Misc
    parser.add_argument("--novel_only", action="store_true", default=True)
    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--train_eval_freq", type=int, default=50)
    # Meta-path
    parser.add_argument("--mp_channels", type=int, default=2)
    parser.add_argument("--mp_budget", type=int, default=2)
    parser.add_argument("--mp_topk", type=int, default=200000)
    parser.add_argument("--mp_disable", type=int, default=0)
    parser.add_argument("--mp_weight_mode", type=str, default="mul")
    parser.add_argument("--mp_row_norm", type=str, default="src")
    parser.add_argument("--use_meta_paths", type=int, default=0)
    parser.add_argument("--meta_paths", type=str, default="")
    # Multi-granularity
    parser.add_argument("--mg_enabled", type=int, default=0)
    parser.add_argument("--mg_L", type=int, default=1)
    parser.add_argument("--mg_one_op", type=int, default=1)
    parser.add_argument("--mg_eps_start", type=float, default=0.5)
    parser.add_argument("--mg_eps_end", type=float, default=0.05)
    parser.add_argument("--mg_eps_decay", type=int, default=2000)
    parser.add_argument("--mg_fusion", type=str, default="softmax2")
    parser.add_argument("--mg_time_gate", type=int, default=1)
    parser.add_argument("--mg_time_gate_hidden", type=int, default=32)
    parser.add_argument("--mg_div_lambda", type=float, default=0.0)
    parser.add_argument("--mg_include_metapath_ops", type=int, default=0)
    parser.add_argument("--mg_meta_paths", type=str, default="")

    args = parser.parse_args(args)

    # Normalize device string
    if isinstance(args.device, str):
        dev_lower = args.device.lower()
        if dev_lower == "cpu":
            args.device = "cpu"
        elif dev_lower.startswith("cuda"):
            parts = dev_lower.split(":")
            if len(parts) == 1:
                args.device = "cuda:0"
            else:
                args.device = f"cuda:{parts[1]}"
        elif dev_lower.isdigit():
            args.device = f"cuda:{dev_lower}"
        else:
            args.device = f"cuda:{args.device}"

    args.dynamic = args.model not in Sta_MODEL
    args.homo = args.model in Homo_MODEL
    args.test_full = not args.dynamic
    os.makedirs(args.log_dir, exist_ok=True)
    return args
