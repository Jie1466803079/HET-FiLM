"""
Multi-task Edge Predictor with Scale Difference Loss

Modified version of multitask_edge.py to use scale difference loss
for better handling of small values in percent_tna prediction.
"""

import os
import torch
from torch import nn
from torch.nn import functional as F

# Import the new loss
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from loss.scale_difference_loss import HybridLoss


def _stage1_weighted_bce_enabled() -> bool:
    """Whether to use weight-aware BCE on Stage 1 positives.

    Permanently disabled 2026-05-28: env var STAGE1_WEIGHTED_BCE is now
    ignored. Pre-existing PBS exports of =1 will silently no-op. The CSV
    rows in results_defb_portfolio_2005q3_casmln_neg.csv were trained
    with this enabled; reruns will not reproduce them.
    """
    return False


class MultiTaskEdgePredictor(nn.Module):
    """
    Classification: dot-product + sigmoid.
    Regression: MLP on concatenated endpoints.
    Regression masked to positives; optional persistence term.
    
    **Modified to use Scale Difference Loss for regression.**
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: int,
        reg_hidden_dim: int = 128,
        regression_loss: str = "l2",  # now also supports "hybrid" and "scale_diff"
        class_loss_scale: float = 1.0,
        weight_loss_scale: float = 1.0,
        persist_loss_scale: float = 1.0,
        use_class_weights: bool = True,
        # New parameters for scale difference loss
        scale_loss_alpha: float = 10.0,
        scale_loss_beta: float = 1.0,
        scale_loss_epsilon: float = 0.01,
        # Joint training mode (BCEWithLogits + Huber)
        joint_mode: bool = False,
        huber_delta: float = 1.0,
        pos_weight_override: float = -1.0,
        # Two-stage overrides: use joint-style losses in two-stage pipeline
        use_bce_with_logits: bool = False,
        use_logspace_huber: bool = False,
        # Use MLP instead of dot-product for link classification
        use_cls_mlp: bool = False,
        # PROSPECTUS_INTEGRATION
        prospectus_fusion: nn.Module = None,
        contrastive_loss: nn.Module = None,  # optional, opt-in via --use_contrastive
        prospectus_loader=None,
        text_stock_prior: bool = False,
        text_mlp_decoder: bool = False,
        fund_stock_contrastive: nn.Module = None,
        fund_stock_contrastive_lambda: float = 0.1,
        # Option 1: text-behavior trajectory alignment (additive, off by default).
        text_behavior_alignment: nn.Module = None,
        text_behavior_alignment_lambda: float = 0.1,
        text_behavior_alignment_gate: bool = False,
        # Option 2: spatial text-stock alignment (additive, off by default).
        spatial_alignment: nn.Module = None,
        spatial_alignment_lambda: float = 0.1,
        spatial_logit_beta: float = 0.0,
        use_combined_gate: bool = False,
        alignment_kl_lambda: float = 0.0,
        use_spatial_attention_bias: bool = False,
        spa_negative_scope: str = "global",
        # C2-4: Text-only auxiliary head (ablation, off by default)
        text_only_head: nn.Module = None,
        text_only_aux_lambda: float = 0.1,
        # M1 (IBF probe): intent-conditioned candidate attention. Opt-in via
        # --use_intent_attention. When set, ProspectusTextFusion fusion-into-encoder
        # is BYPASSED (option B): the encoder runs on raw fund features; text enters
        # only at the head via g_f_eff (from intent_attention) and via mu inside
        # intent_attention. Off by default → all existing flag combinations untouched.
        intent_attention: nn.Module = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.class_loss_scale = class_loss_scale
        self.weight_loss_scale = weight_loss_scale
        self.persist_loss_scale = persist_loss_scale
        self.use_class_weights = use_class_weights
        self.regression_loss_type = regression_loss
        self.joint_mode = joint_mode
        self.huber_delta = huber_delta
        self.pos_weight_override = pos_weight_override
        self.use_bce_with_logits = use_bce_with_logits
        self.use_logspace_huber = use_logspace_huber
        self.use_cls_mlp = use_cls_mlp
        # PROSPECTUS_INTEGRATION
        self.use_prospectus = prospectus_fusion is not None
        self.prospectus_fusion = prospectus_fusion
        self.contrastive_loss_fn = contrastive_loss          # optional, may be None
        self.prospectus_loader = prospectus_loader
        self.modality_dropout_p = 0.0  # set by PhaseScheduler each epoch
        self._dataset_keys = None      # set from dataset at init time
        self._last_abs_proj = None     # cached for contrastive loss / text_stock_prior
        self._last_gate_mean = 0.0     # cached from fusion module
        self._last_stock_raw = None    # cached raw pre-GNN stock features (Approach 3 / Approach 4)
        # Per-snapshot abs_proj cache: filled by _apply_prospectus_fusion, consumed by
        # _compute_spatial_attention_bias to avoid redundant text feature lookup + abs_proj
        # forward. Entries are .detach()'d to preserve the original no_grad semantics on
        # the attention-bias path (gradients still flow through text_align_proj /
        # stock_align_proj inside SpatialAlignment.attention_bias).
        self._abs_proj_per_snapshot = []
        # WEIGHT-AWARE CONTRASTIVE LOSS (Stage 1 auxiliary)
        self.weight_contrastive_fn = None
        self.weight_contrastive_lambda = 0.1
        self._last_z = None

        # EDGE MEMORY NETWORK
        self.use_edge_memory = False
        self.edge_memory_module = None
        self.peer_attention = None
        self.use_weight_profile = False
        self._cached_stock_to_holders = None
        self._cached_weight_profiles = None
        self._cached_edge_profiles = None  # [Abl2] per-edge profiles
        self._edge_memory_cache_key = None
        self._edge_memory_active = False
        self._edge_memory_precomputed = {}
        # Ablation flags (set by run_model.py after construction)
        self._edge_memory_trainable = False   # [Abl1]
        self._edge_profile_mode = "fund"      # [Abl2] "fund" or "edge"
        self._peer_residual_gate = False       # [Abl5]
        self._gru_rich_input = False           # [Abl4]
        # EDGE TRAJECTORY FILM (Section 1 of edge-trajectory plan).
        # Configured by run_model.py via configure_edge_trajectory() after
        # construction. Default OFF: byte-identical to baseline.
        self.use_edge_trajectory = False
        self.edge_trajectory_encoder = None
        self.edge_trajectory_film = None
        # Temporal text trajectory (TTT) encoder. None default; opt-in via
        # configure_text_trajectory() (called from run_model.py when --use_ttt).
        # When enabled, TTT GRU encodes the abs_proj sequence and the output
        # replaces instantaneous abs_proj as TCMP's per-layer FiLM input.
        self.text_trajectory_encoder = None
        self._edge_trajectory_metadata = None  # cached metadata[1] for FiLM keys
        self._edge_trajectory_hid_dim = None   # cached for FiLM init
        self._edge_trajectory_heads = None     # cached for FiLM init
        self._dual_pathway = False             # [Strategy 3] dual-pathway prediction
        if self.use_prospectus:
            print(f"[PROSPECTUS] Text fusion enabled: fusion_dim={prospectus_fusion.OUTPUT_DIM}, "
                  f"contrastive={'ON' if contrastive_loss else 'OFF'}")
        # [Approach 3] pre-GNN text-stock prior
        self.text_stock_prior = text_stock_prior
        # [Approach 2] text-aware MLP decoder
        self.text_mlp_decoder = text_mlp_decoder
        if text_stock_prior:
            # Project raw stock features (dim inferred by LazyLinear) → 128 (matches abs_proj_dim)
            self.stock_text_proj = nn.Sequential(
                nn.LazyLinear(128),
                nn.LayerNorm(128),
                nn.GELU(),
            )
            self.text_stock_alpha = nn.Parameter(torch.tensor(0.0))
            print("[CLS] Using pre-GNN text-stock prior (alpha-weighted parallel logit)")
        if text_mlp_decoder:
            # Input: [z_src (hid_dim), z_dst (hid_dim), abs_proj_src (128)]
            # LazyLinear infers the input dim on first forward.
            self.cls_mlp_text = nn.Sequential(
                nn.LazyLinear(reg_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(reg_hidden_dim, 1),
            )
            print("[CLS] Using text-aware MLP decoder (z_src || z_dst || abs_proj_src)")
        # [Approach 4] fund-stock contrastive auxiliary loss (off by default)
        self.fund_stock_contrastive_fn = fund_stock_contrastive
        self.fund_stock_contrastive_lambda = fund_stock_contrastive_lambda
        if fund_stock_contrastive is not None:
            print(f"[PROSPECTUS] Fund-stock contrastive aux loss enabled, "
                  f"lambda={fund_stock_contrastive_lambda}")
        # Option 1: text-behavior trajectory alignment (additive, off by default).
        self.text_behavior_alignment_fn = text_behavior_alignment
        self.text_behavior_alignment_lambda = text_behavior_alignment_lambda
        self.text_behavior_alignment_gate = text_behavior_alignment_gate
        self._last_align_loss = None
        self._last_align_gate = None
        self._last_align_score = None   # raw cosine, for combined-gate (Option 2)
        if text_behavior_alignment is not None:
            print(f"[TBA] Text-behavior alignment enabled "
                  f"(lambda={text_behavior_alignment_lambda}, "
                  f"gate={'ON' if text_behavior_alignment_gate else 'OFF'})")
        # Option 2: spatial text-stock alignment (additive, off by default).
        self.spatial_alignment_fn = spatial_alignment
        self.spatial_alignment_lambda = spatial_alignment_lambda
        self.spatial_logit_beta = float(spatial_logit_beta)
        self.use_combined_gate = bool(use_combined_gate)
        self.alignment_kl_lambda = float(alignment_kl_lambda)
        self.use_spatial_attention_bias = bool(use_spatial_attention_bias)
        # InfoNCE negative scope: "global" (baseline) or "per_fund" (fund's
        # support-window investable universe). When "per_fund", a (N_fund, N_stock)
        # bool mask is built during _apply_prospectus_fusion and cached as
        # ``_last_fund_support_universe``; the trainer threads it into
        # SpatialAlignment.infonce_loss(). Default "global" → byte-identical.
        self.spa_negative_scope = str(spa_negative_scope)
        self._last_fund_support_universe = None
        self._last_spatial_loss = None
        self._last_combined_alpha_kl = None   # KL term, cached for trainer
        # C2-4: Text-only auxiliary head (off when text_only_head is None)
        self.text_only_head = text_only_head
        self.text_only_aux_lambda = float(text_only_aux_lambda)
        self._last_text_only_loss = None
        if text_only_head is not None:
            print(f"[TOA] Text-only auxiliary head enabled "
                  f"(lambda={text_only_aux_lambda}, hidden_dim={text_only_head.hidden_dim})")
        if spatial_alignment is not None:
            print(f"[SPA] Spatial alignment enabled "
                  f"(lambda={spatial_alignment_lambda}, "
                  f"logit_beta={spatial_logit_beta}, "
                  f"combined_gate={'ON' if use_combined_gate else 'OFF'}, "
                  f"kl_lambda={alignment_kl_lambda}, "
                  f"attn_bias={'ON' if use_spatial_attention_bias else 'OFF'})")
        if joint_mode or use_logspace_huber:
            self.huber_criterion = nn.HuberLoss(delta=huber_delta, reduction="none")
        if joint_mode:
            print(f"[LOSS] Joint mode: BCEWithLogitsLoss + HuberLoss(delta={huber_delta}), log1p targets")
        if use_bce_with_logits:
            print(f"[LOSS] Two-stage override: BCEWithLogitsLoss (pos_weight={'auto' if pos_weight_override < 0 else pos_weight_override})")
        if use_logspace_huber:
            print(f"[LOSS] Two-stage override: HuberLoss(delta={huber_delta}) on log1p targets")

        self.reg_mlp = nn.Sequential(
            nn.LazyLinear(reg_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(reg_hidden_dim, 1),
        )

        if use_cls_mlp:
            self.cls_mlp = nn.Sequential(
                nn.LazyLinear(reg_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(reg_hidden_dim, 1),
            )
            print("[CLS] Using MLP decoder for link classification")

        # Choose loss function based on regression_loss parameter
        if regression_loss == "hybrid":
            # Hybrid loss: MSE + Scale Difference
            print(f"[LOSS] Using Hybrid Loss (α={scale_loss_alpha}, β={scale_loss_beta}, ε={scale_loss_epsilon})")
            self.reg_criterion = HybridLoss(
                alpha=scale_loss_alpha,
                beta=scale_loss_beta,
                epsilon=scale_loss_epsilon
            )
            self.use_hybrid_loss = True
        elif regression_loss == "scale_diff":
            # Pure scale difference loss
            print(f"[LOSS] Using Scale Difference Loss only (ε={scale_loss_epsilon})")
            from loss.scale_difference_loss import ScaleDifferenceLoss
            self.reg_criterion = ScaleDifferenceLoss(epsilon=scale_loss_epsilon)
            self.use_hybrid_loss = False
        elif regression_loss == "l2":
            print("[LOSS] Using MSE Loss (baseline)")
            self.reg_criterion = nn.MSELoss(reduction="none")
            self.use_hybrid_loss = False
        else:
            print("[LOSS] Using L1 Loss")
            self.reg_criterion = nn.L1Loss(reduction="none")
            self.use_hybrid_loss = False

        # M1 (IBF probe): intent-conditioned candidate attention. All M1 state is
        # OFF by default; baseline forward path is byte-identical when
        # self.intent_attention is None.
        self.intent_attention = intent_attention
        self.use_intent_attention = intent_attention is not None
        # Per-snapshot caches populated in encode() when M1 is active.
        self._intent_g_feff = None            # (N_funds, g_feff_dim)
        self._intent_ctx_idx = None           # (N_funds, C_max) long
        self._intent_ctx_mask = None          # (N_funds, C_max) bool
        # Per-forward caches populated in decode_logits(), reused by decode_weight().
        self._last_z_intent = None            # (E, OUTPUT_DIM)
        self._last_g_feff_edge = None         # (E, g_feff_dim)
        if self.use_intent_attention:
            self.intent_stage1_head = nn.Sequential(
                nn.LazyLinear(reg_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(reg_hidden_dim, 1),
            )
            self.intent_stage2_head = nn.Sequential(
                nn.LazyLinear(reg_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(reg_hidden_dim, 1),
            )
            print(f"[M1/IBF] IntentConditionedAttention enabled "
                  f"(lambda_mu={getattr(intent_attention, 'lambda_mu', None)}, "
                  f"safety_cap={getattr(intent_attention, 'safety_cap', None)}, "
                  f"staleness_scale={getattr(intent_attention, 'staleness_scale', None)})")
            print("[M1/IBF] ProspectusTextFusion-into-encoder will be BYPASSED "
                  "(option B: encoder runs on raw fund features; text enters at head only)")

        # Opt-in Stage-2 log-space output clamp. Defaults to None (no clamp) so
        # baseline behaviour is preserved. Set STAGE2_LOG_OUT_CLAMP_MAX to a
        # finite float (e.g. 5.7 ≈ log1p(300)) to cap the regression head's
        # log-space output and prevent the rare catastrophic seed where the
        # lazy-init head produces raw-space outputs in the thousands.
        import os as _os_clamp
        _clmax = _os_clamp.environ.get("STAGE2_LOG_OUT_CLAMP_MAX", "").strip()
        _clmin = _os_clamp.environ.get("STAGE2_LOG_OUT_CLAMP_MIN", "").strip()
        try:
            self._stage2_log_out_clamp_max = float(_clmax) if _clmax else None
        except ValueError:
            self._stage2_log_out_clamp_max = None
        try:
            self._stage2_log_out_clamp_min = float(_clmin) if _clmin else None
        except ValueError:
            self._stage2_log_out_clamp_min = None
        if self._stage2_log_out_clamp_max is not None or self._stage2_log_out_clamp_min is not None:
            print(
                f"[STAGE2_CLAMP] log-space output clamp enabled: "
                f"min={self._stage2_log_out_clamp_min} max={self._stage2_log_out_clamp_max}"
            )

    def encode(self, support_graph, *args, **kwargs):
        # M1 (IBF probe): bypass fusion-into-encoder, precompute g_f_eff + context
        # for the LAST support snapshot (scoring time). Encoder runs on raw fund
        # features. When self.use_intent_attention is False, this entire block is
        # skipped and the original prospectus-fusion path runs unchanged.
        if self.use_intent_attention and self.prospectus_loader is not None:
            self._prepare_intent_caches(support_graph)
        elif self.use_prospectus and self.prospectus_loader is not None:
            # PROSPECTUS_INTEGRATION: fuse text features into fund nodes before GNN
            support_graph = self._apply_prospectus_fusion(support_graph)

        # Component 2 attention modulation: build per-snapshot edge_bias_dict for
        # fund↔stock relations. Bias = β·cos(W_T·abs_proj(E_f), W_S·φ(stock_s)).
        # Cosine is symmetric so we reuse the same tensor for the reverse edge.
        if (self.spatial_alignment_fn is not None
                and getattr(self, 'use_spatial_attention_bias', False)):
            edge_bias_per_snapshot = self._compute_spatial_attention_bias(support_graph)
            kwargs['edge_bias_dict_per_snapshot'] = edge_bias_per_snapshot

        # Edge trajectory FiLM: build per-snapshot edge_attr_traj_dict for
        # fund-stock relations. h(t) is causal; γ, β are zero at init.
        if (getattr(self, 'use_edge_trajectory', False)
                and self.edge_trajectory_encoder is not None):
            traj_per_snapshot = self._compute_edge_trajectory_dict(support_graph)
            kwargs['edge_attr_traj_dict_per_snapshot'] = traj_per_snapshot

        # TCMP: pass per-snapshot fund text vectors (cached by prospectus fusion
        # in self._abs_proj_per_snapshot) to the encoder for per-layer FiLM.
        # Only activates if base_model exposes the kwarg AND text was cached.
        if (getattr(self.base_model, 'tcmp_film_layers', None) is not None
                and getattr(self, '_abs_proj_per_snapshot', None) is not None
                and len(self._abs_proj_per_snapshot) > 0
                and self._abs_proj_per_snapshot[0] is not None):
            # TTT: when configured, run the temporal GRU over abs_proj and
            # substitute its output as TCMP's per-layer FiLM input. Otherwise
            # TCMP sees the raw instantaneous abs_proj (existing behavior).
            if getattr(self, 'text_trajectory_encoder', None) is not None:
                text_per_snap = self.text_trajectory_encoder.encode(
                    self._abs_proj_per_snapshot
                )
            else:
                text_per_snap = self._abs_proj_per_snapshot
            kwargs['text_per_snapshot'] = text_per_snap

        z = self.base_model.encode(support_graph, *args, **kwargs)
        # EDGE MEMORY: process support window (Stage 2 only — skipped in Stage 1)
        if self.use_edge_memory and self.edge_memory_module is not None and self._edge_memory_active:
            self._build_edge_memory(support_graph)
        return z

    def _compute_spatial_attention_bias(self, support_graph):
        """Per-snapshot dict of per-edge attention bias for fund-stock relations.

        Returns a list (one per snapshot) of dicts:
            { ('fund','holds_stock','stock'): (E,),
              ('stock','rev_holds_stock','fund'): (E,) }
        For other edge types: nothing (defaults to None in HGTConv).
        """
        FWD = ('fund', 'holds_stock', 'stock')
        REV = ('stock', 'rev_holds_stock', 'fund')
        if isinstance(support_graph, (list, tuple)):
            snapshots = support_graph
        else:
            snapshots = [support_graph]

        result = []
        cache = self._abs_proj_per_snapshot if isinstance(
            self._abs_proj_per_snapshot, list) else []
        for i, g in enumerate(snapshots):
            bias_dict = {}
            if FWD not in g.edge_types or 'fund' not in g.node_types:
                result.append(bias_dict)
                continue
            fund_x = g['fund'].x
            stock_x = g['stock'].x if 'stock' in g.node_types else None
            if fund_x is None or stock_x is None:
                result.append(bias_dict)
                continue
            etype = FWD
            ei = g[etype].edge_index                                  # (2, E)
            # Reuse abs_proj cached by _apply_prospectus_fusion to avoid duplicate
            # text-feature lookup + abs_proj forward. Cached value is already
            # detached, preserving the original no_grad semantics on this path.
            # gradients still flow through SpatialAlignment's own projection heads.
            abs_proj_out = cache[i] if i < len(cache) else None
            if abs_proj_out is None:
                result.append(bias_dict)
                continue
            # Compute per-edge bias for the forward edge; reuse for the reverse (symmetric).
            bias_fwd = self.spatial_alignment_fn.attention_bias(abs_proj_out, stock_x, ei)
            bias_dict[FWD] = bias_fwd
            if REV in g.edge_types:
                bias_dict[REV] = bias_fwd   # same scalar per (fund, stock) pair
            result.append(bias_dict)
        return result

    def _compute_edge_trajectory_dict(self, support_graph):
        """Per-snapshot dict of per-edge (γ, β) FiLM tensors for fund-stock relations.

        Returns a list (one per support snapshot) of dicts:
            { ('fund','holds_stock','stock'): (γ, β),
              ('stock','rev_holds_stock','fund'): (γ, β) }
        γ and β are zero-tensors at module init time, so this is a baseline-
        identical no-op until learning kicks in.
        """
        FWD = ('fund', 'holds_stock', 'stock')
        REV = ('stock', 'rev_holds_stock', 'fund')
        if isinstance(support_graph, (list, tuple)):
            snapshots = list(support_graph)
        else:
            snapshots = [support_graph]

        # Build snapshot dicts in the format EdgeTrajectoryEncoder expects.
        snap_dicts = []
        for g in snapshots:
            if FWD in g.edge_types:
                ei = g[FWD].edge_index
                ew = getattr(g[FWD], 'edge_attr', None)
                if ew is None:
                    ew = torch.zeros(ei.shape[1], device=ei.device)
                else:
                    ew = ew.float().view(-1)
            else:
                ei = torch.zeros((2, 0), dtype=torch.long,
                                 device=snapshots[0]['fund'].x.device
                                 if 'fund' in snapshots[0].node_types
                                 else torch.device('cpu'))
                ew = torch.zeros(0, device=ei.device)
            snap_dicts.append({'edge_index': ei, 'edge_weight': ew})

        # Registry includes query edges from the LAST snapshot (which the trainer
        # uses as the scoring snapshot — its edge_index is the union of
        # support-pos + query-pos + query-neg in the multitask pipeline).
        query_ei = snapshots[-1][FWD].edge_index if FWD in snapshots[-1].edge_types else None

        # Text-conditions-trajectory: when EdgeTrajectoryEncoder.use_text_init is
        # on, pass the earliest-snapshot's abs_proj as text_per_fund so the GRU's
        # initial hidden state h_0[e] = W_init · text[fund_id(e)]. Using
        # _abs_proj_per_snapshot[0] as the text prior; if unavailable, fall back
        # to zero (encoder handles this by using zeros for h_0).
        text_per_fund = None
        if (getattr(self.edge_trajectory_encoder, 'use_text_init', False)
                and isinstance(getattr(self, '_abs_proj_per_snapshot', None), list)
                and len(self._abs_proj_per_snapshot) > 0
                and self._abs_proj_per_snapshot[0] is not None):
            text_per_fund = self._abs_proj_per_snapshot[0]     # (N_funds, 128)

        # Option 2b: pass per-snapshot text to the encoder for per-step
        # concatenation into GRU input. Uses the full _abs_proj_per_snapshot
        # list (which is already aligned with the snapshots list).
        text_per_snapshot = None
        if (getattr(self.edge_trajectory_encoder, 'use_text_step_input', False)
                and isinstance(getattr(self, '_abs_proj_per_snapshot', None), list)):
            text_per_snapshot = self._abs_proj_per_snapshot

        H, inv_per_snap, sorted_keys, max_stock = self.edge_trajectory_encoder.encode(
            snap_dicts, query_edge_index=query_ei,
            text_per_fund=text_per_fund,
            text_per_snapshot=text_per_snapshot,
        )

        # Per-snapshot γ, β dicts keyed by edge_type.
        result = []
        for t, (g, snap_inv) in enumerate(zip(snapshots, inv_per_snap)):
            d = {}
            if FWD not in g.edge_types or snap_inv.numel() == 0:
                result.append(d)
                continue
            # h(t) for the edges present in this snapshot's edge_index.
            h_t_fwd = H[t, snap_inv]                            # (E_t, traj_dim)
            # TCETF: look up per-edge fund text (source endpoint = fund f in fund→stock).
            # Falls back to None when use_tcetf is False or abs_proj is unavailable
            # (e.g., aux paths bypass prospectus fusion). FiLM ignores it gracefully.
            text_per_edge_fwd = None
            if (getattr(self.edge_trajectory_film, 'use_tcetf', False)
                    and isinstance(getattr(self, '_abs_proj_per_snapshot', None), list)
                    and t < len(self._abs_proj_per_snapshot)
                    and self._abs_proj_per_snapshot[t] is not None):
                abs_proj_t = self._abs_proj_per_snapshot[t]    # (N_funds, 128)
                fund_idx_fwd = g[FWD].edge_index[0]            # source = fund
                text_per_edge_fwd = abs_proj_t[fund_idx_fwd]   # (E_t, 128)
            γ_fwd, β_fwd = self.edge_trajectory_film(h_t_fwd, FWD,
                                                    text_per_edge=text_per_edge_fwd)
            d[FWD] = (γ_fwd, β_fwd)
            if REV in g.edge_types:
                # If the registry is empty (no edges anywhere in the window),
                # H has zero E_reg and the searchsorted+clamp would land on -1.
                # Skip REV entirely in that case — there's nothing to modulate.
                if H.shape[1] == 0 or g[REV].edge_index.shape[1] == 0:
                    pass
                else:
                    # Reverse edge order maps the same persistent (fund, stock) pair
                    # → reuse the same h(t). Build inverse indices for REV by
                    # swapping rows of REV's edge_index and looking up in sorted_keys.
                    rev_ei = g[REV].edge_index
                    rev_keys = rev_ei[1] * max_stock + rev_ei[0]
                    rev_inv = torch.searchsorted(sorted_keys, rev_keys).clamp(max=H.shape[1] - 1)
                    h_t_rev = H[t, rev_inv]
                    # TCETF: for stock→fund, the fund endpoint is destination
                    # (rev_ei[1]). Use the fund's text to condition the reverse
                    # FiLM symmetrically with the forward path.
                    text_per_edge_rev = None
                    if (getattr(self.edge_trajectory_film, 'use_tcetf', False)
                            and isinstance(getattr(self, '_abs_proj_per_snapshot', None), list)
                            and t < len(self._abs_proj_per_snapshot)
                            and self._abs_proj_per_snapshot[t] is not None):
                        abs_proj_t = self._abs_proj_per_snapshot[t]
                        text_per_edge_rev = abs_proj_t[rev_ei[1]]  # dest = fund
                    γ_rev, β_rev = self.edge_trajectory_film(h_t_rev, REV,
                                                            text_per_edge=text_per_edge_rev)
                    d[REV] = (γ_rev, β_rev)
            result.append(d)
        return result

    def precompute_all_edge_memories(self, all_datasets):
        """Precompute edge memory for all unique support windows across train/val/test.

        Call once before Stage 2 training loop. Avoids re-running the GRU
        on every forward pass — the support data is static across epochs.

        When _edge_memory_trainable is set, caches raw memories (before W_k/W_v)
        so that W_k/W_v are applied live in forward() and receive gradients.

        Args:
            all_datasets: list of (support_list, query) pairs from all splits.
        """
        if not self.use_edge_memory or self.edge_memory_module is None:
            return
        trainable = self._edge_memory_trainable
        seen = set()
        n_precomputed = 0
        for support_list, _ in all_datasets:
            if not isinstance(support_list, (list, tuple)):
                support_list = [support_list]
            cache_key = tuple(id(s) for s in support_list)
            if cache_key in seen:
                continue
            seen.add(cache_key)
            snapshots, _ = self._extract_snapshots(support_list)
            dev = next(self.parameters()).device
            with torch.inference_mode():
                memory, _, stock_to_holders = self.edge_memory_module.process_support_window(snapshots)
                del memory
                if trainable:
                    self.peer_attention.precompute_raw(stock_to_holders)
                else:
                    self.peer_attention.precompute_kv(stock_to_holders)
            if trainable:
                # Cache raw memories + mask (W_k/W_v applied live in forward)
                kv_state = (
                    self.peer_attention._all_raw.clone() if self.peer_attention._all_raw is not None else None,
                    None,  # placeholder — no separate V when raw
                    self.peer_attention._all_mask.clone() if self.peer_attention._all_mask is not None else None,
                    self.peer_attention._max_sid,
                )
            else:
                kv_state = (
                    self.peer_attention._all_K.clone() if self.peer_attention._all_K is not None else None,
                    self.peer_attention._all_V.clone() if self.peer_attention._all_V is not None else None,
                    self.peer_attention._all_mask.clone() if self.peer_attention._all_mask is not None else None,
                    self.peer_attention._max_sid,
                )
            wp = None
            ep = None  # [Abl2] edge-level profiles
            if self.use_weight_profile:
                n_funds = support_list[-1]['fund'].x.shape[0] if hasattr(support_list[-1]['fund'], 'x') else 0
                if self._edge_profile_mode == "edge":
                    n_stocks = support_list[-1]['stock'].x.shape[0] if hasattr(support_list[-1]['stock'], 'x') else 0
                    from core.models.edge_memory import compute_edge_weight_profiles
                    ep = compute_edge_weight_profiles(snapshots, n_funds, n_stocks)  # keep on CPU (large)
                else:
                    from core.models.edge_memory import compute_weight_profiles
                    wp = compute_weight_profiles(snapshots, n_funds).to(dev)
            self._edge_memory_precomputed[cache_key] = (stock_to_holders, kv_state, wp, ep)
            n_precomputed += 1
        mode_str = "raw (W_k/W_v trainable)" if trainable else "K/V (frozen)"
        print(f"[EDGE_MEMORY] Precomputed {n_precomputed} unique support windows "
              f"({mode_str}, out of {len(all_datasets)} total batches)")

    def configure_text_trajectory(self, text_dim: int = 128, traj_dim: int = 128):
        """Instantiate the Temporal Text Trajectory (TTT) GRU encoder.

        Called from run_model.py when --use_ttt is set. When the encoder is
        present, the GRU output replaces the cached instantaneous abs_proj as
        the text input to TCMP's per-layer FiLM. Idempotent.

        text_dim: input dim — must match ProspectusTextFusion.ABS_PROJ_DIM (128).
        traj_dim: output dim — should match --tcmp_text_dim since TCMP consumes
            the TTT output directly.
        """
        from core.models.text_trajectory import TextTrajectoryEncoder
        if self.text_trajectory_encoder is not None:
            # Idempotent assertion: second call must match the first.
            assert self.text_trajectory_encoder.text_dim == int(text_dim), (
                f"configure_text_trajectory called again with text_dim={text_dim}, "
                f"but already configured with {self.text_trajectory_encoder.text_dim}")
            assert self.text_trajectory_encoder.traj_dim == int(traj_dim), (
                f"configure_text_trajectory called again with traj_dim={traj_dim}, "
                f"but already configured with {self.text_trajectory_encoder.traj_dim}")
            return
        self.text_trajectory_encoder = TextTrajectoryEncoder(
            text_dim=text_dim, traj_dim=traj_dim,
        )
        self.add_module('text_trajectory_encoder', self.text_trajectory_encoder)
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        self.text_trajectory_encoder.to(device)

    def configure_edge_trajectory(self, dim, metadata, hid_dim, heads,
                                  log1p_input=False, no_present=False,
                                  static=False, no_delta_w=False,
                                  use_tcetf=False, tcetf_text_dim=128,
                                  tcetf_mode="additive",
                                  use_text_init=False, text_init_dim=128,
                                  use_text_step_input=False, text_step_input_dim=128):
        """Construct the EdgeTrajectoryEncoder + EdgeTrajectoryFiLM modules.

        Called from run_model.py after the base_model is built so we have
        access to its hetero metadata and hidden_dim/n_heads. Idempotent.

        log1p_input: if True, the trajectory GRU sees log1p-transformed edge
            weights instead of raw percent. Useful for long-tail distributions.
        no_present:  if True, drop the present_t channel from GRU input
            (2-channel input instead of 3-channel). Ablation.
        static:      if True, replace the recurrent GRU with a parameterless
            per-snapshot encoder that emits raw [w(t), Δw(t)] directly into
            the FiLM heads. Overrides dim to 2 so FiLM is Linear(2 → hid_dim).
            Tests whether the GRU's recurrence is load-bearing.
        """
        from core.models.edge_trajectory import (
            EdgeTrajectoryEncoder, EdgeTrajectoryFiLM,
        )
        # Static path defaults to 2-channel [w, Δw]. With no_delta_w, drops to
        # 1-channel [w]. Non-static (GRU) ignores no_delta_w and uses configured dim.
        if bool(static):
            effective_dim = 1 if bool(no_delta_w) else 2
        else:
            effective_dim = int(dim)
        if self.edge_trajectory_encoder is not None:
            # Idempotent: second call must match first call's config exactly.
            # Silent swallow of a mismatched config survives months of
            # experiments unnoticed — make it loud instead.
            assert self._edge_trajectory_hid_dim == hid_dim, (
                f"configure_edge_trajectory called again with hid_dim={hid_dim}, "
                f"but already configured with hid_dim={self._edge_trajectory_hid_dim}"
            )
            assert self._edge_trajectory_heads == heads, (
                f"configure_edge_trajectory called again with heads={heads}, "
                f"but already configured with heads={self._edge_trajectory_heads}"
            )
            assert self.edge_trajectory_encoder.traj_dim == effective_dim, (
                f"configure_edge_trajectory called again with effective_dim="
                f"{effective_dim} (raw dim={dim}, static={static}), but "
                f"already configured with dim={self.edge_trajectory_encoder.traj_dim}"
            )
            assert self.edge_trajectory_encoder.log1p_input == bool(log1p_input), (
                f"configure_edge_trajectory called again with log1p_input={log1p_input}, "
                f"but already configured with log1p_input={self.edge_trajectory_encoder.log1p_input}"
            )
            assert self.edge_trajectory_encoder.no_present == bool(no_present), (
                f"configure_edge_trajectory called again with no_present={no_present}, "
                f"but already configured with no_present={self.edge_trajectory_encoder.no_present}"
            )
            assert self.edge_trajectory_encoder.static == bool(static), (
                f"configure_edge_trajectory called again with static={static}, "
                f"but already configured with static={self.edge_trajectory_encoder.static}"
            )
            return
        self.use_edge_trajectory = True
        self._edge_trajectory_metadata = metadata
        self._edge_trajectory_hid_dim = hid_dim
        self._edge_trajectory_heads = heads
        self.edge_trajectory_encoder = EdgeTrajectoryEncoder(
            traj_dim=effective_dim, log1p_input=bool(log1p_input),
            no_present=bool(no_present), static=bool(static),
            no_delta_w=bool(no_delta_w),
            use_text_init=bool(use_text_init),
            text_init_dim=int(text_init_dim),
            use_text_step_input=bool(use_text_step_input),
            text_step_input_dim=int(text_step_input_dim),
        )
        self.edge_trajectory_film = EdgeTrajectoryFiLM(
            edge_types=metadata[1], traj_dim=effective_dim,
            hidden_dim=hid_dim, heads=heads,
            use_tcetf=bool(use_tcetf), tcetf_text_dim=int(tcetf_text_dim),
            tcetf_mode=str(tcetf_mode),
        )
        # Register as submodules so .to(device) and optimizer pick them up.
        self.add_module('edge_trajectory_encoder', self.edge_trajectory_encoder)
        self.add_module('edge_trajectory_film', self.edge_trajectory_film)
        # The host wrapper may have already been .to(device)'d before this method
        # runs (run_model.py calls configure AFTER `model.to(args.device)`). Push
        # the freshly-constructed submodules onto the same device so the first
        # forward pass doesn't hit a CPU/CUDA mismatch.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        self.edge_trajectory_encoder.to(device)
        self.edge_trajectory_film.to(device)
        print(f"[EDGE_TRAJ] Enabled: dim={effective_dim} (raw={dim}), "
              f"hid_dim={hid_dim}, heads={heads}, edge_types={len(metadata[1])}, "
              f"log1p={bool(log1p_input)}, no_present={bool(no_present)}, "
              f"static={bool(static)}, device={device}")

    def _extract_snapshots(self, support_graph, z=None):
        """Convert support graph list to snapshot dicts for edge memory.

        Args:
            support_graph: list of HeteroData snapshots.
            z: optional node embeddings (dict or tensor) for [Abl4] rich GRU input.

        Returns:
            snapshots: list of dicts with 'edge_index' and 'edge_weight'.
            node_feats: [Abl4] list of per-edge node feature tensors, or None.
        """
        etype = ('fund', 'holds_stock', 'stock')
        snapshots = []
        node_feats = [] if (self._gru_rich_input and z is not None) else None
        for snap in support_graph:
            if etype in snap.edge_types and hasattr(snap[etype], 'edge_index'):
                ei = snap[etype].edge_index
                ew = snap[etype].edge_attr if hasattr(snap[etype], 'edge_attr') and snap[etype].edge_attr is not None else torch.ones(ei.size(1))
                if ew.dim() > 1:
                    ew = ew.squeeze(-1)
                snapshots.append({'edge_index': ei, 'edge_weight': ew.float()})

                # [Abl4] Extract per-edge node features from GNN embeddings
                if node_feats is not None:
                    if isinstance(z, dict):
                        z_fund = z.get('fund', list(z.values())[0])
                        z_stock = z.get('stock', list(z.values())[-1])
                    else:
                        z_fund = z_stock = z
                    fids = ei[0].clamp(max=z_fund.shape[0] - 1)
                    sids = ei[1].clamp(max=z_stock.shape[0] - 1)
                    edge_node_feat = z_fund[fids] + z_stock[sids]  # simple sum
                    node_feats.append(edge_node_feat.detach())
            else:
                dev = next(self.parameters()).device
                snapshots.append({
                    'edge_index': torch.zeros(2, 0, dtype=torch.long, device=dev),
                    'edge_weight': torch.zeros(0, device=dev),
                })
                if node_feats is not None:
                    node_feats.append(torch.zeros(0, self.edge_memory_module.rich_input_dim, device=dev))
        return snapshots, node_feats

    def _build_edge_memory(self, support_graph):
        """Extract fund-stock edges from support snapshots and run GRU."""
        if not isinstance(support_graph, (list, tuple)):
            support_graph = [support_graph]
        cache_key = tuple(id(s) for s in support_graph)
        if cache_key == self._edge_memory_cache_key:
            return
        self._edge_memory_cache_key = cache_key

        # Fast path: use precomputed edge memory if available
        if cache_key in self._edge_memory_precomputed:
            cached = self._edge_memory_precomputed[cache_key]
            if len(cached) == 4:
                stock_to_holders, kv_state, wp, ep = cached
            else:
                stock_to_holders, kv_state, wp = cached
                ep = None
            self._cached_stock_to_holders = stock_to_holders
            slot0, slot1, all_mask, max_sid = kv_state
            if self._edge_memory_trainable and slot1 is None:
                # [Abl1] Raw memories cached — W_k/W_v applied live in forward()
                self.peer_attention._all_raw = slot0
                self.peer_attention._all_K = None
                self.peer_attention._all_V = None
            else:
                self.peer_attention._all_K = slot0
                self.peer_attention._all_V = slot1
                self.peer_attention._all_raw = None
            self.peer_attention._all_mask = all_mask
            self.peer_attention._max_sid = max_sid
            if wp is not None:
                self._cached_weight_profiles = wp
            if ep is not None:
                self._cached_edge_profiles = ep
            return

        # Slow path: compute from scratch (fallback)
        snapshots, _ = self._extract_snapshots(support_graph)
        memory, _, stock_to_holders = self.edge_memory_module.process_support_window(snapshots)
        self._cached_stock_to_holders = stock_to_holders
        self.peer_attention.precompute_kv(stock_to_holders)
        if self.use_weight_profile:
            n_funds = support_graph[-1]['fund'].x.shape[0] if hasattr(support_graph[-1]['fund'], 'x') else 0
            if self._edge_profile_mode == "edge":
                n_stocks = support_graph[-1]['stock'].x.shape[0] if hasattr(support_graph[-1]['stock'], 'x') else 0
                from core.models.edge_memory import compute_edge_weight_profiles
                self._cached_edge_profiles = compute_edge_weight_profiles(snapshots, n_funds, n_stocks)  # CPU
            else:
                from core.models.edge_memory import compute_weight_profiles
                self._cached_weight_profiles = compute_weight_profiles(snapshots, n_funds).to(memory.device)

    def _apply_prospectus_fusion(self, support_graph):  # PROSPECTUS_INTEGRATION
        """Replace fund node features with text-fused features in each snapshot.

        Also accumulates a union of raw pre-GNN stock features across all support
        snapshots in the window into ``self._last_stock_raw``. The accumulation
        walks snapshots in order and updates each stock row whenever a non-zero
        feature row appears, so stocks present in any support snapshot end up
        with their most-recent real features (not zeros from absent quarters).
        This is critical for the text_stock_prior / fund_stock_contrastive heads
        because the dataset's _select_snapshot zero-pads stock rows for absent quarters.
        """
        if isinstance(support_graph, (list, tuple)):
            # 1. Build the stock-feature accumulator FIRST so it's available to
            #    the alignment computation below as well as for cached use later.
            #    Walks support order so later snapshots overwrite earlier ones
            #    for the same stock when both have features.
            accum = None
            for g in support_graph:
                if 'stock' not in g.node_types or not hasattr(g['stock'], 'x'):
                    continue
                sx = g['stock'].x
                if accum is None:
                    accum = sx.clone()
                    continue
                nonzero_mask = sx.abs().sum(dim=-1) > 0   # (N_stock,) bool
                accum[nonzero_mask] = sx[nonzero_mask]
            self._last_stock_raw = accum

            # 1b. Per-fund InfoNCE universe: bool mask (N_fund, N_stock) over the
            #     union of fund→stock edges in the support window. Each row is the
            #     set of stocks fund f held in any support snapshot. Used by
            #     SpatialAlignment.infonce_loss when --spa_negative_scope=per_fund.
            #     Skipped (None) when scope is the default "global".
            #     The mask is ALSO stashed onto self.spatial_alignment_fn so the
            #     trainer's existing infonce_loss() call (no extra args) picks it
            #     up via the module's self._universe_mask_for_batch fallback —
            #     this keeps the trainer code untouched.
            self._last_fund_support_universe = None
            if (self.spatial_alignment_fn is not None
                    and getattr(self, 'spa_negative_scope', 'global') == 'per_fund'
                    and accum is not None):
                FWD = ('fund', 'holds_stock', 'stock')
                n_stock = accum.shape[0]
                n_fund = None
                for g in support_graph:
                    if 'fund' in g.node_types and hasattr(g['fund'], 'x') and g['fund'].x is not None:
                        n_fund = g['fund'].x.shape[0]
                        break
                if n_fund is not None:
                    universe = torch.zeros(
                        n_fund, n_stock, dtype=torch.bool, device=accum.device,
                    )
                    for g in support_graph:
                        if FWD not in g.edge_types:
                            continue
                        ei = g[FWD].edge_index  # (2, E)
                        if ei is None or ei.numel() == 0:
                            continue
                        universe[ei[0], ei[1]] = True
                    self._last_fund_support_universe = universe
                    if not getattr(self, '_spa_per_fund_logged', False):
                        n_active = int(universe.sum().item())
                        n_funds_with_holdings = int(universe.any(dim=1).sum().item())
                        avg_universe = n_active / max(n_funds_with_holdings, 1)
                        print(f"[SPA] Per-fund InfoNCE negatives ON "
                              f"(N_fund={universe.shape[0]}, N_stock={universe.shape[1]}, "
                              f"funds_with_holdings={n_funds_with_holdings}, "
                              f"avg_universe_size={avg_universe:.1f})",
                              flush=True)
                        self._spa_per_fund_logged = True
            # Always sync the mask onto the SPA module so the trainer's call
            # picks up the current state (None when scope is "global" → baseline).
            if self.spatial_alignment_fn is not None:
                self.spatial_alignment_fn._universe_mask_for_batch = (
                    self._last_fund_support_universe
                )

            # 2. Optional: text-behavior alignment on the (t−1, t) snapshot pair.
            #    Caches self._last_align_loss (read by trainer) and self._last_align_gate
            #    (optionally threaded into ProspectusTextFusion below).
            if (self.text_behavior_alignment_fn is not None
                    and len(support_graph) >= 2 and accum is not None):
                self._compute_text_behavior_alignment(
                    support_graph[-2], support_graph[-1], accum,
                )
            else:
                self._last_align_loss = None
                self._last_align_score = None
                self._last_align_gate = None

            # 3. Fuse each snapshot. If --text_behavior_alignment_gate is set, the
            #    same per-fund gate (from the last support pair) is applied across
            #    all snapshots, expressing "trust this fund's text uniformly based
            #    on its most-recent text↔behavior alignment".
            gate = self._last_align_gate if self.text_behavior_alignment_gate else None
            # Pre-size per-snapshot abs_proj cache so _fuse_single_snapshot writes into it.
            self._abs_proj_per_snapshot = [None] * len(support_graph)
            return [self._fuse_single_snapshot(g, i, behavior_gate=gate)
                    for i, g in enumerate(support_graph)]
        # Single-snapshot path — no temporal Δ available, so alignment is skipped.
        self._abs_proj_per_snapshot = [None]
        fused = self._fuse_single_snapshot(support_graph, 0)
        if 'stock' in support_graph.node_types and hasattr(support_graph['stock'], 'x'):
            self._last_stock_raw = support_graph['stock'].x.clone()
        else:
            self._last_stock_raw = None
        self._last_align_loss = None
        self._last_align_score = None
        self._last_align_gate = None
        return fused

    def _prepare_intent_caches(self, support_graph):  # M1 (IBF probe)
        """Populate per-snapshot caches consumed by intent_attention at the head.

        Cold-start invariants enforced by this design:
          1. NO FUTURE LEAKAGE. support_graph spans periods 1..7 of the window
             (= [T-7, T]); the query/scoring quarter is period 8 (= T+1) and is
             NOT in support_graph. This method uses support_graph[-1] (time T)
             as the anchor for both the prospectus lookup and the context build,
             so no holdings or text from T+1 enter C(f, s).
          2. CANDIDATE-NOT-IN-HOLDINGS. The upstream cold-start filter on
             edge_label_index guarantees s is not in f's holdings over [T-7, T].
             By construction the holdings channel of C(f) cannot contain s.
             The siblings channel CAN contain s (a family-mate fund may hold s);
             IntentConditionedAttention.forward de-duplicates so s appears in
             C(f, s) EXACTLY ONCE as the appended candidate column.

        Caches:
            self._intent_g_feff   (N_funds, g_feff_dim)
            self._intent_ctx_idx  (N_funds, C_max) long
            self._intent_ctx_mask (N_funds, C_max) bool

        Does NOT mutate support_graph (encoder runs on raw features under M1).
        """
        from core.models.intent_conditioned_attention import build_context_per_fund

        if isinstance(support_graph, (list, tuple)):
            last_g = support_graph[-1]
            snapshot_offset = len(support_graph) - 1
        else:
            last_g = support_graph
            snapshot_offset = 0

        if 'fund' not in last_g.node_types or not hasattr(last_g['fund'], 'x'):
            self._intent_g_feff = None
            self._intent_ctx_idx = None
            self._intent_ctx_mask = None
            return

        fund_x = last_g['fund'].x
        N_funds = fund_x.shape[0]
        device = fund_x.device

        # Quarter index for prospectus lookup (mirrors _fuse_single_snapshot logic).
        etype = ('fund', 'holds_stock', 'stock')
        if etype in last_g.edge_types and hasattr(last_g[etype], 'edge_time'):
            snapshot_idx = int(last_g[etype].edge_time[0].item())
        else:
            snapshot_idx = snapshot_offset

        if self._dataset_keys is not None and snapshot_idx < len(self._dataset_keys):
            qi = self.prospectus_loader.timestamp_to_quarter_idx(
                self._dataset_keys[snapshot_idx]
            )
        else:
            qi = snapshot_idx

        if qi < 0 or qi >= self.prospectus_loader.N_QUARTERS:
            self._intent_g_feff = None
            self._intent_ctx_idx = None
            self._intent_ctx_mask = None
            return

        # Per-fund text features at the scoring quarter.
        fund_ids = torch.arange(N_funds, dtype=torch.long)
        text_feats = self.prospectus_loader.get_fund_text_features(fund_ids, qi)
        text_feats = {k: v.to(device) for k, v in text_feats.items()}

        # Modality dropout: keep the same semantics as _fuse_single_snapshot — if
        # active and the random draw fires, force null branch via delta_t = -1
        # and zero abs_emb (the M1 null-token will replace it inside compute_g_f_eff).
        if self.training and self.modality_dropout_p > 0:
            if torch.rand(1).item() < self.modality_dropout_p:
                text_feats['abs_emb'] = torch.zeros_like(text_feats['abs_emb'])
                text_feats['delta_t'] = torch.full_like(text_feats['delta_t'], -1)

        self._intent_g_feff = self.intent_attention.compute_g_f_eff(text_feats)  # (N_funds, g_feff_dim)

        # Context (holdings ∪ siblings) at time T — built on CPU, moved per forward.
        # Caps come from the attention module so CLI flags propagate end-to-end.
        ctx_idx, ctx_mask = build_context_per_fund(
            last_g,
            holdings_cap=getattr(self.intent_attention, 'holdings_cap', 512),
            siblings_cap=getattr(self.intent_attention, 'siblings_cap', 128),
        )
        self._intent_ctx_idx = ctx_idx.to(device)
        self._intent_ctx_mask = ctx_mask.to(device)

    def _compute_text_behavior_alignment(self, g_tm1, g_t, stock_features):
        """Compute ΔE_strategy and Δh_portfolio, run alignment module, cache outputs."""
        etype = ('fund', 'holds_stock', 'stock')
        if (etype not in g_t.edge_types or etype not in g_tm1.edge_types
                or 'fund' not in g_t.node_types
                or not hasattr(g_t['fund'], 'x')):
            self._last_align_loss = None
            self._last_align_score = None
            self._last_align_gate = None
            return

        from core.models.text_behavior_alignment import (
            portfolio_summary_per_fund, churn_per_fund,
        )

        device = stock_features.device
        N_funds = g_t['fund'].x.shape[0]
        N_stocks = stock_features.shape[0]

        def _ew(g):
            e = g[etype]
            if hasattr(e, 'edge_weight') and e.edge_weight is not None:
                return e.edge_weight.to(device)
            return torch.ones(e.edge_index.shape[1], device=device)

        ei_t, ew_t = g_t[etype].edge_index.to(device), _ew(g_t)
        ei_tm1, ew_tm1 = g_tm1[etype].edge_index.to(device), _ew(g_tm1)

        summary_t = portfolio_summary_per_fund(ei_t, ew_t, stock_features, N_funds)
        summary_tm1 = portfolio_summary_per_fund(ei_tm1, ew_tm1, stock_features, N_funds)
        churn = churn_per_fund(ei_t, ei_tm1, N_funds, N_stocks)
        # Churn pairs with zero at t-1 → enters delta as a one-sided "Δ" feature.
        summary_t_full = torch.cat([summary_t, churn.unsqueeze(-1)], dim=-1)
        summary_tm1_full = torch.cat(
            [summary_tm1, torch.zeros(N_funds, 1, device=device)], dim=-1,
        )
        delta_h = summary_t_full - summary_tm1_full

        total_t = torch.zeros(N_funds, device=device).index_add_(0, ei_t[0], ew_t)
        total_tm1 = torch.zeros(N_funds, device=device).index_add_(0, ei_tm1[0], ew_tm1)
        behav_valid = (total_t > 0) & (total_tm1 > 0)

        # Map snapshot index to HDF5 quarter index using the LAST snapshot.
        snapshot_idx = None
        if hasattr(g_t[etype], 'edge_time') and g_t[etype].edge_time.numel() > 0:
            snapshot_idx = int(g_t[etype].edge_time[0].item())
        if (snapshot_idx is None or self._dataset_keys is None
                or snapshot_idx >= len(self._dataset_keys)):
            self._last_align_loss = None
            self._last_align_score = None
            self._last_align_gate = None
            return
        qi = self.prospectus_loader.timestamp_to_quarter_idx(
            self._dataset_keys[snapshot_idx]
        )
        if qi < 0 or qi >= self.prospectus_loader.N_QUARTERS:
            self._last_align_loss = None
            self._last_align_score = None
            self._last_align_gate = None
            return

        fund_ids = torch.arange(N_funds, dtype=torch.long)
        delta = self.prospectus_loader.get_strategy_delta(fund_ids, qi)
        delta_E = delta['delta_E'].to(device)
        text_valid = delta['valid'].to(device)
        valid_mask = behav_valid & text_valid

        align_score, gate, L_align = self.text_behavior_alignment_fn(
            delta_E, delta_h, valid_mask,
        )
        self._last_align_loss = L_align
        self._last_align_gate = gate
        self._last_align_score = align_score   # raw cosine (per-fund), for Option 2 combined gate

    def _fuse_single_snapshot(self, graph, snapshot_offset, behavior_gate=None):  # PROSPECTUS_INTEGRATION
        """Fuse text features into fund nodes for a single graph snapshot.

        behavior_gate: optional (N_funds,) per-fund scalar from TextBehaviorAlignment
            (Option 1). When provided, threaded through ProspectusTextFusion.forward
            to scale the text branch. Default None preserves the historical fusion.

        Side effect: writes a detached copy of this snapshot's abs_proj to
        self._abs_proj_per_snapshot[snapshot_offset] (if the list has been pre-sized).
        This cache is consumed by _compute_spatial_attention_bias to avoid recomputing
        the text-feature lookup + abs_proj forward.
        """
        if not hasattr(graph, 'node_types') or 'fund' not in graph.node_types:
            return graph
        if not hasattr(graph['fund'], 'x') or graph['fund'].x is None:
            return graph

        fund_x = graph['fund'].x  # (N_funds, numerical_dim)
        N_funds = fund_x.shape[0]
        device = fund_x.device

        # Determine quarter index from edge_time if available
        etype = ('fund', 'holds_stock', 'stock')
        if etype in graph.edge_types and hasattr(graph[etype], 'edge_time'):
            snapshot_idx = int(graph[etype].edge_time[0].item())
        else:
            snapshot_idx = snapshot_offset

        # Map snapshot_idx to HDF5 quarter index
        if self._dataset_keys is not None and snapshot_idx < len(self._dataset_keys):
            qi = self.prospectus_loader.timestamp_to_quarter_idx(self._dataset_keys[snapshot_idx])
        else:
            qi = snapshot_idx

        if qi < 0 or qi >= self.prospectus_loader.N_QUARTERS:
            return graph

        # Get text features for all fund nodes (using unified IDs = row indices)
        fund_ids = torch.arange(N_funds, dtype=torch.long)
        text_feats = self.prospectus_loader.get_fund_text_features(fund_ids, qi)
        # Move to device
        text_feats = {k: v.to(device) for k, v in text_feats.items()}

        # Apply modality dropout during training: zero abs_emb AND force
        # delta_t=-1 so the staleness gate hard-zeros the text contribution
        # (zeroing delta_t would mean "fresh text" — the wrong thing).
        if self.training and self.modality_dropout_p > 0:
            if torch.rand(1).item() < self.modality_dropout_p:
                text_feats["abs_emb"] = torch.zeros_like(text_feats["abs_emb"])
                text_feats["delta_t"] = torch.full_like(text_feats["delta_t"], -1)

        # Optional: pass fund_ids into the fusion for the per-fund Embedding prior gate.
        if getattr(self.prospectus_fusion, 'use_fund_aware_gate', False):
            text_feats['fund_ids'] = fund_ids.to(device)

        # Optional: compute per-fund log-ratio = log|ΔE'| - log|ΔE| as a local-Jacobian
        # signal for the gate. Detached: this is a fixed signal, not extra grad pressure
        # on abs_proj from the gate path. Strategy-only (risk excluded by construction).
        log_ratio = None
        if getattr(self.prospectus_fusion, 'use_jacobian_ratio_gate', False):
            endpoints = self.prospectus_loader.get_strategy_endpoints(fund_ids, qi)
            strat_t   = endpoints['strategy_t'].to(device)
            strat_tm1 = endpoints['strategy_tm1'].to(device)
            valid     = endpoints['valid'].to(device)
            with torch.no_grad():
                ep_t   = self.prospectus_fusion.abs_proj(strat_t)
                ep_tm1 = self.prospectus_fusion.abs_proj(strat_tm1)
                d_E      = (strat_t - strat_tm1).norm(dim=-1)          # (N_funds,)
                d_Eprime = (ep_t - ep_tm1).norm(dim=-1)                 # (N_funds,)
                log_ratio = (torch.log(d_Eprime + 1e-6)
                             - torch.log(d_E + 1e-6))                    # (N_funds,)
                # Invalid (cold-start / qi=0): force log_ratio=0 (neutral)
                log_ratio = torch.where(valid, log_ratio, torch.zeros_like(log_ratio))

        # Fuse text + numerical (behavior_gate is None unless --text_behavior_alignment_gate)
        h_fund, abs_proj = self.prospectus_fusion(
            text_feats, fund_x, behavior_gate=behavior_gate, log_ratio=log_ratio,
        )

        # Cache abs_proj for contrastive loss / text_stock_prior (from last snapshot only)
        self._last_abs_proj = abs_proj
        self._last_gate_mean = self.prospectus_fusion._last_gate_mean
        # Per-snapshot detached cache for the attention-bias path (Fix A′).
        # Detach preserves the original no_grad semantics on that path.
        if (isinstance(self._abs_proj_per_snapshot, list)
                and 0 <= snapshot_offset < len(self._abs_proj_per_snapshot)):
            self._abs_proj_per_snapshot[snapshot_offset] = abs_proj.detach()

        # Replace fund features in graph (clone to avoid modifying original).
        # When --no_text_fund_feature is set, skip the substitution: graph['fund'].x
        # stays raw numerical (matches HGT+ baseline fund-feature pathway). abs_proj
        # is still cached above for TBA/SpAB/SpatialAlign/ToA aux losses.
        graph = graph.clone()
        if not getattr(self.prospectus_fusion, 'no_text_fund_feature', False):
            graph['fund'].x = h_fund

        return graph

    def _get_pair_embeddings(self, z, edge_label_index):
        if isinstance(z, (list, tuple)):
            z_src = z[0][edge_label_index[0]]
            z_dst = z[1][edge_label_index[1]]
        else:
            if isinstance(z, dict):
                z_dict = z
                node_types = list(z_dict.keys())
                z_src = z_dict[node_types[0]][edge_label_index[0]]
                z_dst = z_dict[node_types[1]][edge_label_index[1]]
            else:
                z_src = z[edge_label_index[0]]
                z_dst = z[edge_label_index[1]]
        return z_src, z_dst

    def decode_logits(self, z, edge_label_index):
        z_src, z_dst = self._get_pair_embeddings(z, edge_label_index)
        # M1 (IBF probe): candidate-conditional path. Early-return through the
        # M1 Stage-1 head with [z_intent || h_f || g_f_eff || h_s]. Active only
        # when self.use_intent_attention is True (off by default → unchanged).
        if self.use_intent_attention and self._intent_g_feff is not None:
            if isinstance(z, dict):
                node_types = list(z.keys())
                h_fund_all = z[node_types[0]]
                h_stock_all = z[node_types[1]]
            elif isinstance(z, (list, tuple)):
                h_fund_all = z[0]
                h_stock_all = z[1]
            else:
                h_fund_all = z
                h_stock_all = z
            z_intent, g_feff_edge = self.intent_attention(
                g_f_eff_all=self._intent_g_feff,
                h_fund_all=h_fund_all,
                h_stock_all=h_stock_all,
                edge_fund_idx=edge_label_index[0],
                edge_stock_idx=edge_label_index[1],
                context_indices=self._intent_ctx_idx,
                context_mask=self._intent_ctx_mask,
            )
            self._last_z_intent = z_intent          # cached for decode_weight
            self._last_g_feff_edge = g_feff_edge    # cached for decode_weight
            head_in = torch.cat([z_intent, z_src, g_feff_edge, z_dst], dim=-1)
            return self.intent_stage1_head(head_in).view(-1)

        # [Approach 2] Text-aware MLP decoder. Computes logits and skips the
        # default decoder branch via early return-to-additive-blocks pattern below.
        # Gated entirely on text_mlp_decoder; off by default → unchanged behaviour.
        approach2_logits = None
        if (getattr(self, 'text_mlp_decoder', False) and self.use_prospectus
                and self._last_abs_proj is not None):
            fund_idx = edge_label_index[0]
            abs_proj_src = self._last_abs_proj[fund_idx]
            edge_feat = torch.cat([z_src, z_dst, abs_proj_src], dim=-1)
            approach2_logits = self.cls_mlp_text(edge_feat).view(-1)

        # Edge memory in Stage 1: enrich cls input with peer attention context
        if (self.use_edge_memory and self._edge_memory_active
                and self._cached_stock_to_holders is not None
                and self.use_cls_mlp):
            stock_idx = edge_label_index[1]
            m_peer = self.peer_attention(z_src, z_dst, self._cached_stock_to_holders, stock_idx)
            cls_parts = [z_src, z_dst, m_peer]
            if self.use_weight_profile and self._cached_weight_profiles is not None:
                fund_idx = edge_label_index[0]
                cls_parts.append(self._cached_weight_profiles[fund_idx])
            logits = self.cls_mlp(torch.cat(cls_parts, dim=-1)).view(-1)
        elif self.use_cls_mlp:
            logits = self.cls_mlp(torch.cat([z_src, z_dst], dim=-1)).view(-1)
        else:
            logits = (z_src * z_dst).sum(dim=-1)

        # [Approach 2] If the text-MLP decoder produced logits, override the
        # default dot-product / cls_mlp result here. The Approach 3 additive
        # block below still applies on top.
        if approach2_logits is not None:
            logits = approach2_logits

        # [Approach 3] Pre-GNN text-stock prior (parallel logit, learned alpha)
        if (getattr(self, 'text_stock_prior', False) and self.use_prospectus
                and self._last_abs_proj is not None and self._last_stock_raw is not None):
            fund_idx = edge_label_index[0]
            stock_idx = edge_label_index[1]
            abs_proj_src = self._last_abs_proj[fund_idx]                # (E, 128)
            raw_stock = self._last_stock_raw[stock_idx]                  # (E, raw_stock_dim)
            stock_text = self.stock_text_proj(raw_stock)                 # (E, 128)
            text_logit = (abs_proj_src * stock_text).sum(dim=-1)         # (E,)
            logits = logits + self.text_stock_alpha * text_logit

        # Option 2: spatial alignment, with optional combined gate.
        # Computes spatial_score per candidate edge, supervises via InfoNCE on positives
        # (in trainer hook), and optionally adds β·α·spatial_score to Stage-1 logits.
        if (self.spatial_alignment_fn is not None and self._last_abs_proj is not None
                and self._last_stock_raw is not None):
            sa_fund_idx = edge_label_index[0]
            sa_stock_idx = edge_label_index[1]
            spatial_score = self.spatial_alignment_fn.cosine_score(
                self._last_abs_proj, self._last_stock_raw,
                torch.stack([sa_fund_idx, sa_stock_idx], dim=0),
            )                                                            # (E,)

            # Combined posterior α (per edge), if enabled and TBA is also running.
            if (self.use_combined_gate
                    and getattr(self, '_last_align_score', None) is not None):
                alpha = self.spatial_alignment_fn.combined_gate(
                    self._last_align_score, spatial_score, sa_fund_idx,
                )                                                         # (E,)
                self._last_combined_alpha = alpha
                # Optional KL prior penalty cached for trainer to add to loss
                if self.alignment_kl_lambda > 0:
                    from core.models.spatial_alignment import beta_prior_penalty
                    self._last_combined_alpha_kl = beta_prior_penalty(alpha)
                else:
                    self._last_combined_alpha_kl = None
            else:
                alpha = None
                self._last_combined_alpha_kl = None

            # Optional: add β · α · spatial_score (or just β · spatial_score if no combined gate) to logits.
            if self.spatial_logit_beta > 0:
                if alpha is not None:
                    logits = logits + self.spatial_logit_beta * alpha * spatial_score
                else:
                    logits = logits + self.spatial_logit_beta * spatial_score
        return logits

    def decode_weight(self, z, edge_label_index, edge_fund_baseline=None):
        z_src, z_dst = self._get_pair_embeddings(z, edge_label_index)
        # M1 (IBF probe): candidate-conditional path. Stage-2 head consumes
        # [h_f || g_f_eff || h_s || z_intent]. Reuses cache from decode_logits
        # when available (typical forward call sequence); recomputes otherwise.
        if self.use_intent_attention and self._intent_g_feff is not None:
            if (self._last_z_intent is not None
                    and self._last_z_intent.shape[0] == edge_label_index.shape[1]):
                z_intent = self._last_z_intent
                g_feff_edge = self._last_g_feff_edge
            else:
                if isinstance(z, dict):
                    node_types = list(z.keys())
                    h_fund_all = z[node_types[0]]
                    h_stock_all = z[node_types[1]]
                elif isinstance(z, (list, tuple)):
                    h_fund_all = z[0]
                    h_stock_all = z[1]
                else:
                    h_fund_all = z
                    h_stock_all = z
                z_intent, g_feff_edge = self.intent_attention(
                    g_f_eff_all=self._intent_g_feff,
                    h_fund_all=h_fund_all,
                    h_stock_all=h_stock_all,
                    edge_fund_idx=edge_label_index[0],
                    edge_stock_idx=edge_label_index[1],
                    context_indices=self._intent_ctx_idx,
                    context_mask=self._intent_ctx_mask,
                )
            head_in = torch.cat([z_src, g_feff_edge, z_dst, z_intent], dim=-1)
            raw_pred = self.intent_stage2_head(head_in).view(-1)
            if edge_fund_baseline is not None and self.joint_mode:
                baseline = edge_fund_baseline.to(raw_pred.device)
                return raw_pred + torch.log1p(baseline.clamp(min=0))
            return raw_pred

        # EDGE MEMORY: enrich decoder input with peer memory + weight profile
        if self.use_edge_memory and self._cached_stock_to_holders is not None:
            stock_idx = edge_label_index[1]
            fund_idx = edge_label_index[0]

            # [v2] profile_query: pass fund profiles into attention query
            _fund_prof = None
            if getattr(self, '_profile_query', False) and self._cached_weight_profiles is not None:
                _fund_prof = self._cached_weight_profiles[fund_idx]
            m_peer = self.peer_attention(
                z_src, z_dst, self._cached_stock_to_holders, stock_idx,
                fund_profiles=_fund_prof,
            )

            # [v2] FiLM conditioning: m_peer produces scale/shift on GNN prediction
            if getattr(self, '_film_decoder', False) and hasattr(self, '_film_scale_net'):
                gnn_pred = self.reg_mlp(torch.cat([z_src, z_dst], dim=-1))
                scale = self._film_scale_net(m_peer)
                shift = self._film_shift_net(m_peer)
                raw_pred = (scale * gnn_pred + shift).view(-1)

            # [Strategy 3] Dual-pathway: separate GNN and trajectory heads with learned gate
            elif self._dual_pathway and hasattr(self, 'traj_mlp'):
                w_gnn = self.reg_mlp(torch.cat([z_src, z_dst], dim=-1)).view(-1)

                traj_parts = [m_peer]
                if self._edge_profile_mode == "edge" and self._cached_edge_profiles is not None:
                    fids = fund_idx.clamp(max=self._cached_edge_profiles.shape[0] - 1).cpu()
                    sids = stock_idx.clamp(max=self._cached_edge_profiles.shape[1] - 1).cpu()
                    traj_parts.append(self._cached_edge_profiles[fids, sids].to(z_src.device))
                elif self.use_weight_profile and self._cached_weight_profiles is not None:
                    traj_parts.append(self._cached_weight_profiles[fund_idx])
                w_traj = self.traj_mlp(torch.cat(traj_parts, dim=-1)).view(-1)

                alpha = torch.sigmoid(
                    self.pathway_gate(torch.cat([z_src, z_dst, m_peer], dim=-1))
                ).view(-1)
                raw_pred = alpha * w_gnn + (1 - alpha) * w_traj

            # [Abl5] Gated residual connection instead of concatenation
            elif self._peer_residual_gate and hasattr(self, '_gate_proj'):
                gate = torch.sigmoid(self._gate_proj(torch.cat([z_src, z_dst, m_peer], dim=-1)))
                z_base = torch.cat([z_src, z_dst], dim=-1)
                m_proj = self._peer_to_base(m_peer)
                decoder_input = z_base + gate * m_proj

                profile_parts = []
                if self._edge_profile_mode == "edge" and self._cached_edge_profiles is not None:
                    fids = fund_idx.clamp(max=self._cached_edge_profiles.shape[0] - 1).cpu()
                    sids = stock_idx.clamp(max=self._cached_edge_profiles.shape[1] - 1).cpu()
                    profile_parts.append(self._cached_edge_profiles[fids, sids].to(z_src.device))
                elif self.use_weight_profile and self._cached_weight_profiles is not None:
                    profile_parts.append(self._cached_weight_profiles[fund_idx])
                if profile_parts:
                    decoder_input = torch.cat([decoder_input] + profile_parts, dim=-1)
                raw_pred = self.reg_mlp(decoder_input).view(-1)
            else:
                parts = [z_src, z_dst, m_peer]
                if self._edge_profile_mode == "edge" and self._cached_edge_profiles is not None:
                    fids = fund_idx.clamp(max=self._cached_edge_profiles.shape[0] - 1).cpu()
                    sids = stock_idx.clamp(max=self._cached_edge_profiles.shape[1] - 1).cpu()
                    parts.append(self._cached_edge_profiles[fids, sids].to(z_src.device))
                elif self.use_weight_profile and self._cached_weight_profiles is not None:
                    parts.append(self._cached_weight_profiles[fund_idx])
                decoder_input = torch.cat(parts, dim=-1)
                raw_pred = self.reg_mlp(decoder_input).view(-1)
        else:
            decoder_input = torch.cat([z_src, z_dst], dim=-1)
            raw_pred = self.reg_mlp(decoder_input).view(-1)
        if edge_fund_baseline is not None and self.joint_mode:
            baseline = edge_fund_baseline.to(raw_pred.device)
            return raw_pred + torch.log1p(baseline.clamp(min=0))
        return raw_pred

    def compute_contrastive_loss(self, novel_stock_sets=None):  # PROSPECTUS_INTEGRATION
        """Compute InfoNCE contrastive loss on cached abs_proj embeddings."""
        if (not self.use_prospectus or self.contrastive_loss_fn is None
                or self._last_abs_proj is None or novel_stock_sets is None):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.contrastive_loss_fn(self._last_abs_proj, novel_stock_sets)

    def compute_fund_stock_contrastive_loss(self, pos_edge_index):  # PROSPECTUS_INTEGRATION
        """[Approach 4] Compute fund-stock InfoNCE on cached abs_proj and raw stock features.

        pos_edge_index: (2, E_pos) fund→stock indices for positive edges in the query.
        Returns scalar loss, or 0 if prerequisites are unmet.
        """
        if (not self.use_prospectus
                or self.fund_stock_contrastive_fn is None
                or self._last_abs_proj is None
                or self._last_stock_raw is None
                or pos_edge_index is None
                or pos_edge_index.numel() == 0):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.fund_stock_contrastive_fn(
            self._last_abs_proj, self._last_stock_raw, pos_edge_index
        )

    def forward(self, support_graph, edge_label_index, edge_fund_baseline=None):
        z = self.encode(support_graph)
        self._last_z = z
        logits = self.decode_logits(z, edge_label_index)
        weight_pred = self.decode_weight(z, edge_label_index, edge_fund_baseline=edge_fund_baseline)
        # Opt-in log-space output clamp (see __init__ for env-var control).
        if (self._stage2_log_out_clamp_max is not None
                or self._stage2_log_out_clamp_min is not None):
            weight_pred = weight_pred.clamp(
                min=self._stage2_log_out_clamp_min,
                max=self._stage2_log_out_clamp_max,
            )
        return logits, weight_pred

    def compute_weight_contrastive_loss(self, edge_label_index, edge_label, edge_weight):
        """Weight-aware contrastive auxiliary loss using cached z from last forward pass."""
        if self.weight_contrastive_fn is None or self._last_z is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.weight_contrastive_fn(
            self._last_z, edge_label_index, edge_label, edge_weight,
        )

    def compute_loss(
        self,
        logits,
        weight_pred,
        edge_label,
        edge_weight,
        edge_prev_weight=None,
        edge_continue=None,
    ):
        if self.joint_mode:
            return self._compute_joint_loss(logits, weight_pred, edge_label, edge_weight)

        # Classification loss
        if _stage1_weighted_bce_enabled() and edge_weight is not None and edge_weight.numel() == edge_label.numel():
            # Stage-1 weight-aware BCE (Strategy A): per-sample weight =
            # 1 + log1p(edge_weight) for positives, 1.0 for negatives.
            # Forces the encoder to internalise edge-weight semantics.
            pos = edge_label > 0
            ew_safe = torch.clamp(edge_weight, min=0.0)
            sample_w = torch.where(
                pos, 1.0 + torch.log1p(ew_safe), torch.ones_like(edge_label)
            )
            cls_loss = F.binary_cross_entropy_with_logits(
                logits, edge_label, weight=sample_w
            )
        elif self.use_bce_with_logits:
            # Joint-style: BCEWithLogitsLoss with auto pos_weight
            if self.pos_weight_override > 0:
                pw = logits.new_tensor([self.pos_weight_override])
            else:
                n_pos = (edge_label > 0).float().sum().clamp(min=1.0)
                n_neg = (edge_label == 0).float().sum().clamp(min=1.0)
                pw = (n_neg / n_pos).unsqueeze(0)
            cls_loss = F.binary_cross_entropy_with_logits(logits, edge_label, pos_weight=pw)
        elif self.use_class_weights:
            probs = torch.sigmoid(logits)
            pos = (edge_label > 0).float().sum().clamp(min=1.0)
            neg = (edge_label == 0).float().sum().clamp(min=1.0)
            total = pos + neg
            pos_w = total / (2.0 * pos)
            neg_w = total / (2.0 * neg)
            weight_vec = torch.where(edge_label > 0, pos_w, neg_w)
            cls_loss = F.binary_cross_entropy(probs, edge_label, weight=weight_vec)
        else:
            probs = torch.sigmoid(logits)
            cls_loss = F.binary_cross_entropy(probs, edge_label)
        cls_loss = cls_loss * self.class_loss_scale

        # Regression loss
        if self.use_logspace_huber:
            # Joint-style: HuberLoss on log1p targets, positive edges only
            pos_mask = (edge_weight > 0)
            if pos_mask.any():
                log_target = torch.log1p(edge_weight[pos_mask])
                reg_loss = self.huber_criterion(weight_pred[pos_mask], log_target).mean()
            else:
                reg_loss = weight_pred.new_tensor(0.0)
        else:
            # Original regression path (MODIFIED to use hybrid loss if enabled)
            mask = (edge_weight > 0).float()
            if mask.sum() > 0:
                if self.use_hybrid_loss:
                    # Hybrid loss returns (loss, loss_dict)
                    masked_pred = weight_pred[mask > 0]
                    masked_target = edge_weight[mask > 0]
                    reg_loss, loss_dict = self.reg_criterion(masked_pred, masked_target)
                    # Store loss components for logging (optional)
                    self.last_loss_dict = loss_dict
                else:
                    # Standard loss (MSE/L1/ScaleDiff) with reduction='none'
                    if self.regression_loss_type in ["hybrid", "scale_diff"]:
                        # ScaleDifferenceLoss returns scalar
                        masked_pred = weight_pred[mask > 0]
                        masked_target = edge_weight[mask > 0]
                        reg_loss = self.reg_criterion(masked_pred, masked_target)
                    else:
                        # MSE/L1 with reduction='none'
                        reg_loss_all = self.reg_criterion(weight_pred, edge_weight)
                        reg_loss = (reg_loss_all * mask).sum() / mask.sum()
            else:
                reg_loss = weight_pred.new_tensor(0.0)

        # Persistence loss (unchanged)
        if edge_prev_weight is not None and edge_continue is not None:
            cont_mask = (edge_continue > 0).float()
            if cont_mask.sum() > 0:
                if self.use_hybrid_loss:
                    masked_pred = weight_pred[cont_mask > 0]
                    masked_prev = edge_prev_weight[cont_mask > 0]
                    prev_loss, _ = self.reg_criterion(masked_pred, masked_prev)
                else:
                    if self.regression_loss_type in ["hybrid", "scale_diff"]:
                        masked_pred = weight_pred[cont_mask > 0]
                        masked_prev = edge_prev_weight[cont_mask > 0]
                        prev_loss = self.reg_criterion(masked_pred, masked_prev)
                    else:
                        prev_loss_all = self.reg_criterion(weight_pred, edge_prev_weight)
                        prev_loss = (prev_loss_all * cont_mask).sum() / cont_mask.sum()
            else:
                prev_loss = weight_pred.new_tensor(0.0)
        else:
            prev_loss = weight_pred.new_tensor(0.0)

        total_loss = cls_loss + self.weight_loss_scale * reg_loss + self.persist_loss_scale * prev_loss
        return total_loss, cls_loss, reg_loss

    def _compute_joint_loss(self, logits, weight_pred, edge_label, edge_weight):
        """Joint BCE-with-logits + positive-only Huber loss (log-space targets)."""
        # Classification: BCEWithLogitsLoss (numerically stable, no sigmoid needed)
        if self.pos_weight_override > 0:
            pw = logits.new_tensor([self.pos_weight_override])
        else:
            # Auto: neg_count / pos_count
            n_pos = (edge_label > 0).float().sum().clamp(min=1.0)
            n_neg = (edge_label == 0).float().sum().clamp(min=1.0)
            pw = (n_neg / n_pos).unsqueeze(0)
        cls_loss = F.binary_cross_entropy_with_logits(logits, edge_label, pos_weight=pw)

        # Regression: Huber on positive edges only, targets in log-space
        # log1p compresses [0.01, 100] -> [0.01, 4.62], stabilizing training
        pos_mask = (edge_label > 0)
        if pos_mask.any():
            log_target = torch.log1p(edge_weight[pos_mask])
            reg_loss_all = self.huber_criterion(weight_pred[pos_mask], log_target)
            reg_loss = reg_loss_all.mean()
        else:
            reg_loss = logits.new_tensor(0.0)

        total = self.class_loss_scale * cls_loss + self.weight_loss_scale * reg_loss
        return total, cls_loss, reg_loss

