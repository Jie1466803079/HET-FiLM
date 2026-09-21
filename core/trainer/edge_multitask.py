
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score, mean_absolute_error, average_precision_score, mean_squared_error, r2_score
from scipy import stats
from tqdm import tqdm

from core.utils import EarlyStopping, setup_seed

def _forward(model, batch):
    """
    Standardize forward pass for both training and evaluation.
    Returns: loss_all, cls_loss, reg_loss, logits, w_pred, query
    """
    # Unpack batch: (support_data, query_data)
    # Support data provides graph structure/features
    # Query data provides the edges to predict
    support, query = batch
    
    # Run model
    # Model signature: forward(x_dict, edge_index_dict, edge_time_dict=None) -> embedding
    # But MultiTaskEdgePredictor.forward(support, edge_label_index) -> (logits, w_pred)
    edge_fund_baseline = getattr(query, 'edge_fund_baseline', None)
    logits, w_pred = model(support, query.edge_label_index, edge_fund_baseline=edge_fund_baseline)
    
    # Compute Losses (if labels are present)
    edge_label = query.edge_label
    edge_weight = getattr(query, 'edge_weight', None)
    edge_prev_weight = getattr(query, 'edge_prev_weight', None)
    edge_continue = getattr(query, 'edge_continue', None)
    
    # Check if we have weights (might be None in some rare cases, but usually present)
    if edge_weight is None:
        # Dummy loss if no weights
        loss = logits.sum() * 0
        cls_loss = loss
        reg_loss = loss
    else:
        loss, cls_loss, reg_loss = model.compute_loss(
            logits, w_pred, edge_label, edge_weight, edge_prev_weight, edge_continue
        )
    
    return loss, cls_loss, reg_loss, logits, w_pred, query


def train_epoch(model, optimizer, data_loader, grad_clip=0.0, scaler=None):
    model.train()
    total_loss = 0
    num_batches = 0
    use_amp = scaler is not None

    iterator = data_loader if not isinstance(data_loader, list) else data_loader

    tail_alpha = float(getattr(model, 'tail_reweight_alpha', 0.0) or 0.0)
    tail_q67 = float(getattr(model, 'tail_reweight_q67', 0.0) or 0.0)
    tail_active = tail_alpha > 0 and tail_q67 > 0
    if tail_active and not getattr(model, '_tail_logged', False):
        print(f"[TAIL_REWEIGHT] active: alpha={tail_alpha}, q67_train={tail_q67:.6f}",
              flush=True)
        model._tail_logged = True

    for batch in iterator:
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                loss, cls_loss, reg_loss, _, w_pred, query = _forward(model, batch)
        else:
            loss, cls_loss, reg_loss, _, w_pred, query = _forward(model, batch)

        if tail_active and getattr(query, 'edge_weight', None) is not None:
            ew = query.edge_weight
            pos_mask = query.edge_label > 0
            if pos_mask.any():
                pred_pos = w_pred[pos_mask]
                target_pos = ew[pos_mask]
                sample_w = 1.0 + tail_alpha * (target_pos > tail_q67).float()
                if getattr(model, 'joint_mode', False) or getattr(model, 'use_logspace_huber', False):
                    log_target = torch.log1p(torch.clamp(target_pos, min=0))
                    huber_delta = float(getattr(model, 'huber_criterion', None).delta) \
                        if getattr(model, 'huber_criterion', None) is not None else 1.0
                    per = F.huber_loss(pred_pos, log_target, reduction='none', delta=huber_delta)
                else:
                    per = F.mse_loss(pred_pos, target_pos, reduction='none')
                if per.ndim > 1:
                    per = per.mean(dim=tuple(range(1, per.ndim)))
                weighted_reg = (per * sample_w).mean()
                w_scale = float(getattr(model, 'weight_loss_scale', 1.0))
                loss = loss + w_scale * (weighted_reg - reg_loss)
                reg_loss = weighted_reg.detach()

        if (hasattr(model, 'fund_stock_contrastive_fn')
                and model.fund_stock_contrastive_fn is not None):
            if not getattr(model, '_fscontrast_gate_seen', False):
                print(f"[PROSPECTUS][DIAG] fund-stock contrastive gate entered "
                      f"(query type={type(query).__name__})", flush=True)
                model._fscontrast_gate_seen = True
            try:
                eli = query.edge_label_index
                el  = query.edge_label
                pos_mask = (el > 0)
                if not getattr(model, '_fscontrast_pos_seen', False):
                    print(f"[PROSPECTUS][DIAG] edge_label shape={tuple(el.shape)} "
                          f"dtype={el.dtype} n_pos={int(pos_mask.sum().item())} "
                          f"of n_total={el.shape[0]}", flush=True)
                    model._fscontrast_pos_seen = True
                if pos_mask.any():
                    pos_edge_index = eli[:, pos_mask]
                    aux = model.compute_fund_stock_contrastive_loss(pos_edge_index)
                    loss = loss + model.fund_stock_contrastive_lambda * aux
                    if not getattr(model, '_fscontrast_logged_ok', False):
                        try:
                            print(f"[PROSPECTUS] Fund-stock contrastive aux loss "
                                  f"firing OK (first call: aux={float(aux):.4f}, "
                                  f"lambda*aux={float(model.fund_stock_contrastive_lambda * aux):.4f})",
                                  flush=True)
                        except Exception as inner_e:
                            print(f"[PROSPECTUS][WARN] firing-OK print failed: {inner_e!r}", flush=True)
                        model._fscontrast_logged_ok = True
            except Exception as e:
                if not getattr(model, '_fscontrast_warned', False):
                    print(f"[PROSPECTUS][WARN] fund-stock contrastive aux loss raised "
                          f"{type(e).__name__}: {e!r}; continuing without it for this epoch.",
                          flush=True)
                    model._fscontrast_warned = True

        # Option 1: text-behavior trajectory alignment auxiliary loss.
        # Cached by MultiTaskEdgePredictor._compute_text_behavior_alignment during forward.
        if (getattr(model, 'text_behavior_alignment_fn', None) is not None
                and getattr(model, '_last_align_loss', None) is not None
                and float(model.text_behavior_alignment_lambda) > 0):
            try:
                aux = model._last_align_loss
                loss = loss + model.text_behavior_alignment_lambda * aux
                if not getattr(model, '_tba_logged_ok', False):
                    fn = model.text_behavior_alignment_fn
                    print(f"[TBA] Text-behavior alignment firing OK "
                          f"(aux={float(aux):.4f}, "
                          f"lambda*aux={float(model.text_behavior_alignment_lambda * aux):.4f}, "
                          f"align_mean={fn._last_align_mean:.3f}, "
                          f"gate_mean={fn._last_gate_mean:.3f}, "
                          f"valid_frac={fn._last_valid_frac:.3f})",
                          flush=True)
                    model._tba_logged_ok = True
            except Exception as e:
                if not getattr(model, '_tba_warned', False):
                    print(f"[TBA][WARN] alignment aux loss raised "
                          f"{type(e).__name__}: {e!r}; continuing without it for this epoch.",
                          flush=True)
                    model._tba_warned = True

        # Option 2: spatial text-stock alignment auxiliary loss + optional KL prior.
        # The spatial InfoNCE is computed here (using cached _last_abs_proj + _last_stock_raw
        # from the model's forward, and the positive edges from the current batch's query).
        if (getattr(model, 'spatial_alignment_fn', None) is not None
                and float(model.spatial_alignment_lambda) > 0):
            try:
                if (getattr(model, '_last_abs_proj', None) is not None
                        and getattr(model, '_last_stock_raw', None) is not None):
                    eli = query.edge_label_index
                    el = query.edge_label
                    pos_mask = (el > 0)
                    if pos_mask.any():
                        pos_edge_index = eli[:, pos_mask]
                        L_spat = model.spatial_alignment_fn.infonce_loss(
                            model._last_abs_proj, model._last_stock_raw, pos_edge_index,
                        )
                        loss = loss + model.spatial_alignment_lambda * L_spat
                        # Optional KL prior on the combined alpha (cached in decode_logits)
                        if (model.use_combined_gate
                                and getattr(model, '_last_combined_alpha_kl', None) is not None
                                and float(model.alignment_kl_lambda) > 0):
                            kl = model._last_combined_alpha_kl
                            loss = loss + model.alignment_kl_lambda * kl

                        if not getattr(model, '_spa_logged_ok', False):
                            fn = model.spatial_alignment_fn
                            kl_str = (f", kl={float(model._last_combined_alpha_kl):.4f}, "
                                      f"alpha_mean={fn._last_combined_alpha_mean:.3f}"
                                      if model.use_combined_gate and getattr(model, '_last_combined_alpha_kl', None) is not None
                                      else "")
                            print(f"[SPA] Spatial alignment firing OK "
                                  f"(L_spat={float(L_spat):.4f}, "
                                  f"lambda*L={float(model.spatial_alignment_lambda * L_spat):.4f}, "
                                  f"spatial_pos={fn._last_spatial_pos_mean:.3f}, "
                                  f"spatial_neg={fn._last_spatial_neg_mean:.3f}, "
                                  f"n_pos={fn._last_n_pos}{kl_str})", flush=True)
                            model._spa_logged_ok = True
            except Exception as e:
                if not getattr(model, '_spa_warned', False):
                    print(f"[SPA][WARN] spatial alignment aux loss raised "
                          f"{type(e).__name__}: {e!r}; continuing without it for this epoch.",
                          flush=True)
                    model._spa_warned = True

        # C2-4: Text-only edge prediction head (BCE on Stage-1 candidates).
        # Provides a 'text-only AUC' diagnostic and a direct BCE gradient to abs_proj,
        # complementing the contrastive gradient from Spatial Alignment's InfoNCE.
        if (getattr(model, 'text_only_head', None) is not None
                and float(model.text_only_aux_lambda) > 0):
            try:
                if (getattr(model, '_last_abs_proj', None) is not None
                        and getattr(model, '_last_stock_raw', None) is not None
                        and getattr(query, 'edge_label_index', None) is not None):
                    eli = query.edge_label_index
                    el = query.edge_label.float()
                    fund_idx, stock_idx = eli[0], eli[1]
                    text_per_edge = model._last_abs_proj[fund_idx]
                    stock_per_edge = model._last_stock_raw[stock_idx]
                    text_only_logits = model.text_only_head(text_per_edge, stock_per_edge)
                    L_text_only = F.binary_cross_entropy_with_logits(text_only_logits, el)
                    loss = loss + model.text_only_aux_lambda * L_text_only
                    model._last_text_only_loss = float(L_text_only.detach().item())
                    if not getattr(model, '_toa_logged_ok', False):
                        fn = model.text_only_head
                        print(f"[TOA] Text-only aux head firing OK "
                              f"(L_text_only={float(L_text_only):.4f}, "
                              f"lambda*L={float(model.text_only_aux_lambda * L_text_only):.4f}, "
                              f"logit_mean={fn._last_logit_mean:.3f}, "
                              f"n_edges={fn._last_n_edges})", flush=True)
                        model._toa_logged_ok = True
            except Exception as e:
                if not getattr(model, '_toa_warned', False):
                    print(f"[TOA][WARN] text-only aux loss raised "
                          f"{type(e).__name__}: {e!r}; continuing without it for this epoch.",
                          flush=True)
                    model._toa_warned = True

        if (hasattr(model, 'weight_contrastive_fn')
                and model.weight_contrastive_fn is not None):
            try:
                eli = query.edge_label_index
                el = query.edge_label
                ew = getattr(query, 'edge_weight', None)
                if ew is not None:
                    wc_aux = model.compute_weight_contrastive_loss(eli, el, ew)
                    loss = loss + model.weight_contrastive_lambda * wc_aux
                    if not getattr(model, '_wc_logged_ok', False):
                        print(f"[WEIGHT_CONTRASTIVE] Auxiliary loss firing OK "
                              f"(first call: aux={float(wc_aux):.4f}, "
                              f"lambda*aux={float(model.weight_contrastive_lambda * wc_aux):.4f})",
                              flush=True)
                        model._wc_logged_ok = True
            except Exception as e:
                if not getattr(model, '_wc_warned', False):
                    print(f"[WEIGHT_CONTRASTIVE][WARN] aux loss raised "
                          f"{type(e).__name__}: {e!r}; continuing without it.",
                          flush=True)
                    model._wc_warned = True

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(1, num_batches)


@torch.no_grad()
def compute_val_loss(model, data) -> float:
    """Compute average forward-pass loss on a dataset (val or test)."""
    model.eval()
    iterator = data if isinstance(data, list) else [data]
    total_loss = 0.0
    n = 0
    for batch in iterator:
        loss, _, _, _, _, _ = _forward(model, batch)
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


def _safe_auc(lbl, pr):
    """Return 0.0 if labels are single-class to avoid sklearn warnings."""
    lbl = np.asarray(lbl)
    if lbl.size == 0:
        return 0.0
    pr = np.asarray(pr, dtype=float)
    finite = np.isfinite(pr)
    if not finite.all():
        lbl = lbl[finite]
        pr = pr[finite]
    if lbl.size == 0:
        return 0.0
    uniq = np.unique(lbl)
    if uniq.size < 2:
        return 0.0
    try:
        return float(roc_auc_score(lbl, pr))
    except:
        return 0.0

def _safe_ap(lbl, pr):
    lbl = np.asarray(lbl)
    pr = np.asarray(pr, dtype=float)
    finite = np.isfinite(pr)
    if not finite.all():
        lbl = lbl[finite]
        pr = pr[finite]
    if lbl.size == 0:
        return 0.0
    uniq = np.unique(lbl)
    if uniq.size < 2:
        return 0.0
    try:
        return float(average_precision_score(lbl, pr))
    except Exception:
        return 0.0


@torch.no_grad()
def evaluate(model, data, skip_cls: bool = False, skip_reg: bool = False,
             train_seen_nodes: dict = None, return_loss: bool = False) -> Dict[str, float]:
    model.eval()

    def _topk_recall(probs, labels, src, k_list):
        recalls = {}
        for k in k_list:
            hits = 0
            total = 0
            funds = np.unique(src)
            for fund in funds:
                mask = src == fund
                rel = labels[mask]
                if rel.sum() == 0:
                    continue
                total += rel.sum()
                order = np.argsort(-probs[mask])
                topk = order[: min(k, len(order))]
                hits += rel[topk].sum()
            recalls[f"recall@{k}"] = float(hits / total) if total > 0 else 0.0
        return recalls

    def _topk_precision_reg(w_pred, ew_true, src, k_list):
        precisions = {}
        for k in k_list:
            prec_sum = 0.0
            n_funds = 0
            for fund in np.unique(src):
                mask = src == fund
                if mask.sum() < k:
                    continue
                wp = w_pred[mask]
                wt = ew_true[mask]
                top_pred = set(np.argsort(-wp)[:k].tolist())
                top_true = set(np.argsort(-wt)[:k].tolist())
                prec_sum += len(top_pred & top_true) / k
                n_funds += 1
            precisions[f"precision@{k}"] = float(prec_sum / n_funds) if n_funds > 0 else 0.0
        return precisions

    def _rank_ic_within_fund(w_pred, ew_true, src, min_per_fund=3):
        ic_sum = 0.0
        n_funds = 0
        for fund in np.unique(src):
            mask = src == fund
            if mask.sum() < min_per_fund:
                continue
            wp = w_pred[mask]
            wt = ew_true[mask]
            try:
                rho, _ = stats.spearmanr(wp, wt)
            except Exception:
                rho = float('nan')
            if np.isfinite(rho):
                ic_sum += float(rho)
                n_funds += 1
        return float(ic_sum / n_funds) if n_funds > 0 else 0.0

    def _compute_metrics(probs, labels, w_pred, edge_weight_np, prefix=""):
        m = {}
        p = f"{prefix}_" if prefix else ""
        if not skip_cls:
            m[f"{p}auc"] = _safe_auc(labels, probs)
            m[f"{p}ap"] = _safe_ap(labels, probs)
        if not skip_reg:
            pos = labels > 0
            if pos.any() and pos.sum() > 1:
                mae = mean_absolute_error(edge_weight_np[pos], w_pred[pos])
                mse = mean_squared_error(edge_weight_np[pos], w_pred[pos])
                r2 = r2_score(edge_weight_np[pos], w_pred[pos])
                m[f"{p}mae"] = float(mae)
                m[f"{p}rmse"] = float(np.sqrt(mse))
                m[f"{p}r2"] = float(r2)
            else:
                m[f"{p}mae"] = 0.0
                m[f"{p}rmse"] = 0.0
                m[f"{p}r2"] = 0.0
        return m

    joint_log_space = getattr(model, 'joint_mode', False) or getattr(model, 'use_logspace_huber', False)

    def eval_one(batch) -> Tuple[Dict[str, float], float]:
        loss, _, _, logits, w_pred, query = _forward(model, batch)
        batch_loss = loss.item() if return_loss else 0.0

        probs = logits.sigmoid().cpu().numpy()
        labels = query.edge_label.cpu().numpy()
        w_pred_np = w_pred.cpu().numpy()
        if joint_log_space:
            w_pred_np = np.expm1(np.clip(w_pred_np, -10, 10))
        ew_np = query.edge_weight.cpu().numpy() if hasattr(query, 'edge_weight') else np.zeros_like(labels)

        metrics = _compute_metrics(probs, labels, w_pred_np, ew_np)

        if train_seen_nodes is not None and hasattr(query, "edge_label_index"):
            ei = query.edge_label_index.cpu().numpy()
            fund_seen = train_seen_nodes.get('fund', set())
            stock_seen = train_seen_nodes.get('stock', set())
            seen_mask = np.array([
                int(ei[0, i]) in fund_seen and int(ei[1, i]) in stock_seen
                for i in range(ei.shape[1])
            ], dtype=bool)
            unseen_mask = ~seen_mask

            if seen_mask.any():
                metrics.update(_compute_metrics(
                    probs[seen_mask], labels[seen_mask], w_pred_np[seen_mask],
                    ew_np[seen_mask], prefix="seen"))
            if unseen_mask.any():
                metrics.update(_compute_metrics(
                    probs[unseen_mask], labels[unseen_mask], w_pred_np[unseen_mask],
                    ew_np[unseen_mask], prefix="unseen"))

        cont_mask = getattr(query, "edge_continue", None)
        prev_w = getattr(query, "edge_prev_weight", None)

        if cont_mask is not None and prev_w is not None and not skip_reg:
            cont = cont_mask.cpu().numpy()
            prev = prev_w.cpu().numpy()
            current = ew_np

            pos = labels > 0
            lw_mask = pos & (cont > 0) & (np.abs(current - prev) >= 0.5)
            if lw_mask.any():
                metrics["mae_large_change"] = float(mean_absolute_error(current[lw_mask], w_pred_np[lw_mask]))

            entry_mask = pos & (cont == 0)
            if entry_mask.any():
                e_true = current[entry_mask]
                e_pred = w_pred_np[entry_mask]
                metrics["mae_entry"] = float(mean_absolute_error(e_true, e_pred))
                metrics["rmse_entry"] = float(np.sqrt(mean_squared_error(e_true, e_pred)))
                if e_true.size > 1:
                    metrics["r2_entry"] = float(r2_score(e_true, e_pred))

        if hasattr(query, "edge_label_index"):
            src_idx = query.edge_label_index[0].cpu().numpy()
            if not skip_cls:
                k_list = [10, 25, 50]
                metrics.update({f"overall_{k}": v for k, v in _topk_recall(probs, labels, src_idx, k_list).items()})

                if cont_mask is not None:
                    cont_np = cont_mask.cpu().numpy()
                    entry_mask = (labels > 0) & (cont_np == 0)
                    if entry_mask.any():
                        entry_recalls = _topk_recall(probs[entry_mask], labels[entry_mask], src_idx[entry_mask], k_list)
                        metrics.update({f"entry_{k}": v for k, v in entry_recalls.items()})
                    # Entry AUC/AP: positives = entries; negatives = label==0 (true negatives).
                    # Continuations (label>0 & cont>0) are excluded — the model is asked to
                    # discriminate NEW edges from non-edges, not from continuations.
                    sel = ((labels > 0) & (cont_np == 0)) | (labels == 0)
                    if sel.any():
                        # Build binary labels: 1 for entry, 0 for true negative
                        entry_labels = ((labels > 0) & (cont_np == 0)).astype(float)
                        metrics["entry_auc"] = _safe_auc(entry_labels[sel], probs[sel])
                        metrics["entry_ap"]  = _safe_ap(entry_labels[sel], probs[sel])

            if not skip_reg and cont_mask is not None:
                entry_mask = (labels > 0) & (cont_mask.cpu().numpy() == 0)
                if entry_mask.sum() >= 3:
                    # Top-K precision: extended to include @10 alongside existing @3 and @5
                    reg_k = [3, 5, 10]
                    entry_prec = _topk_precision_reg(w_pred_np[entry_mask], ew_np[entry_mask], src_idx[entry_mask], reg_k)
                    metrics.update({f"entry_{k}": v for k, v in entry_prec.items()})
                    metrics["entry_rank_ic_within_fund"] = _rank_ic_within_fund(w_pred_np[entry_mask], ew_np[entry_mask], src_idx[entry_mask])

                    # Scale-based regression metrics on entries (new in 2026-05).
                    # All scoped to entry_mask = (labels > 0) & (cont == 0).
                    # Under NEW_EDGE_STRICT=1, cont is always 0, so this reduces to positive edges
                    # but the explicit cont == 0 guard keeps the metric correct outside that regime.
                    p = w_pred_np[entry_mask].astype(np.float64)
                    t = ew_np[entry_mask].astype(np.float64)
                    # Log-MAE: MAE on log1p targets (matches Stage 2's log-space loss).
                    metrics["log_mae_entry"] = float(np.mean(np.abs(np.log1p(np.clip(p, 0, None)) - np.log1p(np.clip(t, 0, None)))))
                    # Dollar-weighted MAE: large holdings dominate the error.
                    t_sum = float(t.sum())
                    metrics["weighted_mae_entry"] = float((t * np.abs(p - t)).sum() / t_sum) if t_sum > 0 else 0.0
                    # MAPE: only over entries with t > 0 (in practice all entries since label>0 implies t>0).
                    valid = t > 0
                    if valid.any():
                        metrics["mape_entry"] = float(np.mean(np.abs(p[valid] - t[valid]) / t[valid]))
                    else:
                        metrics["mape_entry"] = 0.0
                    # SMAPE: bounded in [0, 2], robust to small targets.
                    denom = np.abs(p) + np.abs(t)
                    metrics["smape_entry"] = float(np.mean(2 * np.abs(p - t) / np.maximum(denom, 1e-12)))
                    # MAPD (aka WAPE): pooled |err| / pooled |truth|.
                    metrics["mapd_entry"] = float(np.abs(p - t).sum() / t_sum) if t_sum > 0 else 0.0

            # Classification-side: Entry Recall@K — recall of entries within top-K predictions per fund.
            # Computed when classification head is active (skip_cls=False) — i.e., Stage 1 or Joint.
            # NOTE: this OVERWRITES the buggy entry_recall@K written above (which mistakenly
            # called _topk_recall on the entry-masked subset). Here we rank entries among ALL
            # candidates per fund using prob scores.
            if not skip_cls and cont_mask is not None:
                cont_np = cont_mask.cpu().numpy()
                entry_labels_binary = ((labels > 0) & (cont_np == 0)).astype(float)
                if entry_labels_binary.sum() > 0:
                    for k in [3, 5, 10]:
                        er_k = _topk_recall(probs, entry_labels_binary, src_idx, [k])
                        metrics[f"entry_recall@{k}"] = er_k[f"recall@{k}"]

        return metrics, batch_loss

    if isinstance(data, list):
        results = [eval_one(b) for b in data]
        all_metrics = [r[0] for r in results]
        keys = set().union(*all_metrics)
        merged = {k: float(np.mean([m.get(k, 0.0) for m in all_metrics])) for k in keys}
        if return_loss:
            merged["val_loss"] = float(np.mean([r[1] for r in results]))
        return merged
    metrics, batch_loss = eval_one(data)
    if return_loss:
        metrics["val_loss"] = batch_loss
    return metrics


@torch.no_grad()
def save_edge_predictions(model, data, save_path, joint_log_space=False):
    """Run inference and save per-edge predictions to .npz file."""
    model.eval()

    def _collect(batch, batch_idx):
        _, _, _, logits, w_pred, query = _forward(model, batch)
        probs = logits.sigmoid().cpu().numpy()
        labels = query.edge_label.cpu().numpy()
        w_pred_np = w_pred.cpu().numpy()
        if joint_log_space:
            w_pred_np = np.expm1(np.clip(w_pred_np, -10, 10))
        ew_np = query.edge_weight.cpu().numpy() if hasattr(query, 'edge_weight') else np.zeros_like(labels)
        ei = query.edge_label_index.cpu().numpy()  # [2, num_edges]
        t_np = np.full(len(labels), batch_idx, dtype=np.int32)
        return ei, probs, w_pred_np, labels, ew_np, t_np

    batches = data if isinstance(data, list) else [data]
    all_ei, all_probs, all_wpred, all_labels, all_ew, all_t = [], [], [], [], [], []
    for i, b in enumerate(batches):
        ei, probs, wp, labels, ew, t = _collect(b, i)
        all_ei.append(ei)
        all_probs.append(probs)
        all_wpred.append(wp)
        all_labels.append(labels)
        all_ew.append(ew)
        all_t.append(t)

    np.savez_compressed(save_path,
        fund_idx=np.concatenate([e[0] for e in all_ei]),
        stock_idx=np.concatenate([e[1] for e in all_ei]),
        prob=np.concatenate(all_probs),
        weight_pred=np.concatenate(all_wpred),
        label=np.concatenate(all_labels),
        weight_true=np.concatenate(all_ew),
        batch_idx=np.concatenate(all_t),
    )
    total = sum(len(p) for p in all_probs)
    print(f"Saved {total} per-edge predictions to {save_path}")


def train_till_end(
    model,
    optimizer,
    dataset,
    args,
    max_epochs: int,
    patience: int,
    disable_progress: bool = False,
    writer=None,
    grad_clip: float = 0.0,
    monitor: str = "auc",
    skip_cls: bool = False,
    skip_reg: bool = False,
    best_ckpt_path: str = None,
    scheduler=None,
    warmup_epochs: int = 0,
):
    train_seen_nodes = getattr(dataset, 'train_seen_nodes', None)
    setup_seed(args.seed)
    start_time = time.time()
    # Linear LR warmup (opt-in via --stage2_warmup_epochs; default 0 = off,
    # behavior byte-identical to before). Ramp is applied at the TOP of each
    # epoch, before train_epoch, so even the very first optimizer steps run at
    # reduced LR — targeting the fresh-regression-head first-step shock that
    # causes Stage-2 blowups on large-scale targets. The plateau scheduler is
    # held off until the ramp completes so it cannot decay the warming LR.
    warmup_epochs = int(warmup_epochs or 0)
    _warmup_base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    if warmup_epochs > 0:
        print(f"[WARMUP] Linear LR warmup enabled: {warmup_epochs} epochs "
              f"(base LRs {_warmup_base_lrs}); plateau scheduler deferred until ramp completes.")
    mode = "min" if monitor in {"mae", "mae_scaled", "mae_unscaled", "mae_entry", "mae_large_change", "val_loss"} else "max"
    best_val = float("inf") if mode == "min" else -float("inf")
    final_test = {}
    earlystop = EarlyStopping(mode=mode, patience=patience)

    # AMP: enable mixed-precision when running on CUDA
    use_amp = getattr(args, 'use_amp', True) and args.device != 'cpu' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("[PERF] AMP (mixed precision) enabled")

    # torch.compile: JIT-compile the model for faster execution (PyTorch 2+)
    compiled_model = model
    if getattr(args, 'use_compile', False) and hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(model)
            print("[PERF] torch.compile enabled")
        except Exception as e:
            print(f"[PERF] torch.compile failed ({e}), falling back to eager mode")
            compiled_model = model

    # How often to run train-set and test-set evaluation (saves ~2 forward passes on skipped epochs)
    train_eval_freq = getattr(args, 'train_eval_freq', 50)
    
    with tqdm(range(max_epochs), disable=disable_progress) as bar:
        for epoch in bar:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                frac = (epoch + 1) / warmup_epochs
                for pg, base in zip(optimizer.param_groups, _warmup_base_lrs):
                    pg['lr'] = base * frac
                if epoch == 0 or epoch == warmup_epochs - 1:
                    print(f"[WARMUP] epoch {epoch}: LR set to {optimizer.param_groups[0]['lr']:.2e}")
            loss = train_epoch(compiled_model, optimizer, dataset.train_dataset, grad_clip, scaler=scaler)

            # Val: always evaluate (needed for early stopping); return_loss folds in compute_val_loss
            val_metrics = evaluate(compiled_model, dataset.val_dataset, skip_cls=skip_cls, skip_reg=skip_reg,
                                   train_seen_nodes=train_seen_nodes, return_loss=True)
            val_loss_value = val_metrics.get("val_loss", 0.0)

            if monitor == "val_loss":
                current = val_loss_value
            else:
                current = val_metrics.get(monitor, 0.0)
            better = current < best_val if mode == "min" else current > best_val

            # Test: only evaluate when val improves (saves ~5 min/epoch)
            if better:
                test_metrics = evaluate(compiled_model, dataset.test_dataset, skip_cls=skip_cls, skip_reg=skip_reg,
                                        train_seen_nodes=train_seen_nodes)
                best_val = current
                final_test = test_metrics
                if best_ckpt_path:
                    torch.save(model.state_dict(), best_ckpt_path)
            else:
                test_metrics = final_test

            # Train eval: only every N epochs (diagnostic only, not used for early stopping)
            if epoch % train_eval_freq == 0:
                train_metrics = evaluate(compiled_model, dataset.train_dataset, skip_cls=skip_cls, skip_reg=skip_reg)
            else:
                train_metrics = {}

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if not (warmup_epochs > 0 and epoch < warmup_epochs):
                        scheduler.step(current)
                else:
                    warmup = getattr(scheduler, '_warmup_epochs', 0)
                    if epoch < warmup:
                        frac = (epoch + 1) / warmup
                        for pg in optimizer.param_groups:
                            pg['lr'] = getattr(scheduler, '_base_lr', pg['lr']) * frac
                    else:
                        scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                if epoch % 10 == 0:
                    bar.set_postfix_str(f"LR: {current_lr:.2e}")

            postfix = {"loss": loss, "best_val": best_val}
            if monitor == "val_loss":
                postfix["val_loss"] = current
            if not skip_cls:
                postfix.update(
                    val_auc=val_metrics.get("auc", 0.0),
                    test_auc=test_metrics.get("auc", 0.0),
                )
                if train_metrics:
                    postfix["train_auc"] = train_metrics.get("auc", 0.0)
            if not skip_reg:
                postfix.update(
                    val_mae=val_metrics.get("mae", 0.0),
                    test_mae=test_metrics.get("mae", 0.0),
                    val_r2=val_metrics.get("r2", 0.0),
                    test_r2=test_metrics.get("r2", 0.0),
                )
            if scheduler is not None:
                postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            bar.set_postfix(**postfix)

            if writer:
                writer.add_scalar("Model/train_loss", loss, epoch)
                if train_metrics:
                    writer.add_scalar("Model/train_auc", train_metrics.get("auc", 0.0), epoch)
                writer.add_scalar("Model/val_auc", val_metrics.get("auc", 0.0), epoch)
                writer.add_scalar("Model/val_mae", val_metrics.get("mae", 0.0), epoch)
                writer.add_scalar("Model/test_auc", test_metrics.get("auc", 0.0), epoch)
                writer.add_scalar("Model/test_mae", test_metrics.get("mae", 0.0), epoch)
                writer.add_scalar("Model/val_r2", val_metrics.get("r2", 0.0), epoch)
                writer.add_scalar("Model/test_r2", test_metrics.get("r2", 0.0), epoch)
                if scheduler is not None:
                    writer.add_scalar("Model/learning_rate", optimizer.param_groups[0]['lr'], epoch)

            epoch_msg = (
                f"Epoch {epoch:4d} | "
                f"train_loss: {loss:.4f} | "
                f"val_loss: {val_loss_value:.4f} | "
                f"val_AUC: {val_metrics.get('auc', 0.0):.4f} | "
                f"val_AP: {val_metrics.get('ap', 0.0):.4f} | "
                f"val_MAE: {val_metrics.get('mae', 0.0):.4f} | "
                f"val_RMSE: {val_metrics.get('rmse', 0.0):.4f} | "
                f"val_R2: {val_metrics.get('r2', 0.0):.4f} | "
                f"val_EntryRankIC: {val_metrics.get('entry_rank_ic_within_fund', 0.0):.4f} | "
                f"val_EntryP@5: {val_metrics.get('entry_precision@5', 0.0):.4f} | "
                f"val_EntryAUC: {val_metrics.get('entry_auc', 0.0):.4f} | "
                f"val_EntryMAE: {val_metrics.get('mae_entry', 0.0):.4f}"
            )
            if hasattr(model, '_last_gate_mean') and model._last_gate_mean > 0:
                epoch_msg += f" | gate={model._last_gate_mean:.3f}"
            print(epoch_msg)

            if earlystop.step(current):
                break

    duration = time.time() - start_time
    result = {
        "val_auc": best_val,
        "test_auc": final_test.get("auc", 0.0),
        "test_ap": final_test.get("ap", 0.0),
        "test_mae": final_test.get("mae", 0.0),
        "test_rmse": final_test.get("rmse", 0.0),
        "test_r2": final_test.get("r2", 0.0),
        "epoch": epoch,
        "time": duration,
        "time_per_epoch": duration / (epoch + 1),
    }
    for k, v in final_test.items():
        rk = f"test_{k}" if not k.startswith("test_") else k
        if rk not in result:
            result[rk] = v
    return result


# PROSPECTUS_INTEGRATION: three-phase training with PhaseScheduler
def train_till_end_prospectus(
    model,
    optimizer,
    dataset,
    args,
    phase_scheduler,
    max_epochs: int,
    patience: int,
    disable_progress: bool = False,
    grad_clip: float = 0.0,
    monitor: str = "auc",
    skip_cls: bool = False,
    skip_reg: bool = False,
    best_ckpt_path: str = None,
):
    """Three-phase training loop for prospectus-enhanced models.

    Uses PhaseScheduler to:
      Phase A (epochs 0-9):   Freeze backbone, train text fusion only
      Phase B (epochs 10-39): Unfreeze all, modality dropout 0.80→0.20
      Phase C (epochs 40+):   All params, dropout 0.20 fixed
    """
    train_seen_nodes = getattr(dataset, 'train_seen_nodes', None)
    setup_seed(args.seed)
    start_time = time.time()
    mode = "min" if monitor in {"mae", "mae_scaled", "mae_unscaled", "mae_entry",
                                 "mae_large_change", "val_loss"} else "max"
    best_val = float("inf") if mode == "min" else -float("inf")
    final_test = {}
    earlystop = EarlyStopping(mode=mode, patience=patience)

    use_amp = getattr(args, 'use_amp', True) and args.device != 'cpu' and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("[PERF] AMP (mixed precision) enabled")

    train_eval_freq = getattr(args, 'train_eval_freq', 50)

    with tqdm(range(max_epochs), disable=disable_progress) as bar:
        for epoch in bar:
            phase_cfg = phase_scheduler.apply_phase(epoch, model, optimizer)
            model.modality_dropout_p = phase_cfg.modality_dropout_p

            loss = train_epoch(model, optimizer, dataset.train_dataset, grad_clip, scaler=scaler)

            val_metrics = evaluate(model, dataset.val_dataset,
                                   skip_cls=skip_cls, skip_reg=skip_reg,
                                   train_seen_nodes=train_seen_nodes, return_loss=True)
            val_loss_value = val_metrics.get("val_loss", 0.0)

            if monitor == "val_loss":
                current = val_loss_value
            else:
                current = val_metrics.get(monitor, 0.0)
            better = current < best_val if mode == "min" else current > best_val

            if better:
                test_metrics = evaluate(model, dataset.test_dataset,
                                        skip_cls=skip_cls, skip_reg=skip_reg,
                                        train_seen_nodes=train_seen_nodes)
                best_val = current
                final_test = test_metrics
                if best_ckpt_path:
                    torch.save(model.state_dict(), best_ckpt_path)
            else:
                test_metrics = final_test

            if epoch % train_eval_freq == 0:
                train_metrics = evaluate(model, dataset.train_dataset,
                                         skip_cls=skip_cls, skip_reg=skip_reg)
            else:
                train_metrics = {}

            postfix = {"loss": loss, "best_val": best_val, "phase": phase_cfg.phase_name[:7]}
            if monitor == "val_loss":
                postfix["val_loss"] = current
            if not skip_cls:
                postfix.update(
                    val_auc=val_metrics.get("auc", 0.0),
                    test_auc=test_metrics.get("auc", 0.0),
                )
                if train_metrics:
                    postfix["train_auc"] = train_metrics.get("auc", 0.0)
            if not skip_reg:
                postfix.update(
                    val_mae=val_metrics.get("mae", 0.0),
                    test_mae=test_metrics.get("mae", 0.0),
                    val_r2=val_metrics.get("r2", 0.0),
                    test_r2=test_metrics.get("r2", 0.0),
                )
            postfix["lr_bb"] = f"{optimizer.param_groups[0]['lr']:.1e}"
            postfix["lr_txt"] = f"{optimizer.param_groups[1]['lr']:.1e}"
            postfix["mdrop"] = f"{phase_cfg.modality_dropout_p:.2f}"
            bar.set_postfix(**postfix)

            gate_val = getattr(model, '_last_gate_mean', 0.0)
            epoch_msg = (
                f"Epoch {epoch:4d} [{phase_cfg.phase_name}] | "
                f"loss: {loss:.4f} | val_loss: {val_loss_value:.4f} | "
                f"val_AUC: {val_metrics.get('auc', 0.0):.4f} | "
                f"val_MAE: {val_metrics.get('mae', 0.0):.4f} | "
                f"val_R2: {val_metrics.get('r2', 0.0):.4f} | "
                f"val_RankIC: {val_metrics.get('rank_ic', 0.0):.4f} | "
                f"val_P@5: {val_metrics.get('overall_precision@5', 0.0):.4f} | "
                f"gate={gate_val:.3f} | mdrop={phase_cfg.modality_dropout_p:.2f} | "
                f"lr_bb={optimizer.param_groups[0]['lr']:.1e} "
                f"lr_txt={optimizer.param_groups[1]['lr']:.1e}"
            )
            print(epoch_msg)

            if earlystop.step(current):
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    duration = time.time() - start_time
    result = {
        "val_auc": best_val,
        "test_auc": final_test.get("auc", 0.0),
        "test_ap": final_test.get("ap", 0.0),
        "test_mae": final_test.get("mae", 0.0),
        "test_rmse": final_test.get("rmse", 0.0),
        "test_r2": final_test.get("r2", 0.0),
        "epoch": epoch,
        "time": duration,
        "time_per_epoch": duration / (epoch + 1),
    }
    for k, v in final_test.items():
        rk = f"test_{k}" if not k.startswith("test_") else k
        if rk not in result:
            result[rk] = v
    return result
