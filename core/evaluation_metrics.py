import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score
from scipy import stats
import pandas as pd
from torch.nn import functional as F
import math


def dtw_distance(a, b):
    """Compute classic DTW distance between two 1D sequences (without negative sentinels)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float('nan')
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])
def _to_rank(x):
    import numpy as np
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    denom = max(len(x) - 1, 1)
    return ranks / denom


def calculate_comprehensive_metrics(predictions, targets, top_k_pct=0.2):
    """
    Calculate comprehensive evaluation metrics for fund prediction

    Args:
        predictions: torch.Tensor or np.array of predicted values
        targets: torch.Tensor or np.array of target values
        top_k_pct: float, percentage for top/bottom portfolio analysis (default 20%)

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Convert to numpy if torch tensors
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Remove any NaN values
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]

    if len(predictions) == 0:
        return get_empty_metrics()

    # Determine evaluation mode
    import os
    loss_mode = os.environ.get('LOSS_MODE', '').lower()
    target_mode = os.environ.get('TARGET_MODE', '').lower()
    use_rank_errors = (loss_mode == 'ic') or (target_mode == 'rank')

    # Basic regression metrics: rank-space if rank/ic mode, else value-space
    if use_rank_errors:
        pr = _to_rank(predictions)
        tr = _to_rank(targets)
        mae = mean_absolute_error(tr, pr)
        mse = mean_squared_error(tr, pr)
        rmse = np.sqrt(mse)
        r2 = float('nan')  # suppress misleading value-space R²
    else:
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        try:
            r2 = r2_score(targets, predictions)
        except:
            r2 = float('nan')

    # Portfolio analysis metrics
    portfolio_metrics = calculate_portfolio_metrics(predictions, targets, top_k_pct)

    # Information Coefficient (IC)
    try:
        ic, ic_pvalue = stats.pearsonr(predictions, targets)
    except:
        ic, ic_pvalue = float('nan'), float('nan')

    # Rank IC (Spearman correlation)
    try:
        rank_ic, rank_ic_pvalue = stats.spearmanr(predictions, targets)
    except:
        rank_ic, rank_ic_pvalue = float('nan'), float('nan')

    # Hit rate (percentage of correct directional predictions)
    pred_direction = np.sign(predictions)
    target_direction = np.sign(targets)
    hit_rate = np.mean(pred_direction == target_direction)

    metrics = {
        # Basic regression metrics (rank-space under rank/ic mode)
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,

        # Information metrics
        'IC': ic,
        'IC_pvalue': ic_pvalue,
        'Rank_IC': rank_ic,
        'Rank_IC_pvalue': rank_ic_pvalue,
        'Hit_Rate': hit_rate,

        # Portfolio metrics
        **portfolio_metrics
    }

    return metrics


def _nw_tstat(x: np.ndarray, lags: int = None):
    """Newey-West HAC t-stat of the mean for a 1D series x.
    Returns (t_stat, p_value, se).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T == 0:
        return float('nan'), float('nan'), float('nan')
    mu = np.mean(x)
    # automatic lag: Bartlett kernel with lags ~ 1.5 * T^{1/3}
    if lags is None:
        lags = max(1, int(1.5 * (T ** (1.0 / 3.0))))
    # sample autocovariances
    eps = x - mu
    gamma0 = np.dot(eps, eps) / T
    var = gamma0
    for L in range(1, lags + 1):
        w = 1.0 - (L / (lags + 1.0))  # Bartlett weight
        cov = np.dot(eps[L:], eps[:-L]) / T
        var += 2.0 * w * cov
    se = math.sqrt(max(var, 0.0) / T)
    t = mu / se if se > 0 else float('inf')
    # two-sided normal p-value
    try:
        p = 2.0 * (1.0 - stats.norm.cdf(abs(t)))
    except Exception:
        p = float('nan')
    return t, p, se


def evaluate_link_predictions(scores, labels, k_list=(5, 10, 20)):
    """
    Evaluate binary link predictions with AUC/AP and simple top-k metrics.
    Args:
        scores: 1D tensor/array of model scores (logits or probabilities)
        labels: 1D tensor/array of 0/1 ground truth
        k_list: iterable of cutoff values for precision/recall/NDCG
    Returns:
        dict with AUC, AP, Precision@k, Recall@k, NDCG@k
    """
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()

    # Filter finite
    mask = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[mask]
    labels = labels[mask]

    metrics = {}
    try:
        metrics["AUC"] = roc_auc_score(labels, scores)
    except Exception:
        metrics["AUC"] = float('nan')
    try:
        metrics["AP"] = average_precision_score(labels, scores)
    except Exception:
        metrics["AP"] = float('nan')

    # Top-k metrics
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    pos_total = float(sorted_labels.sum())
    for k in k_list:
        k = int(k)
        if k <= 0:
            continue
        topk = sorted_labels[: min(k, len(sorted_labels))]
        hits = float(topk.sum())
        metrics[f"Precision@{k}"] = hits / len(topk) if len(topk) > 0 else float('nan')
        metrics[f"Recall@{k}"] = hits / pos_total if pos_total > 0 else float('nan')
        # NDCG@k for binary labels
        denom = np.log2(np.arange(2, len(topk) + 2))
        dcg = (topk / denom).sum()
        ideal = np.sort(sorted_labels)[::-1][: len(topk)]
        idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
        metrics[f"NDCG@{k}"] = dcg / idcg if idcg > 0 else float('nan')

    return metrics


def calculate_time_series_metrics(preds_list, targets_list, top_k_pct=0.2, periods_per_year: int = 12):
    """Compute time-series aggregated metrics from per-period cross-sectional vectors.
    - preds_list/targets_list: list of 1D arrays per period (same length).
    - Returns dict with annualized Sharpe (time-series), NW t-stats for IC/Rank-IC/spread.
    """
    assert len(preds_list) == len(targets_list)
    T = len(preds_list)
    if T == 0:
        return {}

    ic_ts, ric_ts, spread_ts = [], [], []
    for p, t in zip(preds_list, targets_list):
        p = np.asarray(p).flatten()
        t = np.asarray(t).flatten()
        m = ~(np.isnan(p) | np.isnan(t))
        p = p[m]
        t = t[m]
        if p.size < 3:
            ic_ts.append(np.nan)
            ric_ts.append(np.nan)
            spread_ts.append(np.nan)
            continue
        # Pearson (IC) and Spearman (Rank-IC)
        try:
            ic_ts.append(stats.pearsonr(p, t)[0])
        except Exception:
            ic_ts.append(np.nan)
        try:
            ric_ts.append(stats.spearmanr(p, t)[0])
        except Exception:
            ric_ts.append(np.nan)
        # Per-period long-short spread using top_k_pct
        pm = calculate_portfolio_metrics(p, t, top_k_pct)
        spread_ts.append(pm.get('Spread_Mean', np.nan))

    ic_ts = np.asarray(ic_ts, dtype=float)
    ric_ts = np.asarray(ric_ts, dtype=float)
    spread_ts = np.asarray(spread_ts, dtype=float)

    # Means
    ic_mean = np.nanmean(ic_ts) if np.isfinite(ic_ts).any() else float('nan')
    ric_mean = np.nanmean(ric_ts) if np.isfinite(ric_ts).any() else float('nan')
    # IC^2 alignment proxy (avg of squared Pearson IC over time)
    ic_r2_mean = np.nanmean(ic_ts ** 2) if np.isfinite(ic_ts).any() else float('nan')

    # Newey-West t-stats
    ic_t, ic_p, ic_se = _nw_tstat(ic_ts)
    ric_t, ric_p, ric_se = _nw_tstat(ric_ts)
    sp_t, sp_p, sp_se = _nw_tstat(spread_ts)

    # Time-series Sharpe (per-period), annualized by sqrt(PY)
    # Standard Sharpe uses mean/std; NW t-stat is reported separately
    if np.isfinite(spread_ts).any():
        sp_mean = np.nanmean(spread_ts)
        sp_std = np.nanstd(spread_ts, ddof=1)
        sharpe_ts = (sp_mean / sp_std) * math.sqrt(periods_per_year) if sp_std > 0 else float('inf')
    else:
        sharpe_ts = float('nan')

    return {
        'IC_TS_mean': ic_mean,
        'IC_TS_tstat_NW': ic_t,
        'IC_TS_pvalue_NW': ic_p,
        'IC_R2_TS_mean': ic_r2_mean,
        'Rank_IC_TS_mean': ric_mean,
        'Rank_IC_TS_tstat_NW': ric_t,
        'Rank_IC_TS_pvalue_NW': ric_p,
        'Spread_TS_mean': float(np.nanmean(spread_ts)) if np.isfinite(spread_ts).any() else float('nan'),
        'Spread_TS_tstat_NW': sp_t,
        'Spread_TS_pvalue_NW': sp_p,
        'Spread_Sharpe_TS_annual': sharpe_ts,
    }

def calculate_portfolio_metrics(predictions, targets, top_k_pct=0.2):
    """
    Calculate portfolio-based evaluation metrics

    Args:
        predictions: np.array of predicted values
        targets: np.array of target values
        top_k_pct: float, percentage for quintile analysis

    Returns:
        dict: Portfolio evaluation metrics
    """
    n_samples = len(predictions)
    if n_samples < 5:  # Need minimum samples for meaningful analysis
        return get_empty_portfolio_metrics()

    k = max(1, int(n_samples * top_k_pct))

    # Sort indices by predictions (descending)
    sorted_indices = np.argsort(predictions)[::-1]

    # Q1 (top quintile) and Q5 (bottom quintile) based on predictions
    q1_indices = sorted_indices[:k]  # Top k predicted
    q5_indices = sorted_indices[-k:] # Bottom k predicted

    # Calculate actual returns for each quintile
    q1_returns = targets[q1_indices]
    q5_returns = targets[q5_indices]

    q1_mean = np.mean(q1_returns)
    q5_mean = np.mean(q5_returns)

    # Long-short spread (Q1 - Q5)
    spread_mean = q1_mean - q5_mean

    # Calculate spread statistics if we have enough data
    if len(q1_returns) > 1 and len(q5_returns) > 1:
        # Standard error of the spread
        spread_std = np.sqrt(np.var(q1_returns)/len(q1_returns) + np.var(q5_returns)/len(q5_returns))
        spread_sharpe = spread_mean / spread_std if spread_std > 0 else 0

        # T-statistic for spread significance
        spread_tstat = spread_mean / spread_std if spread_std > 0 else 0
        spread_pvalue = 2 * (1 - stats.norm.cdf(abs(spread_tstat))) if spread_std > 0 else 1
    else:
        spread_sharpe = float('nan')
        spread_tstat = float('nan')
        spread_pvalue = float('nan')

    # Additional quintile analysis
    quintile_metrics = calculate_quintile_metrics(predictions, targets)

    portfolio_metrics = {
        'Q1_mean': q1_mean,
        'Q5_mean': q5_mean,
        'Spread_Mean': spread_mean,
        'Spread_Sharpe': spread_sharpe,
        'Spread_tstat': spread_tstat,
        'Spread_pvalue': spread_pvalue,
        **quintile_metrics
    }

    return portfolio_metrics

def calculate_quintile_metrics(predictions, targets, n_quintiles=5):
    """
    Calculate metrics for all quintiles
    """
    n_samples = len(predictions)
    quintile_size = max(1, n_samples // n_quintiles)

    # Sort indices by predictions (descending)
    sorted_indices = np.argsort(predictions)[::-1]

    quintile_metrics = {}

    for q in range(n_quintiles):
        start_idx = q * quintile_size
        end_idx = min((q + 1) * quintile_size, n_samples) if q < n_quintiles - 1 else n_samples

        if start_idx >= n_samples:
            break

        quintile_indices = sorted_indices[start_idx:end_idx]
        quintile_returns = targets[quintile_indices]

        if len(quintile_returns) > 0:
            quintile_metrics[f'Q{q+1}_mean'] = np.mean(quintile_returns)
            quintile_metrics[f'Q{q+1}_std'] = np.std(quintile_returns)
            quintile_metrics[f'Q{q+1}_count'] = len(quintile_returns)

    return quintile_metrics

def get_empty_metrics():
    """Return metrics dict with NaN values for empty data"""
    return {
        'MAE': float('nan'),
        'MSE': float('nan'),
        'RMSE': float('nan'),
        'R2': float('nan'),
        'IC': float('nan'),
        'IC_pvalue': float('nan'),
        'Rank_IC': float('nan'),
        'Rank_IC_pvalue': float('nan'),
        'Hit_Rate': float('nan'),
        **get_empty_portfolio_metrics()
    }

def get_empty_portfolio_metrics():
    """Return portfolio metrics dict with NaN values"""
    metrics = {
        'Q1_mean': float('nan'),
        'Q5_mean': float('nan'),
        'Spread_Mean': float('nan'),
        'Spread_Sharpe': float('nan'),
        'Spread_tstat': float('nan'),
        'Spread_pvalue': float('nan')
    }

    for q in range(1, 6):
        metrics[f'Q{q}_mean'] = float('nan')
        metrics[f'Q{q}_std'] = float('nan')
        metrics[f'Q{q}_count'] = 0

    return metrics

def evaluate_predictions(predictions, targets, model_name="Model"):
    """
    Main evaluation function that can be called by any model

    Args:
        predictions: model predictions (torch.Tensor or np.array)
        targets: true targets (torch.Tensor or np.array)
        model_name: string name for the model

    Returns:
        dict: comprehensive evaluation metrics
    """
    metrics = calculate_comprehensive_metrics(predictions, targets)
    metrics['model_name'] = model_name

    return metrics

def print_evaluation_results(metrics, title="Evaluation Results"):
    """
    Pretty print evaluation results
    """
    print(f"\n{title}")
    print("=" * 60)

    # Basic metrics
    print("Basic Regression Metrics:")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  R²:   {metrics['R2']:.6f}")

    # Information metrics
    print("\nInformation Metrics:")
    print(f"  IC:       {metrics['IC']:.6f} (p-value: {metrics['IC_pvalue']:.6f})")
    print(f"  Rank IC:  {metrics['Rank_IC']:.6f} (p-value: {metrics['Rank_IC_pvalue']:.6f})")
    print(f"  Hit Rate: {metrics['Hit_Rate']:.6f}")

    # Portfolio metrics
    print("\nPortfolio Analysis:")
    print(f"  Q1 Mean Return:    {metrics['Q1_mean']:.6f}")
    print(f"  Q5 Mean Return:    {metrics['Q5_mean']:.6f}")
    print(f"  Long-Short Spread: {metrics['Spread_Mean']:.6f}")
    print(f"  Spread Sharpe:     {metrics['Spread_Sharpe']:.6f}")
    print(f"  Spread t-stat:     {metrics['Spread_tstat']:.6f}")

    # Quintile breakdown
    print("\nQuintile Breakdown:")
    for q in range(1, 6):
        if f'Q{q}_mean' in metrics and not np.isnan(metrics[f'Q{q}_mean']):
            std_val = metrics.get(f'Q{q}_std', float('nan'))
            count_val = metrics.get(f'Q{q}_count', 0)
            print(f"  Q{q}: {metrics[f'Q{q}_mean']:.6f} (std: {std_val:.6f}, n: {count_val})")

def save_metrics_to_csv(metrics, filepath, append=True):
    """
    Save metrics to CSV file

    Args:
        metrics: dict of evaluation metrics
        filepath: path to save CSV file
        append: whether to append to existing file
    """
    df = pd.DataFrame([metrics])

    if append:
        try:
            existing_df = pd.read_csv(filepath)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass

    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")

# Example usage functions for different models

def evaluate_dhgas_model(model, dataset, device="cpu"):
    """Evaluate DHGAS model"""
    model.eval()

    # Check if model exposes a node-level decoder (supports regression/classification)
    has_decoder = hasattr(model, 'decode_nclf') and callable(getattr(model, 'decode_nclf'))
    has_head = hasattr(model, 'nclf_linear') and model.nclf_linear is not None

    if not (has_decoder or has_head):
        print("Model is configured for link prediction. Skipping comprehensive evaluation.")
        return get_empty_metrics()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        # Handle different dataset structures
        if hasattr(dataset, 'test_dataset'):
            test_data = dataset.test_dataset
        else:
            test_data = dataset

        # Check if test_data is a list or single item
        if isinstance(test_data, list):
            preds_per_period = []
            targets_per_period = []
            for data in test_data:
                if isinstance(data, tuple) and len(data) == 2:
                    support, query = data
                else:
                    # Handle single data structure
                    support = data
                    query = data

                # If model provides direct tensor predictor (e.g., XGBoost), use query.x
                if hasattr(model, 'predict_tensor') and hasattr(query, 'x'):
                    out = model.predict_tensor(query.x)
                else:
                    g_support = support[0] if isinstance(support, (list, tuple)) else support
                    z = model.encode(g_support)
                    out = model.decode_nclf(z)

                # Get targets from query and align scales (apply inverse log-shift if present)
                offset = getattr(query, 'target_offset', None)
                if hasattr(query, 'y_raw_target') and offset is not None:
                    targets = query.y_raw_target
                    preds = torch.exp(out.squeeze()) - float(offset)
                else:
                    targets = query.y if hasattr(query, 'y') else (query.fund.y if hasattr(query, 'fund') and hasattr(query.fund, 'y') else None)
                    preds = out.squeeze()
                if targets is None:
                    print("Warning: No targets found in query data")
                    continue
                # Filter finite pairs; handle classification logits (2D) vs targets (1D)
                if preds.dim() > 1:
                    # Per-sample finiteness across classes
                    m_pred = torch.isfinite(preds).all(dim=-1)
                    m_targ = torch.isfinite(targets)
                    mask = m_pred & m_targ
                    # Reduce logits to predicted class index for a 1D summary
                    preds_scalar = preds.argmax(dim=-1).to(dtype=torch.float32)
                else:
                    mask = torch.isfinite(preds) & torch.isfinite(targets)
                    preds_scalar = preds.to(dtype=torch.float32)
                if mask.sum() == 0:
                    continue
                p_use = preds_scalar[mask].detach().cpu()
                t_use = targets[mask].detach().cpu()
                all_predictions.append(p_use)
                all_targets.append(t_use)
                preds_per_period.append(p_use.numpy())
                targets_per_period.append(t_use.numpy())
        elif isinstance(test_data, tuple) and any(isinstance(x, (list, tuple)) for x in test_data):
            preds_per_period = []
            targets_per_period = []
            # If second element is a list/tuple, treat it as per-period queries
            q_part = test_data[1] if len(test_data) > 1 else []
            if isinstance(q_part, (list, tuple)):
                iterable = q_part
            else:
                iterable = test_data
            for query in iterable:
                support = query
                if hasattr(model, 'predict_tensor') and hasattr(query, 'x'):
                    out = model.predict_tensor(query.x)
                else:
                    g_support = support[0] if isinstance(support, (list, tuple)) else support
                    z = model.encode(g_support)
                    out = model.decode_nclf(z)
                offset = getattr(query, 'target_offset', None)
                if hasattr(query, 'y_raw_target') and offset is not None:
                    targets = query.y_raw_target
                    preds = torch.exp(out.squeeze()) - float(offset)
                else:
                    targets = query.y if hasattr(query, 'y') else (query.fund.y if hasattr(query, 'fund') and hasattr(query.fund, 'y') else None)
                    preds = out.squeeze()
                if targets is None:
                    continue
                if preds.dim() > 1:
                    m_pred = torch.isfinite(preds).all(dim=-1)
                    m_targ = torch.isfinite(targets)
                    mask = m_pred & m_targ
                    preds_scalar = preds.argmax(dim=-1).to(dtype=torch.float32)
                else:
                    mask = torch.isfinite(preds) & torch.isfinite(targets)
                    preds_scalar = preds.to(dtype=torch.float32)
                if mask.sum() == 0:
                    continue
                p_use = preds_scalar[mask].detach().cpu()
                t_use = targets[mask].detach().cpu()
                all_predictions.append(p_use)
                all_targets.append(t_use)
                preds_per_period.append(p_use.numpy())
                targets_per_period.append(t_use.numpy())
        else:
            # Handle single data item
            if isinstance(test_data, tuple) and len(test_data) == 2:
                support, query = test_data
            else:
                support = test_data
                query = test_data

            if hasattr(model, 'predict_tensor') and hasattr(query, 'x'):
                out = model.predict_tensor(query.x)
            else:
                g_support = support[0] if isinstance(support, (list, tuple)) else support
                z = model.encode(g_support)
                out = model.decode_nclf(z)

            offset = getattr(query, 'target_offset', None)
            if hasattr(query, 'y_raw_target') and offset is not None:
                targets = query.y_raw_target
                preds = torch.exp(out.squeeze()) - float(offset)
            else:
                if hasattr(query, 'y'):
                    targets = query.y
                elif hasattr(query, 'fund') and hasattr(query.fund, 'y'):
                    targets = query.fund.y
                else:
                    print("Warning: No targets found in query data")
                    return get_empty_metrics()
                preds = out.squeeze()

            # Align to labeled nodes if present_mask exists
            if hasattr(query, "present_mask"):
                pm = query.present_mask.bool()
                if pm.numel() == preds.shape[0] and preds.shape[0] != targets.shape[0]:
                    preds = preds[pm]
                elif pm.numel() == preds.shape[0] == targets.shape[0]:
                    preds = preds[pm]
                    targets = targets[pm]
                elif pm.numel() == targets.shape[0] and preds.shape[0] != targets.shape[0]:
                    targets = targets[pm]

            if preds.dim() > 1:
                m_pred = torch.isfinite(preds).all(dim=-1)
                m_targ = torch.isfinite(targets)
                mask = m_pred & m_targ
                preds_scalar = preds.argmax(dim=-1).to(dtype=torch.float32)
            else:
                mask = torch.isfinite(preds) & torch.isfinite(targets)
                preds_scalar = preds.to(dtype=torch.float32)
            if mask.sum() == 0:
                return get_empty_metrics()
            p_use = preds_scalar[mask].detach().cpu()
            t_use = targets[mask].detach().cpu()
            all_predictions.append(p_use)
            all_targets.append(t_use)

    if len(all_predictions) == 0:
        print("Warning: No predictions collected")
        return get_empty_metrics()

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Base pooled metrics
    metrics = evaluate_predictions(predictions, targets, "DHGAS")

    # If we collected per-period series, compute time-series metrics and prefer them for Sharpe/p-values
    try:
        periods_per_year = int(os.environ.get('PERIODS_PER_YEAR', '12'))
    except Exception:
        periods_per_year = 12
    if 'preds_per_period' in locals() and len(preds_per_period) > 0:
        ts = calculate_time_series_metrics(preds_per_period, targets_per_period, periods_per_year=periods_per_year)
        # Replace pooled Sharpe/t-stats with time-series versions for realism
        if ts:
            metrics['Spread_Sharpe'] = ts.get('Spread_Sharpe_TS_annual', metrics.get('Spread_Sharpe'))
            metrics['Spread_tstat'] = ts.get('Spread_TS_tstat_NW', metrics.get('Spread_tstat'))
            metrics['Rank_IC'] = ts.get('Rank_IC_TS_mean', metrics.get('Rank_IC'))
            metrics['Rank_IC_pvalue'] = ts.get('Rank_IC_TS_pvalue_NW', metrics.get('Rank_IC_pvalue'))
            # Provide additional aligned diagnostics
            metrics['IC_TS_mean'] = ts.get('IC_TS_mean')
            metrics['IC_R2_TS_mean'] = ts.get('IC_R2_TS_mean')
            metrics['Rank_IC_TS_tstat_NW'] = ts.get('Rank_IC_TS_tstat_NW')
            metrics['Spread_Sharpe_TS_annual'] = ts.get('Spread_Sharpe_TS_annual')
            # Dynamic Time Warping on per-period means as a rough alignment score
            seq_pred = np.array([np.nanmean(p) for p in preds_per_period], dtype=float)
            seq_true = np.array([np.nanmean(t) for t in targets_per_period], dtype=float)
            mask = ~(np.isnan(seq_pred) | np.isnan(seq_true))
            seq_pred = seq_pred[mask]
            seq_true = seq_true[mask]
            if seq_pred.size > 0 and seq_true.size > 0:
                dtw = dtw_distance(seq_pred, seq_true)
                metrics['DTW_mean'] = dtw
                metrics['DTW_mean_per_step'] = dtw / max(len(seq_pred), len(seq_true))
            else:
                metrics['DTW_mean'] = float('nan')
                metrics['DTW_mean_per_step'] = float('nan')
        # Expose sequences for downstream plotting (do not write to CSV)
        metrics['preds_per_period'] = [np.asarray(p, dtype=float) for p in preds_per_period]
        metrics['targets_per_period'] = [np.asarray(t, dtype=float) for t in targets_per_period]
    else:
        # Fallback: DTW on the pooled prediction/target sequence
        p_np = predictions.detach().cpu().numpy().flatten()
        t_np = targets.detach().cpu().numpy().flatten()
        mask = ~(np.isnan(p_np) | np.isnan(t_np))
        p_np = p_np[mask]
        t_np = t_np[mask]
        if p_np.size > 0 and t_np.size > 0:
            dtw = dtw_distance(p_np, t_np)
            metrics['DTW_mean'] = dtw
            metrics['DTW_mean_per_step'] = dtw / max(len(p_np), len(t_np))
        else:
            metrics['DTW_mean'] = float('nan')
            metrics['DTW_mean_per_step'] = float('nan')
    return metrics

def evaluate_generic_model(predictions, targets, model_name):
    """Generic evaluation for any model given predictions and targets"""
    return evaluate_predictions(predictions, targets, model_name)

if __name__ == "__main__":
    # Example usage
    print("Evaluation Metrics Module")
    print("Usage: import evaluation_metrics and call evaluate_predictions()")
