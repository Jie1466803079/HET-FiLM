import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_link_predictions(pred_scores, labels, k_list=[1, 3, 5, 10]):
    """
    Evaluate link prediction performance.
    Args:
        pred_scores: Predicted scores (torch.Tensor or np.array)
        labels: Ground truth labels (torch.Tensor or np.array)
        k_list: List of k values for Top-K metrics (placeholder for compatibility)
    Returns:
        Dictionary containing AUC and AP
    """
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Flatten if necessary
    pred_scores = pred_scores.reshape(-1)
    labels = labels.reshape(-1)

    # AUC and AP
    try:
        auc = roc_auc_score(labels, pred_scores)
        ap = average_precision_score(labels, pred_scores)
    except ValueError:
        # Handle edge cases (e.g. only one class present)
        auc = 0.5
        ap = 0.0

    metrics = {
        "auc": auc,
        "ap": ap
    }

    return metrics
