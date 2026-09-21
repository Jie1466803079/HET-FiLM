import torch
from evaluation_metrics import evaluate_link_predictions


def evaluate_link_dhgas(
    model,
    graph,
    edge_label_index=None,
    edge_label=None,
    pos_edges=None,
    neg_edges=None,
    device="cpu",
):
    """
    Evaluate link prediction (AUC/AP/top-K) given a graph and pos/neg edges
    or edge_label_index/edge_label (as produced by hetero_linksplit).
    """
    model.eval()
    with torch.no_grad():
        if edge_label_index is not None and edge_label is not None:
            edge_label_index = edge_label_index.to(device)
            edge_label = edge_label.to(device)
            pos_mask = edge_label == 1
            neg_mask = edge_label == 0
            pos_edges = edge_label_index[:, pos_mask]
            neg_edges = edge_label_index[:, neg_mask]
        if pos_edges is None or neg_edges is None:
            raise ValueError("pos_edges/neg_edges or edge_label_index/edge_label required")

        g = graph.to(device)
        z = model.encode(g)
        pos_score = model.decode(z, pos_edges)
        neg_score = model.decode(z, neg_edges)
        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(pos_score, dtype=torch.float32),
                torch.zeros_like(neg_score, dtype=torch.float32),
            ],
            dim=0,
        )
        metrics = evaluate_link_predictions(scores, labels)
        return metrics
