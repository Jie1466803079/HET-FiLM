"""Shared helpers for raw-EW message scaling in baseline models.

Activated by EDGE_WEIGHT_MESSAGE=1 (same toggle CMLN uses). Weights are the
raw support-window edge_attr values (no transform); edge types without
edge_attr contribute 1.0 so unweighted relations pass messages unchanged.
"""
import os

import torch


def use_ew_message():
    return os.environ.get("EDGE_WEIGHT_MESSAGE", "0") != "0"


def _etype_weight(store, num_edges, device):
    ew = getattr(store, "edge_attr", None)
    if ew is not None and ew.numel() == num_edges:
        return ew.float().view(-1).to(device)
    return torch.ones(num_edges, device=device)


def build_edge_weight_vector(data, e_dict):
    """Per-edge weights concatenated in e_dict iteration order.

    Matches the edge ordering produced by HeteroData.to_homogeneous() after
    make_hodata assigns edge_index per type in the same iteration order.
    """
    ws = []
    for etype, e in e_dict.items():
        ws.append(_etype_weight(data[etype], e.size(1), e.device))
    return torch.cat(ws) if ws else None


def build_edge_weight_dict(data, e_dict):
    """Per-edge-type weight tensors (1.0 fallback), for hetero convs (HAN)."""
    return {
        etype: _etype_weight(data[etype], e.size(1), e.device)
        for etype, e in e_dict.items()
    }


def snapshot_edge_attr(graph, etype):
    """Aligned edge_attr for one snapshot/etype (1.0 fallback) for time-merge."""
    store = graph[etype]
    return _etype_weight(store, store.edge_index.size(1), store.edge_index.device)
