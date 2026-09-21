"""
Weight-Aware Supervised Contrastive Loss for Stage 1.

Forces GNN embeddings to encode holding-weight magnitude: edges with similar
weights cluster together, edges with different weights separate. This breaks
the information bottleneck where Stage 1 BCE treats all positive edges equally.

Uses quantile-based weight bins as pseudo-labels for SupCon (Khosla et al. 2020).
"""

import torch
from torch import nn
from torch.nn import functional as F


class WeightAwareContrastiveLoss(nn.Module):
    """Supervised contrastive loss with weight-quantile bins.

    For positive edges in a batch:
    1. Subsample to max_samples for tractability
    2. Compute Hadamard edge embeddings, L2-normalize
    3. Bin edges by weight quantile
    4. Apply SupCon: same-bin edges are positives, cross-bin are negatives
    """

    def __init__(self, n_bins: int = 4, temperature: float = 0.1,
                 max_samples: int = 512):
        super().__init__()
        self.n_bins = n_bins
        self.temperature = temperature
        self.max_samples = max_samples
        self._logged_first = False

    def forward(self, z, edge_label_index, edge_label, edge_weight):
        """
        Args:
            z: node embeddings — dict {node_type: Tensor} or tuple (z_src_all, z_dst_all).
            edge_label_index: (2, E) fund→stock indices.
            edge_label: (E,) binary labels.
            edge_weight: (E,) holding weights.

        Returns:
            Scalar loss (0.0 when insufficient data).
        """
        device = edge_label.device

        pos_mask = (edge_label > 0) & (edge_weight > 0)
        n_pos = int(pos_mask.sum().item())
        if n_pos < self.n_bins * 2:
            return edge_label.new_tensor(0.0)

        pos_idx = pos_mask.nonzero(as_tuple=True)[0]

        if n_pos > self.max_samples:
            perm = torch.randperm(n_pos, device=device)[:self.max_samples]
            pos_idx = pos_idx[perm]
            n_pos = self.max_samples

        fund_ids = edge_label_index[0, pos_idx]
        stock_ids = edge_label_index[1, pos_idx]
        weights = edge_weight[pos_idx]

        # Extract node embeddings
        if isinstance(z, dict):
            node_types = list(z.keys())
            z_fund = z[node_types[0]][fund_ids]
            z_stock = z[node_types[1]][stock_ids]
        elif isinstance(z, (list, tuple)):
            z_fund = z[0][fund_ids]
            z_stock = z[1][stock_ids]
        else:
            z_fund = z[fund_ids]
            z_stock = z[stock_ids]

        edge_emb = z_fund * z_stock  # Hadamard product
        edge_emb = F.normalize(edge_emb, dim=-1)

        # Assign weight-quantile bins
        bin_labels = self._quantile_bin(weights)

        # Need at least 2 distinct bins with >= 2 members each
        unique_bins, counts = bin_labels.unique(return_counts=True)
        multi_bins = (counts >= 2).sum().item()
        if multi_bins < 2:
            return edge_label.new_tensor(0.0)

        if not self._logged_first:
            bin_dist = {int(b): int(c) for b, c in zip(unique_bins.tolist(), counts.tolist())}
            print(f"[WEIGHT_CONTRASTIVE] First call: n_pos={n_pos}, "
                  f"bins={bin_dist}, temp={self.temperature}", flush=True)
            self._logged_first = True

        loss = self._supcon(edge_emb, bin_labels)
        return loss

    def _quantile_bin(self, weights: torch.Tensor) -> torch.Tensor:
        """Assign each weight to a quantile bin (0 .. n_bins-1)."""
        n = weights.shape[0]
        quantiles = torch.linspace(0.0, 1.0, self.n_bins + 1, device=weights.device)[1:-1]
        thresholds = torch.quantile(weights.float(), quantiles)

        bins = torch.zeros(n, dtype=torch.long, device=weights.device)
        for i, t in enumerate(thresholds):
            bins[weights > t] = i + 1
        return bins

    def _supcon(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Supervised contrastive loss (Khosla et al. 2020).

        Args:
            features: (N, D) L2-normalized embeddings.
            labels: (N,) integer bin labels.
        """
        N = features.shape[0]

        sim = torch.mm(features, features.T) / self.temperature  # (N, N)

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Masks
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        self_mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
        pos_mask = label_eq & self_mask  # same bin, not self
        neg_mask = ~label_eq & self_mask  # different bin

        # Denominator: sum over all non-self entries
        exp_sim = torch.exp(sim) * self_mask.float()
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8))

        # Log-prob for positive pairs
        log_prob = sim - log_denom  # (N, N)

        # Mean of log-prob over positive pairs for each anchor
        pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)
        mean_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / pos_count

        # Only include anchors that have at least one positive pair
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return features.new_tensor(0.0)

        loss = -mean_log_prob[has_pos].mean()
        return loss
