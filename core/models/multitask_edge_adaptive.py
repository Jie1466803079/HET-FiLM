"""
Multi-task Edge Predictor with Adaptive Distance Scaling (IDEA Paper)

Implements Equation 13 from the IDEA paper:
weight_pred = m_e * exp(-sigma_ij * ||h_i - h_j||^2)

where:
- m_e is the maximum edge weight (100.0 for percent_tna)
- sigma_ij is a learned scaling factor specific to the node pair (i, j)
- ||h_i - h_j||^2 is the squared Euclidean distance between node embeddings
"""

import torch
from torch import nn
from torch.nn import functional as F
from .multitask_edge import MultiTaskEdgePredictor

class MultiTaskEdgePredictorAdaptive(MultiTaskEdgePredictor):
    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: int,
        reg_hidden_dim: int = 128,
        regression_loss: str = "l2",
        class_loss_scale: float = 1.0,
        weight_loss_scale: float = 1.0,
        persist_loss_scale: float = 1.0,
        use_class_weights: bool = True,
        scale_loss_alpha: float = 10.0,
        scale_loss_beta: float = 1.0,
        scale_loss_epsilon: float = 0.01,
        max_edge_weight: float = 100.0,  # m_e parameter
    ):
        super().__init__(
            base_model=base_model,
            hidden_dim=hidden_dim,
            reg_hidden_dim=reg_hidden_dim,
            regression_loss=regression_loss,
            class_loss_scale=class_loss_scale,
            weight_loss_scale=weight_loss_scale,
            persist_loss_scale=persist_loss_scale,
            use_class_weights=use_class_weights,
            scale_loss_alpha=scale_loss_alpha,
            scale_loss_beta=scale_loss_beta,
            scale_loss_epsilon=scale_loss_epsilon,
        )
        
        self.max_edge_weight = max_edge_weight
        
        # Sigma MLP: Predicts the adaptive scaling factor sigma_ij from concatenated embeddings
        # Input: [h_i || h_j] (size 2 * hidden_dim)
        # Output: scalar sigma_ij > 0
        self.sigma_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, reg_hidden_dim),
            nn.ReLU(),
            nn.Linear(reg_hidden_dim, 1),
            nn.Softplus()  # Ensure sigma is positive
        )
        
        print(f"[Adaptive] Initialized with max_edge_weight={max_edge_weight}")

    def decode_weight(self, z, edge_label_index):
        """
        Predict weight using Adaptive Distance Scaling.
        w_hat = m_e * exp(-sigma * d^2)
        """
        # Get embeddings for source and destination nodes
        z_src, z_dst = self._get_pair_embeddings(z, edge_label_index)
        
        # 1. Calculate squared Euclidean distance: ||h_i - h_j||^2
        # (batch_size,)
        dist_sq = torch.sum((z_src - z_dst) ** 2, dim=-1)
        
        # 2. Predict sigma_ij using MLP on concatenated features
        # Input: (batch_size, 2*hidden) -> Output: (batch_size, 1)
        concat_features = torch.cat([z_src, z_dst], dim=-1)
        sigma = self.sigma_mlp(concat_features).view(-1)
        
        # 3. Apply formula: w = m_e * exp(-sigma * dist^2)
        # Result matches target range [0, 100]
        weight_pred = self.max_edge_weight * torch.exp(-sigma * dist_sq)
        
        return weight_pred

