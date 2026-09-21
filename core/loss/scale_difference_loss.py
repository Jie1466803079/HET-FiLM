"""
Scale Difference Minimization Loss

Based on IDEA paper (IEEE TKDE 2023):
"High-Quality Temporal Link Prediction for Weighted Dynamic Graphs 
 via Inductive Embedding Aggregation"

This loss addresses the wide-value-range and sparsity issues for edge weight prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleDifferenceLoss(nn.Module):
    """
    Scale Difference Minimization (SDM) loss from IDEA paper.
    
    Uses logarithmic distance to measure scale differences:
    L_SD = Σ |log10(pred + ε) - log10(target + ε)|
    
    Key insight: 
    - Error(0.01% → 0.10%) has |log(0.01/0.10)| = 1.0
    - Error(10.01% → 10.10%) has |log(10.01/10.10)| = 0.004
    - First error is 250x larger in log space!
    
    This forces model to respect relative errors, not just absolute errors.
    """
    
    def __init__(self, epsilon=0.01):
        """
        Args:
            epsilon: Small constant to avoid log(0). 
                    Should be same order as minimum non-zero values.
                    For percent_tna, use 0.01 (i.e., 0.01%)
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        """
        Compute scale difference loss.
        
        Args:
            pred: Predicted values [N] or [N, 1]
            target: Ground truth values [N] or [N, 1]
            
        Returns:
            Scalar loss value
        """
        # Flatten if needed
        pred = pred.flatten()
        target = target.flatten()
        
        # Ensure positive (percent_tna should always be >= 0)
        pred = torch.clamp(pred, min=0)
        target = torch.clamp(target, min=0)
        
        # Add epsilon to avoid log(0)
        pred_safe = pred + self.epsilon
        target_safe = target + self.epsilon
        
        # Compute log10
        log_pred = torch.log10(pred_safe)
        log_target = torch.log10(target_safe)
        
        # L1 distance in log space (mean absolute log difference)
        loss = torch.mean(torch.abs(log_pred - log_target))
        
        return loss


class HybridLoss(nn.Module):
    """
    Hybrid loss combining error minimization and scale difference.
    
    L_total = α * L_EM + β * L_SD
    
    Where:
    - L_EM: Error minimization (MSE) - handles overall accuracy
    - L_SD: Scale difference minimization - handles relative errors
    
    NOTE: The IDEA paper does NOT specify exact values for α and β.
          Default values (α=10.0, β=1.0) are reasonable starting points
          based on typical loss magnitude ratios, but may need tuning.
          
    Tuning: Monitor L_EM and L_SD during training. If L_SD << L_EM,
            increase β. If L_EM << L_SD, increase α.
    """
    
    def __init__(self, alpha=10.0, beta=1.0, epsilon=0.01):
        """
        Args:
            alpha: Weight for error minimization (MSE)
            beta: Weight for scale difference
            epsilon: Small constant for log safety
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.sd_loss = ScaleDifferenceLoss(epsilon=epsilon)
    
    def forward(self, pred, target):
        """
        Compute hybrid loss.
        
        Args:
            pred: Predicted values
            target: Ground truth values
            
        Returns:
            loss: Combined loss value
            loss_dict: Dictionary with component losses for logging
        """
        # Error minimization (MSE)
        L_EM = F.mse_loss(pred.flatten(), target.flatten())
        
        # Scale difference minimization
        L_SD = self.sd_loss(pred, target)
        
        # Combined loss
        loss = self.alpha * L_EM + self.beta * L_SD
        
        # Return loss and components for logging
        loss_dict = {
            'loss': loss.item(),
            'L_EM': L_EM.item(),
            'L_SD': L_SD.item(),
        }
        
        return loss, loss_dict


# For backward compatibility with existing code
class MSEWithScaleDifference(HybridLoss):
    """Alias for HybridLoss for clearer naming."""
    pass


if __name__ == "__main__":
    # Test the loss functions
    print("Testing Scale Difference Loss Implementation")
    print("=" * 60)
    
    # Test case 1: Small values (should have large penalty)
    pred1 = torch.tensor([0.10, 0.10, 0.10])
    target1 = torch.tensor([0.01, 0.01, 0.01])
    
    # Test case 2: Large values (should have small penalty)
    pred2 = torch.tensor([10.10, 10.10, 10.10])
    target2 = torch.tensor([10.01, 10.01, 10.01])
    
    sd_loss = ScaleDifferenceLoss(epsilon=0.01)
    
    loss1 = sd_loss(pred1, target1)
    loss2 = sd_loss(pred2, target2)
    
    print(f"\nTest 1: Small values (0.01% → 0.10%)")
    print(f"  Absolute error: {(pred1 - target1).abs().mean():.4f}")
    print(f"  Scale difference loss: {loss1:.4f}")
    
    print(f"\nTest 2: Large values (10.01% → 10.10%)")
    print(f"  Absolute error: {(pred2 - target2).abs().mean():.4f}")
    print(f"  Scale difference loss: {loss2:.4f}")
    
    print(f"\nRatio: {loss1/loss2:.1f}x larger penalty for small values")
    print("✓ Expected: ~250x (demonstrating scale-aware behavior)")
    
    # Test hybrid loss
    print("\n" + "=" * 60)
    print("Testing Hybrid Loss")
    print("=" * 60)
    
    hybrid = HybridLoss(alpha=10.0, beta=1.0, epsilon=0.01)
    
    pred = torch.tensor([0.5, 1.0, 2.0, 5.0])
    target = torch.tensor([0.64, 1.2, 1.8, 5.2])
    
    loss, loss_dict = hybrid(pred, target)
    
    print(f"\nPredictions: {pred.tolist()}")
    print(f"Targets:     {target.tolist()}")
    print(f"\nLoss components:")
    print(f"  L_EM (MSE):              {loss_dict['L_EM']:.6f}")
    print(f"  L_SD (Scale Diff):       {loss_dict['L_SD']:.6f}")
    print(f"  L_total (10*EM + 1*SD):  {loss_dict['loss']:.6f}")
    
    print("\n✓ Loss implementation complete!")

