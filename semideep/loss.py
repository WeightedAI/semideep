"""
Weighted loss functions for distance-based semi-supervised learning.

This module implements the weighted loss mechanism described in:
"Enhancing Classification with Semi-Supervised Deep Learning Using Distance-Based Sample Weights"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Union


class WeightedLoss(nn.Module):
    """Weighted loss function that applies sample weights to base loss.
    
    This implements the weighted loss formula:
    L = (1/N) * Σ_i w_i * Loss(y_i, f(x_i))
    
    where:
    - w_i is the weight for training sample i
    - Loss is a base loss function (e.g., CrossEntropyLoss)
    - y_i is the true label
    - f(x_i) is the model prediction
    """
    
    def __init__(
        self, 
        base_loss: Optional[Union[nn.Module, Callable]] = None,
        reduction: str = 'none'
    ):
        """Initialize the weighted loss.
        
        Args:
            base_loss: Base loss function to wrap. If None, uses CrossEntropyLoss.
            reduction: Reduction method for the base loss ('none', 'mean', or 'sum').
                       Should be 'none' for proper weighting.
        """
        super().__init__()
        
        if base_loss is None:
            self.base_loss = nn.CrossEntropyLoss(reduction='none')
        elif isinstance(base_loss, nn.Module):
            if hasattr(base_loss, 'reduction'):
                # If the loss has a reduction parameter, ensure it's set to 'none'
                base_loss.reduction = 'none'
            self.base_loss = base_loss
        else:
            # Custom callable loss function
            self.base_loss = base_loss
            
        self.reduction = reduction
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the weighted loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            weights: Optional sample weights. If None, equivalent to standard loss.
            
        Returns:
            Weighted loss value
        """
        # Compute base loss for each sample
        sample_losses = self.base_loss(predictions, targets)
        
        if weights is None:
            # If no weights provided, use standard reduction
            if self.reduction == 'mean':
                return torch.mean(sample_losses)
            elif self.reduction == 'sum':
                return torch.sum(sample_losses)
            return sample_losses
        
        # Apply weights to individual sample losses
        weighted_losses = sample_losses * weights
        
        # Apply reduction
        if self.reduction == 'sum':
            return torch.sum(weighted_losses)
        elif self.reduction == 'none':
            return weighted_losses
        else:  # Default to mean reduction as in paper equation (3)
            return torch.mean(weighted_losses)


class WeightedCrossEntropyLoss(WeightedLoss):
    """Convenience class for weighted cross entropy loss."""
    
    def __init__(self, reduction: str = 'mean'):
        """Initialize with cross entropy as the base loss.
        
        Args:
            reduction: Reduction method ('none', 'mean', or 'sum')
        """
        super().__init__(nn.CrossEntropyLoss(reduction='none'), reduction)


class WeightedBCEWithLogitsLoss(WeightedLoss):
    """Convenience class for weighted binary cross entropy with logits loss."""
    
    def __init__(self, reduction: str = 'mean', pos_weight: Optional[torch.Tensor] = None):
        """Initialize with BCE with logits as the base loss.
        
        Args:
            reduction: Reduction method ('none', 'mean', or 'sum')
            pos_weight: Optional weight for positive class
        """
        super().__init__(nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight), reduction)
