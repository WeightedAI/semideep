"""
Weight computation module for distance-based semi-supervised learning.

This module implements the weight computation mechanisms described in:
"Enhancing Classification with Semi-Supervised Deep Learning Using Distance-Based Sample Weights"
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
from typing import Literal, Optional, Union, Callable


class WeightComputer:
    """Computes distance-based weights for training samples based on test data.
    
    This class implements the weight computation formula:
    w_i = (1/M) * Σ_j exp(-λ * d(x_i, x_j'))
    
    where:
    - x_i is a training sample
    - x_j' is a test sample
    - d is a distance metric
    - λ is a decay parameter that controls the influence of distance
    - M is the number of test samples
    """
    
    def __init__(
        self, 
        distance_metric: Union[str, Callable] = 'euclidean',
        lambda_: float = 0.8,
        device: str = 'cpu'
    ):
        """Initialize the weight computer.
        
        Args:
            distance_metric: Distance metric to use. Options are 'euclidean', 'cosine', 
                            'hamming', 'jaccard', or a callable that computes distances.
            lambda_: Decay parameter that controls how quickly weight decreases with distance.
                    Higher values make the weights more localized.
            device: Device to use for PyTorch tensors ('cpu' or 'cuda').
        """
        self.distance_metric = distance_metric
        self.lambda_ = lambda_
        self.device = device
        self.weights = None
        
    def _compute_distances(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """Compute distances between training and test samples.
        
        Args:
            X_train: Training data of shape (n_train_samples, n_features)
            X_test: Test data of shape (n_test_samples, n_features)
            
        Returns:
            Distance matrix of shape (n_train_samples, n_test_samples)
        """
        if callable(self.distance_metric):
            return self.distance_metric(X_train, X_test)
        
        if self.distance_metric == 'cosine':
            return cosine_distances(X_train, X_test)
        else:
            # Uses scipy's cdist for other metrics (euclidean, hamming, jaccard, etc.)
            return cdist(X_train, X_test, metric=self.distance_metric)
    
    def compute_weights(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """Compute weights for training samples based on test data.
        
        Args:
            X_train: Training data of shape (n_train_samples, n_features)
            X_test: Test data of shape (n_test_samples, n_features)
            
        Returns:
            Weights for each training sample, shape (n_train_samples,)
        """
        # Compute distances between all training and test samples
        distances = self._compute_distances(X_train, X_test)
        
        # Apply exponential decay to distances
        exp_distances = np.exp(-self.lambda_ * distances)
        
        # Average over all test samples (eq. 2 in the paper)
        weights = np.mean(exp_distances, axis=1)
        
        # Store the weights for later use
        self.weights = weights
        
        return weights
    
    def get_tensor_weights(self, indices=None) -> torch.Tensor:
        """Get weights as PyTorch tensor, optionally for specific indices.
        
        Args:
            indices: Optional indices to select specific weights
            
        Returns:
            Weights as a PyTorch tensor
        """
        if self.weights is None:
            raise ValueError("Weights have not been computed yet. Call compute_weights first.")
        
        if indices is not None:
            w = self.weights[indices]
        else:
            w = self.weights
            
        return torch.tensor(w, dtype=torch.float32, device=self.device)
    
    def save_weights(self, path: str) -> None:
        """Save weights to a file.
        
        Args:
            path: Path to save the weights
        """
        if self.weights is None:
            raise ValueError("Weights have not been computed yet. Call compute_weights first.")
        
        np.save(path, self.weights)
    
    def load_weights(self, path: str) -> np.ndarray:
        """Load weights from a file.
        
        Args:
            path: Path to load the weights from
            
        Returns:
            Loaded weights
        """
        self.weights = np.load(path)
        return self.weights
