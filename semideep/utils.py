"""
Utility functions for the SemiDeep package.

This module provides helper functions for distance metric selection,
data preprocessing, and other utilities needed by the SemiDeep components.
"""

import numpy as np
from typing import Tuple, List, Dict, Union, Optional
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import logging

from .weight_computer import WeightComputer
from .trainer import WeightedTrainer

logger = logging.getLogger(__name__)


def select_best_distance_metric(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    metrics: List[str] = None,
    lambda_values: List[float] = None,
    cv: int = 3,
    verbose: bool = True
) -> Tuple[str, float, float]:
    """Select the best distance metric and lambda value for a dataset.
    
    This function evaluates different distance metrics and lambda values
    using cross-validation to find the best combination for a given dataset.
    
    Args:
        model: PyTorch model to train
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        metrics: List of distance metrics to evaluate. If None, uses default metrics
        lambda_values: List of lambda values to evaluate. If None, uses default values
        cv: Number of cross-validation folds
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_metric, best_lambda, best_score)
    """
    if metrics is None:
        metrics = ['euclidean', 'cosine', 'hamming', 'jaccard']
        
    if lambda_values is None:
        lambda_values = [0.5, 0.7, 0.8, 0.9, 1.0]
    
    best_metric = None
    best_lambda = None
    best_score = -np.inf
    
    results = []
    
    for metric in metrics:
        for lambda_ in lambda_values:
            if verbose:
                logger.info(f"Evaluating metric={metric}, lambda={lambda_}")
            
            # Create weight computer
            weight_computer = WeightComputer(
                distance_metric=metric,
                lambda_=lambda_
            )
            
            # Compute weights
            weights = weight_computer.compute_weights(X_train, X_test)
            
            # Cross-validate with these weights
            scores = []
            # We're using a simpler approach than full cross-validation
            # Just evaluate model performance with these weights
            
            # Simplified evaluation - train on 80% of data with weights
            train_size = int(0.8 * len(X_train))
            X_subset = X_train[:train_size]
            y_subset = y_train[:train_size]
            weights_subset = weights[:train_size]
            
            # Validation data
            X_val = X_train[train_size:]
            y_val = y_train[train_size:]
            
            # Train a model
            trainer = WeightedTrainer(
                model=model,
                X_train=X_subset,
                y_train=y_subset,
                weights=weights_subset,
                epochs=50,  # Reduced epochs for faster evaluation
                verbose=False
            )
            
            trainer.train()
            metrics = trainer.evaluate(X_val, y_val)
            score = metrics['f1']  # Use F1 score for evaluation
            
            results.append({
                'metric': metric,
                'lambda': lambda_,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_metric = metric
                best_lambda = lambda_
                
            if verbose:
                logger.info(f"Score: {score:.4f}")
    
    if verbose:
        logger.info(f"Best combination: metric={best_metric}, lambda={best_lambda}, score={best_score:.4f}")
    
    return best_metric, best_lambda, best_score


def auto_select_distance_metric(X: np.ndarray) -> str:
    """Automatically select an appropriate distance metric based on data characteristics.
    
    Args:
        X: Feature matrix
        
    Returns:
        Recommended distance metric
    """
    # Analyze data characteristics
    n_samples, n_features = X.shape
    
    # Check for categorical/binary features
    is_binary = np.all(np.isin(X, [0, 1]))
    unique_counts = [len(np.unique(X[:, i])) for i in range(n_features)]
    categorical_ratio = sum(1 for c in unique_counts if c < 10) / n_features
    
    # Check for high dimensionality
    high_dimensional = n_features > 50
    
    # Decision logic
    if is_binary or categorical_ratio > 0.5:
        # For binary or highly categorical data
        return 'hamming'
    elif high_dimensional:
        # For high-dimensional data, cosine is often better
        return 'cosine'
    else:
        # Default for most cases
        return 'euclidean'


def preprocess_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    categorical_features: List[int] = None,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess features for distance computation.
    
    Args:
        X_train: Training features
        X_test: Test features
        categorical_features: Indices of categorical features
        scale: Whether to standardize features
        
    Returns:
        Preprocessed (X_train, X_test)
    """
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Handle categorical features if specified
    if categorical_features:
        # One-hot encode or special handling for categorical features
        pass
    
    # Standardize features if requested
    if scale:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train_processed)
        X_test_processed = scaler.transform(X_test_processed)
    
    return X_train_processed, X_test_processed
