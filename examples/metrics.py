#!/usr/bin/env python
"""
Advanced distance metric selection example for SemiDeep.

This script demonstrates how to use the automatic distance metric selection 
capabilities in SemiDeep, similar to the approach used in SemiCART.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import argparse
import sys
import os

# Add parent directory to path to import semideep
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semideep import (
    WeightComputer, 
    WeightedTrainer, 
    select_best_distance_metric,
    auto_select_distance_metric
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SemiDeep-Advanced")


class SimpleTabularModel(nn.Module):
    """Simple MLP model for tabular data."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=2):
        """Initialize model."""
        super().__init__()
        
        # Build layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x):
        """Forward pass."""
        features = self.feature_layers(x)
        output = self.output_layer(features)
        return output


def main():
    """Run the demonstration."""
    parser = argparse.ArgumentParser(description='SemiDeep advanced metric selection example')
    parser.add_argument('--auto', action='store_true', 
                        help='Use automatic metric selection based on data characteristics')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='Test set size ratio')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    args = parser.parse_args()
    
    # Load and preprocess dataset
    logger.info("Loading Breast Cancer dataset")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Create model
    input_dim = X_train.shape[1]
    
    if args.auto:
        # Use automatic metric selection based on data characteristics
        best_metric = auto_select_distance_metric(X_train)
        best_lambda = 0.8  # Default lambda
        logger.info(f"Auto-selected distance metric: {best_metric}, lambda: {best_lambda}")
    else:
        # Use cross-validation to find the best metric and lambda value
        logger.info("Finding best distance metric and lambda via cross-validation...")
        
        metrics_to_try = ['euclidean', 'cosine', 'hamming', 'jaccard']
        lambda_values = [0.5, 0.7, 0.8, 0.9, 1.0]
        
        def create_model():
            return SimpleTabularModel(input_dim=input_dim, hidden_dims=[64, 32], num_classes=2)
        
        model_for_cv = create_model()
        best_metric, best_lambda, best_score = select_best_distance_metric(
            model=model_for_cv,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            metrics=metrics_to_try,
            lambda_values=lambda_values,
            verbose=True
        )
        
        logger.info(f"Best metric: {best_metric}, Best lambda: {best_lambda}, CV Score: {best_score:.4f}")
    
    # Train with the best metric and lambda
    model = SimpleTabularModel(input_dim=input_dim, hidden_dims=[64, 32], num_classes=2)
    
    # Create weight computer with best metric
    weight_computer = WeightComputer(
        distance_metric=best_metric,
        lambda_=best_lambda
    )
    
    # Compute weights
    weights = weight_computer.compute_weights(X_train, X_test)
    
    # Train with weighted loss
    trainer = WeightedTrainer(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        weights=weights,
        epochs=args.epochs,
        verbose=True
    )
    
    logger.info(f"Training model with {best_metric} distance metric and lambda={best_lambda}")
    trainer.train()
    
    # Evaluate on test set
    metrics = trainer.evaluate(X_test, y_test)
    
    logger.info("Test Set Performance:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # For comparison, train a baseline model without weights
    logger.info("Training baseline model without weights for comparison...")
    baseline_model = SimpleTabularModel(input_dim=input_dim, hidden_dims=[64, 32], num_classes=2)
    baseline_trainer = WeightedTrainer(
        model=baseline_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        weights=None,  # No weights
        epochs=args.epochs,
        verbose=True
    )
    
    baseline_trainer.train()
    baseline_metrics = baseline_trainer.evaluate(X_test, y_test)
    
    logger.info("Baseline Test Set Performance:")
    for metric_name, value in baseline_metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Compare the two models
    logger.info("\nPerformance Improvement:")
    for metric_name in metrics:
        if metric_name != 'val_loss':
            improvement = metrics[metric_name] - baseline_metrics[metric_name]
            percent = improvement / max(baseline_metrics[metric_name], 1e-10) * 100
            logger.info(f"{metric_name}: {improvement:.4f} ({percent:.2f}%)")


if __name__ == "__main__":
    main()
