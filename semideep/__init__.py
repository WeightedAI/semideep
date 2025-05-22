"""
SemiDeep: Distance-Based Sample Weights for Semi-Supervised Deep Learning
======================================================================

A PyTorch implementation of the method described in:
"Enhancing Classification with Semi-Supervised Deep Learning Using Distance-Based Sample Weights"

Core modules:
- weight_computer: Computes distance-based weights between training and test samples
- loss: Custom weighted loss functions
- trainer: Training utilities for applying the weighting mechanism
- experiment: Tools for running and evaluating experiments
"""

__version__ = "0.1.0"

from .weight_computer import WeightComputer
from .loss import WeightedLoss
from .trainer import WeightedTrainer
from .experiment import ExperimentRunner
from .utils import select_best_distance_metric, auto_select_distance_metric, preprocess_features

__all__ = [
    'WeightComputer',
    'WeightedLoss',
    'WeightedTrainer',
    'ExperimentRunner',
    'select_best_distance_metric',
    'auto_select_distance_metric',
    'preprocess_features',
]
