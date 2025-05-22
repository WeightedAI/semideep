# SemiDeep

A PyTorch implementation of the paper "Enhancing Classification with Semi-Supervised Deep Learning Using Distance-Based Sample Weights" (ICMLT 2025).

<img src="https://img.shields.io/badge/PyTorch-1.7+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"/> <img src="https://img.shields.io/badge/Python-3.7+-3776AB?logo=python&logoColor=white" alt="Python"/> <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>

## Overview

SemiDeep is a Python package that implements a semi-supervised deep learning approach using distance-based sample weights. By computing weights based on the proximity between training and test samples, this method enhances model generalization and robustness, especially in scenarios with:

- Limited labeled data
- Class imbalance
- Noisy labels
- Domain shift between training and test data

The core concept is to assign higher weights to training samples that are more similar to test samples, thereby focusing the learning process on the most informative examples. This approach is implemented as a PyTorch extension that can be easily integrated with existing deep learning models.

## Theory and Method

The SemiDeep approach is based on the following key insights:

1. **Distance-Based Weighting**: Training samples that are closer to test samples in the feature space are likely more relevant for generalization
2. **Exponential Decay Function**: The influence of distance is controlled by an exponential decay function with a λ parameter
3. **Weighted Loss**: Sample weights are incorporated into the loss function to guide the learning process

### Weight Computation Formula

For each training sample x_i, the weight w_i is computed as:

```
w_i = (1/M) * Σ_j exp(-λ · d(x_i, x_j'))
```

where:
- `x_i` is a training sample
- `x_j'` is a test sample
- `d` is a distance metric (euclidean, cosine, hamming, or jaccard)
- `λ` is a decay parameter controlling the influence of distance (typically 0.5-1.0)
- `M` is the number of test samples

### Weighted Loss Function

The weighted loss function is:

```
L = (1/N) * Σ_i w_i · Loss(y_i, f(x_i))
```

where Loss is typically cross-entropy for classification tasks.

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.7.0+
- NumPy 1.19.0+
- scikit-learn 0.24.0+
- pandas 1.1.0+
- matplotlib 3.3.0+
- seaborn 0.11.0+
- tqdm 4.50.0+

### Install from source

```bash
git clone https://github.com/aydinabedinia/SemiDeep.git
cd SemiDeep
pip install -e .
```

## Package Structure

SemiDeep is organized into the following modules:

```
semideep/
├── weight_computer.py  # Weight computation between training and test samples
├── loss.py            # Weighted loss functions 
├── trainer.py         # Training utilities with weighting mechanism
├── experiment.py      # Experiment runner for comparative evaluation
└── utils.py           # Helper functions for metric selection and preprocessing
```

## Quick Start

```python
from semideep import WeightedTrainer
import torch.nn as nn

# Define your PyTorch model
model = your_model()

# Create a weighted trainer
trainer = WeightedTrainer(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    weights="distance",  # Use distance-based weighting
    distance_metric="cosine",
    lambda_=0.8
)

# Train the model
trainer.train(X_val, y_val)

# Evaluate the model
metrics = trainer.evaluate(X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
```

## Key Components

### WeightComputer

Computes distance-based weights between training and test samples using various distance metrics:

```python
from semideep import WeightComputer

weight_computer = WeightComputer(
    distance_metric="euclidean",  # Options: euclidean, cosine, hamming, jaccard
    lambda_=0.8  # Decay parameter
)

# Compute weights
weights = weight_computer.compute_weights(X_train, X_test)

# Access as PyTorch tensor for batch training
tensor_weights = weight_computer.get_tensor_weights()

# Save/load weights
weight_computer.save_weights("model_weights.npy")
weight_computer.load_weights("model_weights.npy")
```

### Distance Metric Selection

SemiDeep provides utilities for automatically selecting the optimal distance metric for your data:

```python
from semideep import auto_select_distance_metric, select_best_distance_metric

# Automatically select based on data characteristics
best_metric = auto_select_distance_metric(X_train)

# Find optimal metric and lambda through cross-validation
best_metric, best_lambda, best_score = select_best_distance_metric(
    model,
    X_train, y_train, X_test,
    metrics=['euclidean', 'cosine', 'hamming', 'jaccard'],
    lambda_values=[0.5, 0.7, 0.8, 0.9, 1.0]
)
```

### WeightedLoss

Applies sample weights to standard loss functions:

```python
from semideep import WeightedLoss, WeightedCrossEntropyLoss
import torch.nn as nn

# Option 1: Wrap standard cross entropy loss
weighted_loss = WeightedLoss(nn.CrossEntropyLoss())

# Option 2: Use convenience class
weighted_ce_loss = WeightedCrossEntropyLoss()

# Option 3: Binary classification
weighted_bce_loss = WeightedBCEWithLogitsLoss()

# Use in training
loss = weighted_loss(predictions, targets, weights)
```

### WeightedTrainer

Integrates weight computation and weighted loss for training:

```python
from semideep import WeightedTrainer

trainer = WeightedTrainer(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    weights="distance", 
    distance_metric="euclidean",
    lambda_=0.8,
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    device="cuda"
)

# Train and evaluate
history = trainer.train(X_val, y_val)
metrics = trainer.evaluate(X_test, y_test)

# Make predictions
predictions = trainer.predict(X_new)
probabilities = trainer.predict_proba(X_new)
```

### ExperimentRunner

Runs comprehensive experiments comparing baseline, weighted, and IDW approaches:

```python
from semideep import ExperimentRunner

# Define model factory function
def model_factory():
    return YourModel()

# Create experiment runner
runner = ExperimentRunner(
    model_factory=model_factory,
    distance_metrics=["euclidean", "cosine"],
    lambda_values=[0.5, 0.8, 1.0],
    test_sizes=[0.1, 0.5, 0.9],
    output_dir="./results"
)

# Run single dataset experiment
results = runner.run_dataset_experiment(
    X, y, 
    dataset_name="my_dataset",
    test_size=0.2,
    epochs=100
)

# Run experiments with different test sizes
test_size_results = runner.run_test_size_experiment(X, y, "my_dataset")

# Generate visualizations
runner.plot_metric_comparison(metric="f1")
runner.plot_test_size_impact(dataset_name="my_dataset", metric="accuracy")

# Generate comprehensive report
report = runner.generate_report()
```

## Examples

The `examples` directory contains scripts demonstrating different aspects of SemiDeep:

### Basic Usage: basic.py

```python
from semideep import WeightedTrainer
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

# Create and train model
model = SimpleNN(X_train.shape[1])
trainer = WeightedTrainer(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    weights="distance",
    distance_metric="euclidean",
    lambda_=0.8,
    epochs=50
)

trainer.train()
metrics = trainer.evaluate(X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
```

### Advanced Distance Metric Selection: metrics.py

This script demonstrates how to select the optimal distance metric for your data:

```bash
# Find the best distance metric and lambda value through cross-validation
python examples/metrics.py

# Use automatic data-driven metric selection
python examples/metrics.py --auto

# Specify test size and number of epochs
python examples/metrics.py --test-size 0.2 --epochs 150
```

### Comparative Experiments: experiment.py

This script runs comprehensive experiments comparing baseline, weighted, and IDW approaches:

```bash
# Run experiment on a dataset
python examples/experiment.py --dataset breast_cancer

# Experiment with different test sizes
python examples/experiment.py --dataset breast_cancer --test-sizes

# Generate report and visualizations
python examples/experiment.py --dataset breast_cancer --report
```

## Performance Benchmarks

SemiDeep has been evaluated on multiple datasets and consistently shows improvements over baseline methods, especially in challenging scenarios:

| Dataset | Test Size | Baseline Accuracy | SemiDeep Accuracy | Improvement |
|---------|-----------|-------------------|-------------------|-----------|
| Breast Cancer | 20% | 0.947 | 0.965 | +1.9% |
| Breast Cancer | 50% | 0.912 | 0.944 | +3.5% |
| Breast Cancer | 80% | 0.868 | 0.921 | +6.1% |

As the test set size increases (less training data available), the benefits of distance-based weighting become more pronounced.

## Contributing

Contributions to SemiDeep are welcome! Please feel free to submit a Pull Request.

### Guidelines
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use SemiDeep in your research, please cite:

```bibtex
@inproceedings{abedinia2025enhancing,
  title={Enhancing Classification with Semi-Supervised Deep Learning Using Distance-Based Sample Weights},
  author={Abedinia, Aydin, Tabakhi, Shima, Seydi, Vahid},
  booktitle={https://doi.org/10.48550/arXiv.2505.14345},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to the PyTorch team for their excellent deep learning framework
- The authors of scikit-learn for their comprehensive machine learning tools
