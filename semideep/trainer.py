"""
Trainer module for distance-based semi-supervised learning.

This module implements the training process using the weighted loss mechanism 
described in "Enhancing Classification with Semi-Supervised Deep Learning Using Distance-Based Sample Weights"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Optional, Union, Tuple, Callable
import time
import logging
from tqdm.auto import tqdm

from .weight_computer import WeightComputer
from .loss import WeightedLoss, WeightedCrossEntropyLoss

logger = logging.getLogger(__name__)


class WeightedTrainer:
    """Trainer class that implements the distance-based weighting approach.
    
    This class handles the training of deep learning models with the
    distance-based weighting mechanism.
    """
    
    def __init__(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        weights: Union[str, np.ndarray, None] = "distance",
        distance_metric: str = "euclidean",
        lambda_: float = 0.8,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = None,
        verbose: bool = True
    ):
        """Initialize the weighted trainer.
        
        Args:
            model: PyTorch model to train
            X_train: Training features
            y_train: Training labels
            X_test: Test features for computing distances. Required if weights="distance"
            weights: Type of weighting to use. Options:
                    - "distance": Use distance-based weighting
                    - "idw": Use inverse distance weighting
                    - numpy array: Directly use provided weights
                    - None: No weighting (standard training)
            distance_metric: Distance metric for computing weights
            lambda_: Decay parameter for distance weighting
            loss_fn: Loss function. If None, uses WeightedCrossEntropyLoss
            optimizer: Optimizer. If None, uses Adam
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use ('cpu' or 'cuda'). If None, uses cuda if available
            verbose: Whether to print progress
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize model
        self.model = model.to(self.device)
        
        # Store data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        
        # Training parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        
        # Set up loss function
        if loss_fn is None:
            self.loss_fn = WeightedCrossEntropyLoss().to(self.device)
        else:
            self.loss_fn = loss_fn.to(self.device)
            
        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
            
        # Compute weights if needed
        self.sample_weights = None
        if weights is not None:
            # Check if weights is a string specifying the weighting method
            if isinstance(weights, str):
                if weights == "distance":
                    if X_test is None:
                        raise ValueError("X_test must be provided when weights='distance'")
                    weight_computer = WeightComputer(
                        distance_metric=distance_metric,
                        lambda_=lambda_,
                        device=self.device
                    )
                    self.sample_weights = weight_computer.compute_weights(X_train, X_test)
                elif weights == "idw":  # Inverse Distance Weighting
                    if X_test is None:
                        raise ValueError("X_test must be provided when weights='idw'")
                    # Compute distances
                    if distance_metric == 'cosine':
                        from sklearn.metrics.pairwise import cosine_distances
                        distances = cosine_distances(X_train, X_test)
                    else:
                        from scipy.spatial.distance import cdist
                        distances = cdist(X_train, X_test, metric=distance_metric)
                    # Apply IDW formula: w_i = (1/M) * Σ_j (1/d(x_i, x_j))
                    # Add small epsilon to avoid division by zero
                    epsilon = 1e-10
                    inv_distances = 1.0 / (distances + epsilon)
                    self.sample_weights = np.mean(inv_distances, axis=1)
            elif isinstance(weights, np.ndarray):
                # Direct array of weights provided
                self.sample_weights = weights
            else:
                # Try to convert to numpy array
                try:
                    self.sample_weights = np.array(weights)
                except:
                    raise ValueError("Weights must be 'distance', 'idw', or a numpy array/list of values")
                
        # Create dataset and dataloader
        self.train_dataset, self.train_loader = self._create_dataloader(
            X_train, y_train, self.sample_weights, batch_size
        )
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
    def _create_dataloader(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        weights: Optional[np.ndarray] = None,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Tuple[Dataset, DataLoader]:
        """Create a PyTorch dataset and dataloader from numpy arrays.
        
        Args:
            X: Features
            y: Labels
            weights: Sample weights
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            
        Returns:
            Tuple of (Dataset, DataLoader)
        """
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        
        if weights is not None:
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)
            
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
        
        return dataset, dataloader
    
    def train(
        self, 
        X_val: Optional[np.ndarray] = None, 
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, List[float]]:
        """Train the model using the weighted loss function.
        
        Args:
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Training history dictionary
        """
        self.model.train()
        
        # Set up validation data if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            _, val_loader = self._create_dataloader(X_val, y_val, batch_size=self.batch_size, shuffle=False)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0
            
            # Create progress bar if verbose
            train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") if self.verbose else self.train_loader
            
            for batch in train_iter:
                self.optimizer.zero_grad()
                
                # Unpack batch
                if len(batch) == 3:  # With weights
                    X_batch, y_batch, w_batch = batch
                else:  # Without weights
                    X_batch, y_batch = batch
                    w_batch = None
                
                # Forward pass
                outputs = self.model(X_batch)
                
                # Compute loss
                loss = self.loss_fn(outputs, y_batch, w_batch)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                epoch_loss += loss.item()
                batches += 1
            
            # Compute average epoch loss
            avg_epoch_loss = epoch_loss / batches
            self.history['train_loss'].append(avg_epoch_loss)
            
            # Evaluate on validation set if provided
            if val_loader is not None:
                val_metrics = self.evaluate(X_val, y_val)
                for key, value in val_metrics.items():
                    if key in self.history:
                        self.history[key].append(value)
                        
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_epoch_loss:.4f} - Val Loss: {val_metrics['val_loss']:.4f} - Val Acc: {val_metrics['accuracy']:.4f}")
            else:
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_epoch_loss:.4f}")
        
        train_time = time.time() - start_time
        if self.verbose:
            print(f"Training completed in {train_time:.2f} seconds")
            
        return self.history
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """Evaluate the model on given data.
        
        Args:
            X: Features
            y: Labels
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Convert data to PyTorch tensors and create dataloader
        _, dataloader = self._create_dataloader(X, y, batch_size=self.batch_size, shuffle=False)
        
        # Track metrics
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch (without weights for evaluation)
                X_batch, y_batch = batch[0], batch[1]
                
                # Forward pass
                outputs = self.model(X_batch)
                
                # Compute loss
                loss = self.loss_fn(outputs, y_batch)
                total_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Store predictions and targets
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # Concatenate predictions and targets
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_targets)
        
        # Compute metrics
        metrics = {
            'val_loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        # Precision, recall, F1 (handling binary vs multiclass)
        if len(np.unique(y_true)) == 2:  # Binary classification
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred)
            
            # AUC (needs probability of positive class)
            if y_prob.shape[1] >= 2:  # If we have probability outputs
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:  # Multiclass
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            # Multiclass AUC with OvR strategy
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except ValueError:
                metrics['auc'] = np.nan  # In case of error
        
        if return_predictions:
            return metrics, y_pred, y_prob
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features
            
        Returns:
            Class predictions
        """
        self.model.eval()
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Create dataloader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Make predictions
        all_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                X_batch = batch[0]
                outputs = self.model(X_batch)
                _, preds = torch.max(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
        
        return np.concatenate(all_preds)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions on new data.
        
        Args:
            X: Features
            
        Returns:
            Probability predictions
        """
        self.model.eval()
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Create dataloader
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Make predictions
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                X_batch = batch[0]
                outputs = self.model(X_batch)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs)
    
    def save_model(self, path: str) -> None:
        """Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str) -> None:
        """Load model parameters from a file.
        
        Args:
            path: Path to the saved model
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
