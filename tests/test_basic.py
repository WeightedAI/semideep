import unittest
import numpy as np
from semideep import WeightComputer

class TestWeightComputer(unittest.TestCase):
    def test_weight_computation(self):
        """Test that weight computation works as expected."""
        # Create sample data
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_test = np.array([[2.0, 2.0], [4.0, 4.0]])
        
        # Initialize weight computer
        weight_computer = WeightComputer(distance_metric="euclidean", lambda_=0.8)
        
        # Compute weights
        weights = weight_computer.compute_weights(X_train, X_test)
        
        # Check weights shape
        self.assertEqual(weights.shape, (3,))
        
        # Check weights are positive
        self.assertTrue(np.all(weights > 0))
        
        # Check weights sum to > 0
        self.assertTrue(np.sum(weights) > 0)

if __name__ == '__main__':
    unittest.main()
