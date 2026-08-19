"""
scratchml/svm.py

Support Vector Machine (SVM) Classifier built from scratch using NumPy.

Mathematical Concepts Covered:
1. Linear SVM Hypothesis: y_pred = sign(X . w - b)
2. Loss Function (Hinge Loss + L2 Regularization):
   L(w, b) = lambda * ||w||^2 + (1/N) * sum(max(0, 1 - y_i * (X_i . w - b)))
3. Subgradient Descent:
   - If y_i * (X_i . w - b) >= 1:
     dw = 2 * lambda * w
     db = 0
   - Else:
     dw = 2 * lambda * w - y_i * X_i
     db = y_i

Generally Used Libraries:
- scikit-learn: sklearn.svm.SVC, sklearn.svm.LinearSVC
"""

import numpy as np

class LinearSVM:
    """
    Support Vector Machine Classifier with Soft Margin (L2 Regularization) optimized via Subgradient Descent.
    """
    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, epochs: int = 1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the SVM model using Subgradient Descent.
        Assumes binary targets are represented as {-1, 1}.
        """
        n_samples, n_features = X.shape
        
        # Ensure target labels are strictly -1 and 1
        y_transformed = np.where(y <= 0, -1, 1)

        # Parameter Initialization
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Math: condition = y_i * (x_i . w - b)
                # Line importance: Checks if sample lies correctly outside the margin.
                condition = y_transformed[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1

                if condition:
                    # Only regularization loss gradient flows
                    dw = 2 * self.lambda_param * self.weights
                    db = 0.0
                else:
                    # Both regularization and misclassification hinge loss gradients flow
                    dw = 2 * self.lambda_param * self.weights - y_transformed[idx] * x_i
                    db = y_transformed[idx]

                # Parameter Updates
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Math: sign(X . w - b)
        # Line importance: Maps real values to margin sides {-1, 1}.
        linear_output = np.dot(X, self.weights) - self.bias
        return np.where(linear_output >= 0, 1, 0)
