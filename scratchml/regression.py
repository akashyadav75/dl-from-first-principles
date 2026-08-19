"""
scratchml/regression.py

Industrial-grade Linear and Logistic Regression models built from scratch using NumPy.

Mathematical Concepts Covered:
1. Linear Regression: 
   - Hypothesis: y = X . w + b
   - Analytical Solution (Normal Equation): w = (X^T . X)^-1 . X^T . y
   - Gradient Descent: w = w - lr * dW, b = b - lr * dB
2. Logistic Regression:
   - Hypothesis: y = sigmoid(X . w + b)
   - Loss: Binary Cross-Entropy
   - Regularization: L1 (Lasso) and L2 (Ridge)

Generally Used Libraries:
- scikit-learn: sklearn.linear_model.LinearRegression, sklearn.linear_model.LogisticRegression
"""

import numpy as np
from scratchml.losses import MeanSquaredError, BinaryCrossEntropy
from scratchml.activations import Sigmoid

class LinearRegression:
    """
    Ordinary Least Squares (OLS) Linear Regression with Gradient Descent and Closed-Form support.
    """
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, fit_intercept: bool = True):
        self.lr = learning_rate
        self.epochs = epochs
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = 'gradient_descent'):
        """
        Fits the Linear Regression model to the training data.
        """
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        # Line importance: Direct analytical solution using the Normal Equation
        if method == 'normal_equation':
            if self.fit_intercept:
                # Add column of ones for intercept
                X_b = np.c_[np.ones((n_samples, 1)), X]
                # Math: theta = (X^T * X)^-1 * X^T * y
                theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
                self.bias = theta[0, 0]
                self.weights = theta[1:].reshape(-1, 1)
            else:
                self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
                self.bias = 0.0
            return self

        # Initialize parameters for Gradient Descent
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        loss_fn = MeanSquaredError()

        # Line importance: Gradient Descent optimization loop
        for epoch in range(self.epochs):
            y_pred = self.predict(X).reshape(-1, 1)
            
            # Compute loss
            loss = loss_fn(y, y_pred)
            self.loss_history.append(loss)

            # Math: dW = (1/N) * X^T * (y_pred - y_true)
            # Math: dB = (1/N) * sum(y_pred - y_true)
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Math: y = X * w + b
        # Line importance: Core linear transformation.
        return np.dot(X, self.weights) + self.bias


class LogisticRegression:
    """
    Logistic Regression Classifier with L1 and L2 regularization support.
    """
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, 
                 penalty: str = 'l2', C: float = 1.0, fit_intercept: bool = True):
        self.lr = learning_rate
        self.epochs = epochs
        self.penalty = penalty
        self.C = C  # Inverse of regularization strength: smaller values specify stronger regularization
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        self.sigmoid = Sigmoid()

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        # Parameter Initialization
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        loss_fn = BinaryCrossEntropy()

        for epoch in range(self.epochs):
            # Line importance: Forward pass mapping continuous space into probability domain [0, 1]
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_output)

            # Compute Base Loss
            loss = loss_fn(y, y_pred)

            # Add Regularization penalty
            # Math: L1 = (1/C) * sum(|w|), L2 = (1/2C) * sum(w^2)
            if self.penalty == 'l2':
                loss += (0.5 / self.C) * np.sum(self.weights ** 2)
            elif self.penalty == 'l1':
                loss += (1.0 / self.C) * np.sum(np.abs(self.weights))
            self.loss_history.append(loss)

            # Compute gradients
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Apply gradient penalty for regularization
            if self.penalty == 'l2':
                dw += (1.0 / self.C) * self.weights
            elif self.penalty == 'l1':
                dw += (1.0 / self.C) * np.sign(self.weights)

            # Parameter Updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Line importance: Returns continuous probability values.
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        # Line importance: Binarizes probabilities into hard classes.
        return (self.predict_proba(X) >= threshold).astype(int)
