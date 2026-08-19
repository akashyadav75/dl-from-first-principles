"""
scratchml/losses.py

Loss Functions and their mathematical derivatives implemented from scratch using NumPy.

Mathematical Concepts Covered:
1. Mean Squared Error (MSE): L = (1/2N) * sum((y_pred - y_true)^2), dL/dy_pred = (1/N) * (y_pred - y_true)
2. Mean Absolute Error (MAE): L = (1/N) * sum(|y_pred - y_true|), dL/dy_pred = (1/N) * sign(y_pred - y_true)
3. Binary Cross-Entropy (BCE): L = -(1/N) * sum(y*log(p) + (1-y)*log(1-p)), dL/dy_pred = -(1/N) * [y/p - (1-y)/(1-p)]
4. Categorical Cross-Entropy (CCE): L = -(1/N) * sum(sum(y_ij * log(p_ij))), dL/dy_pred = -(1/N) * (y / p)

Generally Used Libraries:
- PyTorch: torch.nn (MSELoss, BCELoss, CrossEntropyLoss)
- TensorFlow: tf.keras.losses (MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy)
"""

import numpy as np

class Loss:
    """
    Base class for all loss functions.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.forward(y_true, y_pred)

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) Loss for Regression tasks.
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Math: MSE = (1 / 2N) * sum((y_pred - y_true)^2)
        # Line importance: Computes average squared deviation. Factor of 2 simplifies the derivative.
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Math: dL / dy_pred = (y_pred - y_true) / N
        # Line importance: Returns gradient vector with respect to predictions, scaled by batch size.
        return (y_pred - y_true) / y_true.size


class MeanAbsoluteError(Loss):
    """
    Mean Absolute Error (MAE) Loss / L1 Loss for robust regression.
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Math: MAE = (1 / N) * sum(|y_pred - y_true|)
        # Line importance: Computes absolute error; less sensitive to outliers than MSE.
        return np.mean(np.abs(y_pred - y_true))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Math: dL / dy_pred = (1 / N) * sign(y_pred - y_true)
        # Line importance: Gradient is either +1 or -1 scaled by batch size, undefined exactly at 0 (handled by np.sign).
        return np.sign(y_pred - y_true) / y_true.size


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy (BCE) Loss for binary classification tasks.
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Math: BCE = -mean(y * log(p) + (1 - y) * log(1 - p))
        # Line importance: Clip predictions to prevent taking log(0) which returns NaN/inf.
        p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Math: dL / dp = -(y / p - (1 - y) / (1 - p)) / N
        # Line importance: Clamping prevents division by zero when calculating gradients.
        p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return -((y_true / p) - ((1.0 - y_true) / (1.0 - p))) / y_true.size


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy (CCE) Loss for multi-class classification tasks.
    Assumes one-hot encoded targets.
    """
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Math: CCE = -mean(sum_k(y_k * log(p_k)))
        # Line importance: Clamping predictions avoids log(0) numeric instability.
        p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return -np.sum(y_true * np.log(p)) / y_true.shape[0]

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Math: dL / dp = -(y / p) / N
        # Line importance: Gradient of multi-class loss with respect to probabilities.
        p = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return -(y_true / p) / y_true.shape[0]


class BCEWithLogitsLoss(Loss):
    """
    Numerically stable Binary Cross-Entropy with built-in Sigmoid activation.
    Combines Sigmoid and BCE into a single mathematical step to avoid logarithmic underflow.
    
    Math Formulation:
    L = -[y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))]
    Simplifies to:
    L = max(x, 0) - x * y + log(1 + exp(-|x|))
    """
    def forward(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        # Line importance: Uses the log-sum-exp split formulation to guarantee absolute numerical stability.
        loss = np.maximum(logits, 0) - logits * y_true + np.log(1.0 + np.exp(-np.abs(logits)))
        return float(np.mean(loss))

    def gradient(self, y_true: np.ndarray, logits: np.ndarray) -> np.ndarray:
        # Math: dL / dx = (sigmoid(x) - y) / N
        # Line importance: Extremely simple, elegant, and stable gradient expression.
        # Uses numerically stable sigmoid formulation internally.
        p = np.where(logits >= 0, 
                     1.0 / (1.0 + np.exp(-logits)), 
                     np.exp(logits) / (1.0 + np.exp(logits)))
        return (p - y_true) / y_true.size


