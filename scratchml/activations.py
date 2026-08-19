"""
scratchml/activations.py

Activation Functions and their mathematical derivatives implemented from scratch using NumPy.

Mathematical Concepts Covered:
1. Sigmoid: f(x) = 1 / (1 + e^-x), f'(x) = f(x) * (1 - f(x))
2. Tanh: f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x), f'(x) = 1 - f(x)^2
3. ReLU (Rectified Linear Unit): f(x) = max(0, x), f'(x) = 1 if x > 0 else 0
4. LeakyReLU: f(x) = max(alpha * x, x), f'(x) = 1 if x > 0 else alpha
5. Softmax: f(x)_i = e^x_i / sum(e^x_j) (with numerical stability shift), f'(x) is handled inside Cross-Entropy loss.

Generally Used Libraries: 
- PyTorch: torch.nn.functional (F.sigmoid, F.relu, F.softmax)
- TensorFlow: tf.keras.activations (sigmoid, relu, softmax)
"""

import numpy as np

class Activation:
    """
    Base class for all activation functions.
    Ensures standard interface for forward and backward passes.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the activation function with respect to its input.
        """
        raise NotImplementedError


class Sigmoid(Activation):
    """
    Sigmoid Activation Function.
    Maps real-valued inputs to [0, 1], representing probabilities.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Math: f(x) = 1 / (1 + exp(-x))
        # Line importance: Numerically stable split formulation to prevent exp overflow/underflow.
        # Uses standard sigmoid for x >= 0, and exp(x)/(1 + exp(x)) for x < 0.
        return np.where(x >= 0, 
                        1.0 / (1.0 + np.exp(-x)), 
                        np.exp(x) / (1.0 + np.exp(x)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        # Math: f'(x) = f(x) * (1 - f(x))
        s = self.forward(x)
        return s * (1.0 - s)


class Tanh(Activation):
    """
    Hyperbolic Tangent Activation Function.
    Maps real-valued inputs to [-1, 1], zero-centered.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Math: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        # Line importance: Uses NumPy's highly optimized vectorized tanh.
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        # Math: tanh'(x) = 1 - tanh^2(x)
        # Line importance: Calculates derivative with respect to input using the forward value.
        t = np.tanh(x)
        return 1.0 - t ** 2


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU).
    Solves the vanishing gradient problem for positive inputs.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Math: max(0, x)
        # Line importance: Vectorized maximum operation, setting all negative inputs to 0.
        return np.maximum(0.0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        # Math: 1 if x > 0 else 0
        # Line importance: Generates a boolean mask and casts to float, acting as a gate for backpropagation.
        return (x > 0).astype(float)


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit (LeakyReLU).
    Prevents "dying ReLU" problem by allowing a small, non-zero gradient when x < 0.
    """
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Math: x if x > 0 else alpha * x
        # Line importance: Uses np.where to apply the alpha scaling scalar element-wise.
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        # Math: 1 if x > 0 else alpha
        # Line importance: Returns gradient of 1.0 for positive inputs and alpha for negative inputs.
        return np.where(x > 0, 1.0, self.alpha)


class Softmax(Activation):
    """
    Softmax Activation Function.
    Converts logits into a probability distribution over K classes.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Math: Softmax_i = exp(x_i) / sum(exp(x_j))
        # Line importance: Subtracting max(x) along columns ensures numerical stability (prevents exp overflow).
        # Keepdims=True keeps dimensions compatible for broadcasting.
        exp_shift = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_shift / np.sum(exp_shift, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        # Note: Softmax derivative is a Jacobian matrix: S_i * (kronecker_delta_ij - S_j).
        # In practice, combined with Cross-Entropy Loss, the gradient simplifies to (y_hat - y).
        # Line importance: Returns forward pass for reference or custom chain-rule implementations.
        return self.forward(x)


