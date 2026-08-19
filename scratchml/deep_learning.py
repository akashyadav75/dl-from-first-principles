"""
scratchml/deep_learning.py

Industrial-grade Deep Learning Engine built completely from scratch using ONLY NumPy.
Features Modular Layer Architecture, Backpropagation, and advanced Optimizers (SGD, Adam).

Mathematical Concepts Covered:
1. Dense (Fully Connected) Layer: 
   - Forward: Y = X . W + b
   - Backward: dX = dY . W^T, dW = X^T . dY, db = sum(dY, axis=0)
2. Optimizers:
   - SGD with Momentum: v = beta * v + lr * dW, W = W - v
   - Adam: m = beta1 * m + (1-beta1) * dW, v = beta2 * v + (1-beta2) * dW^2, W = W - lr * m_hat / (sqrt(v_hat) + eps)

Generally Used Libraries:
- PyTorch: torch.nn (Linear, Parameter), torch.optim (SGD, Adam)
- TensorFlow: tf.keras.layers (Dense), tf.keras.optimizers (SGD, Adam)
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from scratchml.activations import Activation, Softmax
from scratchml.losses import Loss

# =====================================================================
# 1. BASE LAYER INTERFACE
# =====================================================================

class Layer:
    """
    Abstract Base Class for all Neural Network Layers.
    """
    def __init__(self):
        self.trainable = False
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# =====================================================================
# 2. DENSE (FULLY CONNECTED) LAYER
# =====================================================================

class Dense(Layer):
    """
    Fully Connected (Dense) Neural Network Layer.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.trainable = True
        
        # Math: He (Kaiming) Normal Initialization for weights
        # Line importance: Prevents vanishing/exploding gradients in deep layers.
        self.params['W'] = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.params['b'] = np.zeros((1, output_dim))
        
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['b'] = np.zeros_like(self.params['b'])
        self.inputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Math: Y = X * W + b
        # Line importance: Linear mapping from input dimensions to output dimensions.
        self.inputs = inputs
        return np.dot(inputs, self.params['W']) + self.params['b']

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Math: dW = X^T * dY
        # Math: db = sum(dY, axis=0)
        # Math: dX = dY * W^T
        # Line importance: Computes local gradients and returns upstream gradient for chain rule.
        self.grads['W'] = np.dot(self.inputs.T, output_gradient)
        self.grads['b'] = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.params['W'].T)


# =====================================================================
# 3. ACTIVATION LAYER
# =====================================================================

class ActivationLayer(Layer):
    """
    Wrapper Layer that applies an activation function element-wise.
    """
    def __init__(self, activation: Activation):
        super().__init__()
        self.activation = activation
        self.inputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        # Line importance: Applies activation function forward pass.
        return self.activation.forward(inputs)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        # Math: dX = dY * f'(X)
        # Line importance: Chain rule combination of downstream gradient and local derivative.
        return output_gradient * self.activation.derivative(self.inputs)


# =====================================================================
# 4. OPTIMIZERS
# =====================================================================

class Optimizer:
    """
    Base class for all optimization algorithms.
    """
    def update(self, layer: Layer):
        raise NotImplementedError


class SGDMomentum(Optimizer):
    """
    Stochastic Gradient Descent (SGD) with Momentum.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities: Dict[int, Dict[str, np.ndarray]] = {}

    def update(self, layer: Layer):
        if not layer.trainable:
            return

        layer_id = id(layer)
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                'W': np.zeros_like(layer.params['W']),
                'b': np.zeros_like(layer.params['b'])
            }

        # Math: v = beta * v + lr * dw
        # Math: w = w - v
        # Line importance: Momentum dampens oscillations and speeds up training.
        for param_key in ['W', 'b']:
            self.velocities[layer_id][param_key] = (
                self.momentum * self.velocities[layer_id][param_key] + 
                self.lr * layer.grads[param_key]
            )
            layer.params[param_key] -= self.velocities[layer_id][param_key]


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) Optimizer.
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.m: Dict[int, Dict[str, np.ndarray]] = {}
        self.v: Dict[int, Dict[str, np.ndarray]] = {}
        self.t = 0  # Time step

    def update(self, layer: Layer):
        if not layer.trainable:
            return

        layer_id = id(layer)
        if layer_id not in self.m:
            self.m[layer_id] = {
                'W': np.zeros_like(layer.params['W']),
                'b': np.zeros_like(layer.params['b'])
            }
            self.v[layer_id] = {
                'W': np.zeros_like(layer.params['W']),
                'b': np.zeros_like(layer.params['b'])
            }

        self.t += 1

        for param_key in ['W', 'b']:
            # 1. Update biased first moment estimate
            # Math: m = beta1 * m + (1 - beta1) * g
            self.m[layer_id][param_key] = (
                self.beta1 * self.m[layer_id][param_key] + 
                (1.0 - self.beta1) * layer.grads[param_key]
            )

            # 2. Update biased second raw moment estimate
            # Math: v = beta2 * v + (1 - beta2) * g^2
            self.v[layer_id][param_key] = (
                self.beta2 * self.v[layer_id][param_key] + 
                (1.0 - self.beta2) * (layer.grads[param_key] ** 2)
            )

            # 3. Compute bias-corrected first moment estimate
            # Math: m_hat = m / (1 - beta1^t)
            m_hat = self.m[layer_id][param_key] / (1.0 - self.beta1 ** self.t)

            # 4. Compute bias-corrected second raw moment estimate
            # Math: v_hat = v / (1 - beta2^t)
            v_hat = self.v[layer_id][param_key] / (1.0 - self.beta2 ** self.t)

            # 5. Update parameters
            # Math: theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
            # Line importance: Adapts learning rates individually for every weight.
            layer.params[param_key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =====================================================================
# 5. SEQUENTIAL MODEL
# =====================================================================

class Sequential:
    """
    Sequential Container for Neural Network Layers.
    """
    def __init__(self, layers: List[Layer] = None):
        self.layers = layers if layers is not None else []

    def add(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Line importance: Flows inputs sequentially through all layers.
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        # Line importance: Backpropagates upstream gradients in reverse order.
        grad = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, 
            loss_fn: Loss, optimizer: Optimizer, batch_size: int = 32):
        """
        Fits the neural network to the dataset using mini-batch gradient descent.
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle dataset at start of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = int(np.ceil(n_samples / batch_size))

            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                xb = X_shuffled[start_idx:end_idx]
                yb = y_shuffled[start_idx:end_idx]

                # 1. Forward Pass
                y_pred = self.forward(xb)

                # 2. Compute Loss
                loss = loss_fn(yb, y_pred)
                epoch_loss += loss

                # 3. Backward Pass
                loss_grad = loss_fn.gradient(yb, y_pred)
                self.backward(loss_grad)

                # 4. Parameter Updates
                for layer in self.layers:
                    optimizer.update(layer)

            # Print progress
            avg_loss = epoch_loss / num_batches
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


