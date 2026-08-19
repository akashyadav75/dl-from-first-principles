"""
scratchml/gan.py

A complete, production-grade Generative Adversarial Network (GAN) built completely from scratch using NumPy.

Mathematical Concepts Covered:
1. Minimax Game: min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
2. Discriminator Loss: L_D = -[log D(x) + log(1 - D(G(z)))]
3. Generator Loss: L_G = -[log D(G(z))] (Non-saturating heuristic target)
4. Backpropagation: Flowing binary cross-entropy gradients through both networks sequentially.

Generally Used Libraries:
- PyTorch: torch.nn (Linear, BCEWithLogitsLoss)
- TensorFlow: tf.keras.models (Sequential)
"""

import numpy as np
from typing import Tuple
from scratchml.deep_learning import Sequential, Dense, ActivationLayer, Adam
from scratchml.activations import LeakyReLU, Sigmoid
from scratchml.losses import BinaryCrossEntropy

class GAN:
    """
    Generative Adversarial Network (GAN) engine containing a Generator and a Discriminator.
    """
    def __init__(self, noise_dim: int, data_dim: int, hidden_dim: int = 32):
        self.noise_dim = noise_dim
        self.data_dim = data_dim

        # 1. Generator Network (G)
        # Map noise z -> realistic data representation
        self.generator = Sequential([
            Dense(noise_dim, hidden_dim),
            ActivationLayer(LeakyReLU(0.2)),
            Dense(hidden_dim, data_dim),
            ActivationLayer(Sigmoid()) # Assumes normalized data in range [0, 1]
        ])

        # 2. Discriminator Network (D)
        # Map data representation x -> probability of being real [0, 1]
        self.discriminator = Sequential([
            Dense(data_dim, hidden_dim),
            ActivationLayer(LeakyReLU(0.2)),
            Dense(hidden_dim, 1),
            ActivationLayer(Sigmoid())
        ])

        # Optimizers
        self.opt_D = Adam(learning_rate=0.0002, beta1=0.5)
        self.opt_G = Adam(learning_rate=0.0002, beta1=0.5)
        
        self.loss_fn = BinaryCrossEntropy()

    def train_step(self, real_data: np.ndarray) -> Tuple[float, float]:
        """
        Executes a single minimax training iteration.
        """
        batch_size = real_data.shape[0]

        # =============================================================
        # 1. TRAIN DISCRIMINATOR (D)
        # =============================================================
        # Generate fake data from noise
        # Math: z ~ Uniform/Normal, G_z = G(z)
        z = np.random.randn(batch_size, self.noise_dim)
        fake_data = self.generator.forward(z)

        # Forward passes
        d_pred_real = self.discriminator.forward(real_data)
        d_pred_fake = self.discriminator.forward(fake_data)

        # Labels for Binary Cross Entropy
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))

        # Compute Discriminator Loss
        # Math: L_D = -[log D(x) + log(1 - D(G(z)))]
        loss_D_real = self.loss_fn(labels_real, d_pred_real)
        loss_D_fake = self.loss_fn(labels_fake, d_pred_fake)
        loss_D = loss_D_real + loss_D_fake

        # Backward pass through Discriminator for Real Data
        grad_D_real = self.loss_fn.gradient(labels_real, d_pred_real)
        self.discriminator.backward(grad_D_real)
        for layer in self.discriminator.layers:
            self.opt_D.update(layer)

        # Backward pass through Discriminator for Fake Data
        grad_D_fake = self.loss_fn.gradient(labels_fake, d_pred_fake)
        self.discriminator.backward(grad_D_fake)
        for layer in self.discriminator.layers:
            self.opt_D.update(layer)

        # =============================================================
        # 2. TRAIN GENERATOR (G)
        # =============================================================
        # Generate new fake data
        z = np.random.randn(batch_size, self.noise_dim)
        fake_data = self.generator.forward(z)

        # Pass fake data through Discriminator
        # Line importance: We must forward through Discriminator to set the layer inputs/caches properly before backward pass.
        d_pred_fake_for_G = self.discriminator.forward(fake_data)

        # Generator wants Discriminator to classify fake data as real (target = 1)
        # Math: L_G = -log D(G(z))
        loss_G = self.loss_fn(labels_real, d_pred_fake_for_G)

        # Upstream Gradient: flow loss through Discriminator to Generator
        grad_G_upstream = self.loss_fn.gradient(labels_real, d_pred_fake_for_G)
        
        # Propagate upstream gradient back through the Discriminator to get grad wrt fake_data
        grad_fake_data = self.discriminator.backward(grad_G_upstream)

        # Line importance: Re-run forward pass on generator to reset input caches for G backward pass
        self.generator.forward(z)
        self.generator.backward(grad_fake_data)

        # Update Generator parameters
        for layer in self.generator.layers:
            self.opt_G.update(layer)

        return float(loss_D), float(loss_G)

    def generate(self, num_samples: int) -> np.ndarray:
        """
        Generates synthetic data samples from trained Generator.
        """
        z = np.random.randn(num_samples, self.noise_dim)
        return self.generator.forward(z)


