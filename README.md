# 🚀 scratchml

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Zero External Dependencies](https://img.shields.io/badge/dependencies-NumPy%20%26%20Pandas%20only-green)](https://numpy.org/)

**scratchml** is a production-grade, highly-vectorized, educational Machine Learning and Deep Learning library built entirely **from scratch** using only **NumPy** and **Pandas**. 

This library was designed to eliminate "black-box" abstractions by implementing core mathematical equations, optimization routines, data streaming pipelines, and advanced neural architectures (CNN, LSTM, Transformers, GANs) from basic mathematical derivations up.

---

## 🌟 Key Features & Implemented Architectures

### 1. 🧠 Deep Learning & Advanced Architectures (`scratchml/advanced_dl.py`)
*   **Vectorized CNN (`Conv2D`)**: Implements spatial convolution optimized via the **`im2col` (image-to-column)** and **`col2im`** GEMM (General Matrix Multiplication) formulation. Replaces slow Python loops with a single, highly-vectorized BLAS matrix multiplication.
*   **Sequence-level LSTM**: Implements Backpropagation Through Time (BPTT) over 3D batch sequences of shape `(batch_size, seq_len, input_dim)`.
*   **Scaled Dot-Product Attention**: The core mathematical engine behind Transformers:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 2. 🎛️ Automatic Differentiation Engine (`scratchml/autograd.py`)
*   A lightweight, fully functional **Autograd Engine** featuring a custom `Tensor` object. It dynamically builds a Directed Acyclic Graph (DAG) during the forward pass and performs a **Topological Sort** on `.backward()` to automatically compute gradients via the chain rule.

### 3. 🎨 Generative Adversarial Networks (`scratchml/gan.py`)
*   A complete **GAN** implementing a Generator and a Discriminator playing a minimax game:
    $$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$$
*   Features stable upstream gradient routing from the Discriminator back into the Generator.

### 4. 📊 Supervised & Unsupervised Machine Learning
*   **Regression**: Linear Regression (Gradient Descent & Closed-Form Normal Equation) and Logistic Regression with **L1 (Lasso)** and **L2 (Ridge)** regularization.
*   **Trees & Ensembles**: Decision Trees (Gini, Entropy, and Variance Reduction criteria) and **Random Forest Classifiers** with bootstrap bagging and feature bagging.
*   **Support Vector Machines**: Soft-Margin Linear SVM optimized via Subgradient Descent.
*   **Unsupervised Learning**: K-Means Clustering and Principal Component Analysis (PCA) via Eigendecomposition.
*   **Probabilistic & Instance-based**: Gaussian Naive Bayes and K-Nearest Neighbors (KNN Classifier & Regressor).

### 5. 🗄️ Production Data Pipeline (`scratchml/data.py`)
*   Memory-efficient `Dataset` and generator-based `DataLoader` streaming pipeline. Handles shuffling, batching, and on-the-fly collation to **prevent Out-Of-Memory (OOM) errors** for larger-than-RAM datasets.

### 6. 🛡️ Mathematical Stability & Observability
*   Absolute numerical stability using stable split formulations for Sigmoid and a unified **`BCEWithLogitsLoss`** with log-sum-exp split formulations.

---

## 📁 Repository Structure

```bash
scratchml/
├── __init__.py           # Package-level initialization & versioning
├── activations.py        # Sigmoid, Tanh, ReLU, LeakyReLU, Softmax (Forward & Derivatives)
├── autograd.py           # Computation Graph DAG & Topological Sort Autograd Engine
├── losses.py             # MSE, MAE, BCE, CCE, and stable BCEWithLogitsLoss
├── regression.py         # Linear & Logistic Regression (with L1/L2 penalties)
├── trees.py              # Decision Trees (Classifier/Regressor) & Random Forest
├── svm.py                # Soft-Margin Linear SVM via Subgradient Descent
├── unsupervised.py       # K-Means Clustering & PCA (via Eigendecomposition)
├── neighbors.py          # K-Nearest Neighbors (Classifier & Regressor)
├── naive_bayes.py        # Gaussian Naive Bayes (with log-sum-exp trick)
├── deep_learning.py      # Dense Layers, SGDMomentum, Adam Optimizers, Sequential Container
├── advanced_dl.py        # im2col Conv2D, Sequence LSTM (BPTT), Scaled Dot-Product Attention
├── gan.py                # Generative Adversarial Network (Generator & Discriminator)
├── data.py               # Memory-efficient Dataset and DataLoader Pipeline
└── metrics.py            # Accuracy, Precision, Recall, F1, Confusion Matrix, R2-Score
test_scratchml.py         # Comprehensive verification and integration test suite
setup.py                  # Installation setup script
requirements.txt          # Minimal external dependencies
LICENSE                   # MIT License
.gitignore                # Python-specific git ignore files
```

---

## 🚀 Getting Started

### Installation

Clone the repository and install the package locally:

```bash
git clone https://github.com/akashyadav75/scratchml.git
cd scratchml
pip install -r requirements.txt
pip install -e .
```

### Running the Verification Suite

Run the integration test suite to verify convergence and mathematical correctness for all models:

```bash
python test_scratchml.py
```

---

## 📖 Educational Code Examples

### 1. Training a Neural Network with the Modular Engine
```python
import numpy as np
from scratchml.deep_learning import Sequential, Dense, ActivationLayer, Adam
from scratchml.activations import ReLU, Softmax
from scratchml.losses import CategoricalCrossEntropy

# 1. Instantiate the Model
model = Sequential([
    Dense(input_dim=2, output_dim=16),
    ActivationLayer(ReLU()),
    Dense(input_dim=16, output_dim=3),
    ActivationLayer(Softmax())
])

# 2. Configure Optimizers and Losses
optimizer = Adam(learning_rate=0.01)
loss_fn = CategoricalCrossEntropy()

# 3. Train Model
model.fit(X_train, y_train_onehot, epochs=50, loss_fn=loss_fn, optimizer=optimizer, batch_size=16)
```

### 2. Using the Autograd Engine
```python
from scratchml.autograd import Tensor

# Define nodes in the computation graph
x = Tensor([[2.0, 3.0]])
w = Tensor([[1.0], [4.0]])
b = Tensor([[0.5]])
y = Tensor([[14.0]])

# Forward pass automatically builds the DAG
loss = (x.dot(w) + b) - y

# Backward pass automatically performs topological sorting and executes the chain rule
loss.backward()

print(f"Gradient wrt weights (dw):\n{w.grad}") # [[2.0], [3.0]]
```

### 3. Memory-Efficient Data Streaming
```python
from scratchml.data import SimpleDataset, DataLoader

# Wrap your NumPy arrays in a Dataset
dataset = SimpleDataset(X_large, y_large)

# Stream mini-batches on-the-fly to prevent memory overflows
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch_x, batch_y in dataloader:
    predictions = model.forward(batch_x)
    # Train step ...
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎓 Author

Developed by **Akash Yadav** (akashyadav812733@gmail.com). Feel free to reach out for questions, research collaborations, or contributions!
