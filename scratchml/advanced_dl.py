"""
scratchml/advanced_dl.py

Advanced Deep Learning Architectures built from scratch using NumPy.
Covers CNN, RNN/LSTM, and Transformer/Self-Attention mechanisms.

Mathematical Concepts Covered:
1. Convolutional 2D Layer (CNN):
   - Forward: Y(c, r, d) = sum_i sum_j sum_k X(c+i, r+j, k) * W(i, j, k, d) + b(d)
2. Long Short-Term Memory (LSTM):
   - Forget Gate: f_t = sigmoid(W_f . [h_t-1, x_t] + b_f)
   - Input Gate: i_t = sigmoid(W_i . [h_t-1, x_t] + b_i)
   - Candidate: c_tilde_t = tanh(W_c . [h_t-1, x_t] + b_c)
   - Cell State: c_t = f_t * c_t-1 + i_t * c_tilde_t
   - Output Gate: o_t = sigmoid(W_o . [h_t-1, x_t] + b_o)
   - Hidden State: h_t = o_t * tanh(c_t)
3. Scaled Dot-Product Attention (Transformer):
   - Attention(Q, K, V) = softmax((Q . K^T) / sqrt(d_k)) . V

Generally Used Libraries:
- PyTorch: torch.nn (Conv2d, LSTM, MultiheadAttention)
- TensorFlow: tf.keras.layers (Conv2D, LSTM, Attention)
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from scratchml.activations import Sigmoid, Tanh, Softmax

# =====================================================================
# 1. CONVOLUTIONAL 2D LAYER (CNN)
# =====================================================================

class Conv2D:
    """
    2D Convolutional Layer optimized using the im2col (image-to-column) GEMM formulation.
    This replaces nested spatial loops with a single, highly-vectorized NumPy matrix multiplication.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization for kernels
        # Shape: (out_channels, in_channels, kernel_size, kernel_size)
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels, 1))
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.X = None
        self.X_col = None

    def _im2col(self, X: np.ndarray) -> np.ndarray:
        """
        Rearranges image blocks into columns.
        Input shape: (batch_size, in_channels, height, width)
        Returns: (in_channels * k * k, batch_size * h_out * w_out) matrix.
        """
        batch_size, in_channels, h, w = X.shape
        h_out = int((h - self.kernel_size + 2 * self.padding) / self.stride) + 1
        w_out = int((w - self.kernel_size + 2 * self.padding) / self.stride) + 1

        # Apply padding if configured
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Line importance: Pre-allocate memory for column matrix
        col = np.zeros((in_channels, self.kernel_size, self.kernel_size, batch_size, h_out, w_out))

        # Line importance: Simple 2D spatial extraction loop (extremely fast compared to 6 nested loops)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                h_lim = h_out * self.stride
                w_lim = w_out * self.stride
                # Correct broadcasting by aligning batch size and channel dimensions properly
                col[:, i, j, :, :, :] = X_padded[:, :, i:i+h_lim:self.stride, j:j+w_lim:self.stride].transpose(1, 0, 2, 3)

        # Transpose and reshape to output column format
        col = col.transpose(0, 1, 2, 4, 5, 3).reshape(in_channels * self.kernel_size * self.kernel_size, -1)
        return col

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Computes 2D Convolution on input tensor X of shape (batch_size, in_channels, height, width).
        Uses GEMM (General Matrix Multiplication) formulation via im2col.
        """
        self.X = X
        batch_size, in_channels, h, w = X.shape
        h_out = int((h - self.kernel_size + 2 * self.padding) / self.stride) + 1
        w_out = int((w - self.kernel_size + 2 * self.padding) / self.stride) + 1

        # 1. Transform spatial inputs into column structure
        self.X_col = self._im2col(X)

        # 2. Reshape weights for GEMM multiplication
        # Shape: (out_channels, in_channels * kernel_size * kernel_size)
        W_row = self.W.reshape(self.out_channels, -1)

        # 3. Perform single high-speed matrix multiplication (GEMM)
        # Math: Y = W_row . X_col + b
        # Line importance: Replaces 6 nested loops with 1 optimized C-level BLAS matrix multiplication.
        out = np.dot(W_row, self.X_col) + self.b

        # 4. Reshape back to spatial output dimensions
        out = out.reshape(self.out_channels, h_out, w_out, batch_size)
        out = out.transpose(3, 0, 1, 2) # (batch_size, out_channels, h_out, w_out)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Vectorized 2D Convolution Backward Pass.
        """
        # Reshape upstream gradient for matrix multiplication
        # Shape: (out_channels, batch_size * h_out * w_out)
        d_out_reshaped = d_out.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        # 1. Compute bias gradient
        self.db = np.sum(d_out_reshaped, axis=1, keepdims=True)

        # 2. Compute weight kernel gradient (GEMM)
        # Math: dW = d_out_reshaped . X_col^T
        self.dW = np.dot(d_out_reshaped, self.X_col.T).reshape(self.W.shape)

        # 3. Compute input gradient (GEMM)
        # Math: dX_col = W^T . d_out_reshaped
        W_row = self.W.reshape(self.out_channels, -1)
        dX_col = np.dot(W_row.T, d_out_reshaped)

        # 4. Reconstruct input spatial tensor using col2im
        # Line importance: Reconstructs original dimensions and handles padding stripping.
        batch_size, in_channels, h, w = self.X.shape
        dX = self._col2im(dX_col, batch_size, in_channels, h, w)
        return dX

    def _col2im(self, col: np.ndarray, batch_size: int, in_channels: int, h: int, w: int) -> np.ndarray:
        """
        Transforms column matrix back to spatial image representation (col2im).
        """
        h_out = int((h - self.kernel_size + 2 * self.padding) / self.stride) + 1
        w_out = int((w - self.kernel_size + 2 * self.padding) / self.stride) + 1
        h_padded = h + 2 * self.padding
        w_padded = w + 2 * self.padding

        # Reshape column matrix
        col_reshaped = col.reshape(in_channels, self.kernel_size, self.kernel_size, h_out, w_out, batch_size)
        col_reshaped = col_reshaped.transpose(0, 1, 2, 5, 3, 4)

        X_padded = np.zeros((batch_size, in_channels, h_padded, w_padded))

        # Accumulate gradients spatially
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                h_lim = h_out * self.stride
                w_lim = w_out * self.stride
                X_padded[:, :, i:i+h_lim:self.stride, j:j+w_lim:self.stride] += col_reshaped[:, i, j, :, :, :].transpose(1, 0, 2, 3)

        # Strip padding
        if self.padding > 0:
            return X_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return X_padded


# =====================================================================
# 2. LONG SHORT-TERM MEMORY LAYER (LSTM)
# =====================================================================

class LSTMCell:
    """
    A single LSTM cell representing one time step of recurrent computation.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim

        # Gate weights initialization
        # Gates: Forget (f), Input (i), Candidate (c), Output (o)
        self.Wf = np.random.randn(hidden_dim, concat_dim) * np.sqrt(2.0 / concat_dim)
        self.Wi = np.random.randn(hidden_dim, concat_dim) * np.sqrt(2.0 / concat_dim)
        self.Wc = np.random.randn(hidden_dim, concat_dim) * np.sqrt(2.0 / concat_dim)
        self.Wo = np.random.randn(hidden_dim, concat_dim) * np.sqrt(2.0 / concat_dim)

        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """
        Forward pass of a single LSTM cell.
        Inputs:
            x_t: (input_dim, 1) current input
            h_prev: (hidden_dim, 1) previous hidden state
            c_prev: (hidden_dim, 1) previous cell state
        """
        # Concat input and previous hidden state
        # Line importance: Combines spatial input with temporal memory.
        concat = np.vstack((h_prev, x_t))

        # 1. Forget Gate (how much old memory to discard)
        f_t = self.sigmoid(np.dot(self.Wf, concat) + self.bf)

        # 2. Input Gate (how much new information to store)
        i_t = self.sigmoid(np.dot(self.Wi, concat) + self.bi)

        # 3. Candidate Cell State (new candidate values)
        c_tilde_t = self.tanh(np.dot(self.Wc, concat) + self.bc)

        # 4. Update Cell State
        # Math: c_t = f_t * c_prev + i_t * c_tilde_t
        # Line importance: Linear gradient highway that prevents vanishing gradients in sequence modeling.
        c_t = f_t * c_prev + i_t * c_tilde_t

        # 5. Output Gate (what part of the cell state to output)
        o_t = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

        # 6. Hidden State (current output)
        # Math: h_t = o_t * tanh(c_t)
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t, (f_t, i_t, c_tilde_t, o_t, concat)

    def backward(self, dh_next: np.ndarray, dc_next: np.ndarray, c_prev: np.ndarray, 
                 cache: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass (Backpropagation Through Time) of a single LSTM cell.
        Inputs:
            dh_next: (hidden_dim, 1) incoming gradient from current hidden state
            dc_next: (hidden_dim, 1) incoming gradient from cell state
            c_prev: (hidden_dim, 1) previous cell state
            cache: (f_t, i_t, c_tilde_t, o_t, concat) from forward pass
        Returns:
            dx_t: gradient wrt input x_t
            dh_prev: gradient wrt previous hidden state
            dc_prev: gradient wrt previous cell state
        """
        f_t, i_t, c_tilde_t, o_t, concat = cache
        tanh_c = self.tanh(f_t * c_prev + i_t * c_tilde_t) # tanh(c_t)

        # 1. Gradient of Output Gate
        # Math: dh_next = dL/dh_t -> dL/do_t = dh_next * tanh(c_t) * o_t * (1 - o_t)
        do_t = dh_next * tanh_c * o_t * (1.0 - o_t)

        # 2. Gradient of Cell State
        # Math: dL/dc_t = dh_next * o_t * (1 - tanh^2(c_t)) + dc_next
        dc_t = dh_next * o_t * (1.0 - tanh_c ** 2) + dc_next

        # 3. Gradient of Candidate State
        # Math: dL/dc_tilde_t = dc_t * i_t * (1 - c_tilde_t^2)
        dc_tilde_t = dc_t * i_t * (1.0 - c_tilde_t ** 2)

        # 4. Gradient of Input Gate
        # Math: dL/di_t = dc_t * c_tilde_t * i_t * (1 - i_t)
        di_t = dc_t * c_tilde_t * i_t * (1.0 - i_t)

        # 5. Gradient of Forget Gate
        # Math: dL/df_t = dc_t * c_prev * f_t * (1 - f_t)
        df_t = dc_t * c_prev * f_t * (1.0 - f_t)

        # 6. Accumulate weight gradients
        self.dWf = np.dot(df_t, concat.T)
        self.dWi = np.dot(di_t, concat.T)
        self.dWc = np.dot(dc_tilde_t, concat.T)
        self.dWo = np.dot(do_t, concat.T)

        self.dbf = df_t
        self.dbi = di_t
        self.dbc = dc_tilde_t
        self.dbo = do_t

        # 7. Compute gradients wrt input and previous hidden state (via concat vector)
        dconcat = (
            np.dot(self.Wf.T, df_t) + 
            np.dot(self.Wi.T, di_t) + 
            np.dot(self.Wc.T, dc_tilde_t) + 
            np.dot(self.Wo.T, do_t)
        )

        # Split dconcat into dh_prev and dx_t
        dh_prev = dconcat[:self.hidden_dim, :]
        dx_t = dconcat[self.hidden_dim:, :]

        # 8. Gradient wrt previous cell state
        # Math: dL/dc_prev = dc_t * f_t
        dc_prev = dc_t * f_t

        return dx_t, dh_prev, dc_prev


class LSTM:
    """
    Industrial-grade sequential LSTM layer.
    Accepts 3D input batch sequences of shape (batch_size, seq_len, input_dim)
    and manages temporal state updates internally.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = LSTMCell(input_dim, hidden_dim)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass over complete sequence.
        Input: (batch_size, seq_len, input_dim)
        Output: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = X.shape
        self.X = X
        self.caches = []
        self.hs = np.zeros((batch_size, seq_len, self.hidden_dim))
        self.cs = np.zeros((batch_size, seq_len, self.hidden_dim))

        # Initialize hidden and cell states to zeros
        h_t = np.zeros((self.hidden_dim, batch_size))
        c_t = np.zeros((self.hidden_dim, batch_size))

        # Line importance: Iterate sequentially through the time dimension.
        for t in range(seq_len):
            x_t = X[:, t, :].T # Shape (input_dim, batch_size)
            h_t, c_t, cache = self.cell.forward(x_t, h_t, c_t)
            self.hs[:, t, :] = h_t.T
            self.cs[:, t, :] = c_t.T
            self.caches.append(cache)

        return self.hs

    def backward(self, dh: np.ndarray) -> np.ndarray:
        """
        Backpropagation Through Time (BPTT).
        Input: dh gradient of shape (batch_size, seq_len, hidden_dim)
        Returns: dX input gradient of shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = self.X.shape
        dX = np.zeros_like(self.X)

        dh_next = np.zeros((self.hidden_dim, batch_size))
        dc_next = np.zeros((self.hidden_dim, batch_size))

        # Line importance: Traverse backwards through time (BPTT).
        for t in reversed(range(seq_len)):
            # Accumulate gradient from upstream loss and next time step
            dh_t = dh[:, t, :].T + dh_next
            c_prev = self.cs[:, t-1, :].T if t > 0 else np.zeros((self.hidden_dim, batch_size))
            
            dx_t, dh_next, dc_next = self.cell.backward(dh_t, dc_next, c_prev, self.caches[t])
            dX[:, t, :] = dx_t.T

        return dX


# =====================================================================
# 3. TRANSFORMER SELF-ATTENTION MECHANISM
# =====================================================================

class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention mechanism used in Transformers.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        # Projection matrices for Query, Key, and Value
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.softmax = Softmax()

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Computes self-attention over input sequence X of shape (seq_len, d_model).
        """
        # Project inputs to Query, Key, and Value spaces
        # Math: Q = X . W_q, K = X . W_k, V = X . W_v
        # Line importance: Dynamic projections mapping sequences into different subspaces.
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)

        # Compute raw attention scores
        # Math: scores = Q . K^T
        scores = np.dot(Q, K.T)

        # Scale scores to prevent vanishing gradients in softmax
        # Math: scaled_scores = scores / sqrt(d_k)
        # Line importance: Scaling factor is essential for training stability.
        d_k = K.shape[1]
        scaled_scores = scores / np.sqrt(d_k)

        # Apply Softmax to get attention weights (probabilities)
        # Math: weights = softmax(scaled_scores)
        weights = self.softmax.forward(scaled_scores)

        # Compute final attention output
        # Math: output = weights . V
        # Line importance: Dynamically aggregates values based on relevance scores.
        output = np.dot(weights, V)

        return output


