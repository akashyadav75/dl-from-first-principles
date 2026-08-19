"""
scratchml/autograd.py

A lightweight, fully functional Automatic Differentiation (Autograd) Engine built from scratch using NumPy.
This module tracks operations on modern `Tensor` objects, dynamically constructs a Directed Acyclic Graph (DAG),
and performs a Topological Sort to calculate backward gradients automatically using the chain rule.

Mathematical Concepts Covered:
1. Computation Graph: A DAG where nodes are Tensors and edges are mathematical operations.
2. Chain Rule: dL/dx = dL/dy * dy/dx
3. Topological Sort: Ordering nodes such that for every directed edge u -> v, u comes before v.
   Ensures that gradients are fully accumulated before backpropagating further.

Generally Used Library: PyTorch Autograd (torch.Tensor, Tensor.backward()).
"""

import numpy as np
from typing import List, Tuple, Union, Set

class Tensor:
    """
    A Tensor represents a node in the computation graph.
    It wraps a NumPy array and tracks gradients, operations, and parent nodes.
    """
    def __init__(self, data: Union[int, float, list, np.ndarray], creators: List['Tensor'] = None, op: str = ""):
        self.data = np.array(data, dtype=float)
        self.grad: np.ndarray = np.zeros_like(self.data)
        self.creators = creators if creators is not None else []
        self.op = op
        self._backward_fn = lambda: None

    def backward(self):
        """
        Executes backpropagation starting from this Tensor.
        Uses topological sorting to ensure correct gradient accumulation order.
        """
        # Line importance: Topological sort of the computation graph.
        topo: List['Tensor'] = []
        visited: Set['Tensor'] = set()

        def build_topo(v: 'Tensor'):
            if v not in visited:
                visited.add(v)
                for creator in v.creators:
                    build_topo(creator)
                topo.append(v)

        build_topo(self)

        # Initialize the gradient of the root node (usually the loss) to 1.0
        self.grad = np.ones_like(self.data)

        # Line importance: Backpropagate gradients in reverse topological order.
        for node in reversed(topo):
            node._backward_fn()

    # =================================================================
    # MATHEMATICAL OPERATIONS
    # =================================================================

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, creators=[self, other], op="+")

        def _backward():
            # Math: d/dx(x + y) = 1, d/dy(x + y) = 1
            # Line importance: Accumulate gradient (+=) to handle multiple paths/shared nodes correctly.
            self.grad += out.grad
            other.grad += out.grad

        out._backward_fn = _backward
        return out

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, creators=[self, other], op="-")

        def _backward():
            # Math: d/dx(x - y) = 1, d/dy(x - y) = -1
            self.grad += out.grad
            other.grad -= out.grad

        out._backward_fn = _backward
        return out

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, creators=[self, other], op="*")

        def _backward():
            # Math: d/dx(x * y) = y, d/dy(x * y) = x (using chain rule)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward_fn = _backward
        return out

    def dot(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix dot product operation.
        """
        out = Tensor(np.dot(self.data, other.data), creators=[self, other], op="dot")

        def _backward():
            # Math: d/dX (X . W) = dL/dY . W^T
            # Math: d/dW (X . W) = X^T . dL/dY
            self.grad += np.dot(out.grad, other.data.T)
            other.grad += np.dot(self.data.T, out.grad)

        out._backward_fn = _backward
        return out

    def sum(() -> 'Tensor'):
        """
        Summation operation along all dimensions.
        """
        pass

    def __repr__(self):
        return f"Tensor({self.data.tolist()}, op='{self.op}')"


# Sum implementation
def tensor_sum(tensor: Tensor) -> Tensor:
    out = Tensor(np.sum(tensor.data), creators=[tensor], op="sum")

    def _backward():
        # Math: d/dx(sum(x)) = 1
        tensor.grad += np.ones_like(tensor.data) * out.grad

    out._backward_fn = _backward
    return out

# Bind sum method to Tensor class
Tensor.sum = lambda self: tensor_sum(self)


# =====================================================================
# STUDENT EXAMPLE & VERIFICATION
# =====================================================================
if __name__ == "__main__":
    print("--- Running Autograd Engine Student Example ---")
    
    # Let's compute: loss = (x * w + b) - y
    # and backpropagate to find d_loss/dw, d_loss/db
    x = Tensor([[2.0, 3.0]])
    w = Tensor([[1.0], [4.0]])
    b = Tensor([[0.5]])
    y = Tensor([[14.0]])

    # Forward pass
    xw = x.dot(w)          # [[14.0]]
    pred = xw + b          # [[14.5]]
    loss = pred - y        # [[0.5]]

    # Backward pass
    loss.backward()

    print(f"Prediction: {pred.data}")
    print(f"Loss: {loss.data}")
    print(f"Gradient wrt weights (dw):\n{w.grad}") # Expected: [[2.0], [3.0]]
    print(f"Gradient wrt bias (db):\n{b.grad}")    # Expected: [[1.0]]
