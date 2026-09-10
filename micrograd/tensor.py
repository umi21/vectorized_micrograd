import numpy as np

class Tensor:
    
    def __init__(self, data, _children=()):
        if isinstance(data, np.ndarray):
            self.data = data # numpy array
        else:
            self.data = np.array(data)
        self.grad = np.zeros(self.data.shape)
        self._backward = lambda : None
        self._children = _children
    
    def __repr__(self):
        return f"Tensor({self.data})"

    # the Operations include maxtix multiplication, addition, the activations, ..

    def __matmul__(self, other): # matrix multiplication A @ B
        out = Tensor(self.data @ self.grad, (self, other))

        def _backward():
            self.grad += out.grad @ self.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Tensor(self.data - other.data, (self, other))

        def _backward():
            self.grad += out.grad
            self.grad += -1 * out.grad
        out._backward = _backward
        return out

