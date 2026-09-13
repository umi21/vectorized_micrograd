import numpy as np

class Tensor:
    """ stores a vector/tensor value and its gradient """
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

    def __pow__(self, other):
        out = Tensor(self.data**other, (self,))

        def _backward():
            self.grad = other * self.data**(other-1) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))

        def _backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,))

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
        
    def __mul__(self, other):
        pass

    def tanh(self):
        pass

    def sigmoid(self):
        pass

    def sofmax(self):
        pass
