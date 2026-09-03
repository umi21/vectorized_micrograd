import numpy as np

class Tensor:
    def __init__(self, data, _children=()):
        if isinstance(data, np.ndarray):
            self.data = data # numpy array
        else:
            self.data = np.ndarray(data)
        self.grad = np.zeros(*self.data.shape)
        self._backward = lambda : None
        self._children = _children

    
    # appart from the data every thing is operations