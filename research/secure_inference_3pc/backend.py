import torch
import numpy as np

# dtype_converted = {np.int32: torch.int32, np.int64: torch.int64}

class NumpyBackend:
    def __init__(self):
        self.int8 = np.int8
        self.int32 = np.int32

    def zeros(self, shape, dtype):
        return torch.zeros(shape=shape, dtype=dtype)

    def concatenate(self, tensors):
        return np.concatenate(tensors)

    def add(self, a, b, out):
        return np.add(a, b, out=out)

    def multiply(self, a, b, out):
        return np.multiply(a, b, out=out)

    def subtract(self, a, b, out):
        return np.subtract(a, b, out=out)

    def array(self, data):
        return np.array(data)

    def reshape(self, data, shape):
        return np.reshape(data, shape)
    # def any(self, tensor):

backend = NumpyBackend()