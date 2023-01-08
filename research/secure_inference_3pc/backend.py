import torch
import numpy as np
from research.secure_inference_3pc.const import IS_TORCH_BACKEND
dtype_converted = {np.int32: torch.int32, np.int64: torch.int64, torch.int8:torch.int8}

class NumpyBackend:
    def __init__(self):
        self.int8 = np.int8
        self.int32 = np.int32

    def zeros(self, shape, dtype):
        return np.zeros(shape=shape, dtype=dtype)

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

    def size(self, data):
        return np.size(data)

    def astype(self, data, dtype):
        return data.astype(dtype)

    def arange(self, start, end=None, dtype=None):
        if end is None:
            end = start
            start = 0
        return np.arange(start, end, dtype=dtype)

    def unsqueeze(self, data, axis):
        if axis == 0:
            return data[np.newaxis]
        elif axis == -1:
            return data[..., np.newaxis]
        else:
            return np.expand_dims(data, axis=axis)

    def right_shift(self, data, shift):
        return data >> shift

    def bitwise_and(self, x, y, out=None):
        return np.bitwise_and(x, y, out=out)

    def cumsum(self, data, axis, out=None):
        return np.cumsum(data, axis=axis, out=data)

    def flip(self, data, axis):
        return np.flip(data, axis=axis)

class TorchBackend:
    def __init__(self):
        self.int8 = torch.int8
        self.int32 = torch.int32

    def zeros(self, shape, dtype):
        return torch.zeros(size=shape, dtype=dtype_converted[dtype])

    def concatenate(self, tensors):
        return torch.cat(tensors)

    def add(self, a, b, out):
        return torch.add(a, b, out=out)

    def multiply(self, a, b, out):
        return torch.mul(a, b, out=out)

    def subtract(self, a, b, out):
        return torch.sub(a, b, out=out)

    def array(self, data):
        return np.array(data)

    def reshape(self, data, shape):
        return torch.reshape(data, shape)

    def size(self, data):
        return torch.numel(data)

    def astype(self, data, dtype):
        return data.to(dtype_converted[dtype])

    def arange(self, start, end=None, dtype=None):
        if end is None:
            end = start
            start = 0
        if dtype is not None:
            dtype = dtype_converted[dtype]
        return torch.arange(start, end, dtype=dtype)

    def unsqueeze(self, data, axis):
        return torch.unsqueeze(data, dim=axis)

    def flip(self, data, axis):
        return torch.flip(data, dims=(axis,))

    def bitwise_and(self, x, y, out=None):
        # TODO: make inplace!
        return x & y

if IS_TORCH_BACKEND:
    backend = TorchBackend()
else:
    backend = NumpyBackend()