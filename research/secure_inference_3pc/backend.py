import torch
import numpy as np
from research.secure_inference_3pc.const import IS_TORCH_BACKEND
dtype_converted = {np.int32: torch.int32, np.int64: torch.int64, torch.int8:torch.int8, torch.bool:torch.bool, torch.int32:torch.int32, torch.int64:torch.int64}
torch_dtype_converted = {torch.int32: np.int32, torch.int64: np.int64, torch.int8:np.int8, torch.bool:np.bool, np.int32:np.int32, np.int64:np.int64, np.int8:np.int8, np.bool:np.bool, None:None}

# TODO: make repeat a more generic function

class NumpyBackend:
    def __init__(self):
        self.int8 = np.int8
        self.int32 = np.int32
        self.bool = np.bool

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
        return np.arange(start, end, dtype=torch_dtype_converted[dtype])

    def unsqueeze(self, data, axis):
        if axis == 0:
            return data[np.newaxis]
        elif axis == -1:
            return data[..., np.newaxis]
        else:
            return np.expand_dims(data, axis=axis)

    def right_shift(self, data, shift, out=None):
        return np.right_shift(data, shift, out=out)

    def bitwise_and(self, x, y, out=None):
        return np.bitwise_and(x, y, out=out)

    def cumsum(self, data, axis, out=None):
        return np.cumsum(data, axis=axis, out=data)

    def flip(self, data, axis):
        return np.flip(data, axis=axis)

    def unsigned_gt(self, a, b):
        return a.astype(np.uint64, copy=False) > b.astype(np.uint64, copy=False)

    def pad(self, data, pad_width, mode):
        return np.pad(data, pad_width, mode=mode, constant_values=0)

    def stack(self, data, axis=0):
        return np.stack(data, axis=axis)

    def mean(self, data, axis, keepdims=False):
        size = sum(data.shape[i] for i in axis)
        return np.sum(data, axis=axis, keepdims=keepdims) // size

    def sum(self, data, axis, keepdims=False):
        out = np.sum(data, axis=axis, keepdims=keepdims)
        return out

    def ones_like(self, data):
        return np.ones_like(data)

    def zeros_like(self, data):
        return np.zeros_like(data)

    def repeat(self, data, repeats):
        # TODO: add axis
        return data.repeat(repeats, axis=-1)

    def permute(self, data, order):
        return data.transpose(order)

class TorchBackend:
    def __init__(self):
        self.int8 = torch.int8
        self.int32 = torch.int32
        self.bool = torch.bool
        self.numpy_backend = NumpyBackend()

    def zeros(self, shape, dtype):
        out = torch.zeros(size=shape, dtype=dtype_converted[dtype])
        assert (self.numpy_backend.zeros(shape, torch_dtype_converted[dtype]) == out.numpy()).all()
        return out

    def concatenate(self, tensors):
        out = torch.cat(tensors)
        assert (self.numpy_backend.concatenate([tensor.numpy() for tensor in tensors]) == out.numpy()).all()
        return out

    def add(self, a, b, out):
        out = torch.add(a, b, out=out)
        return out

    def multiply(self, a, b, out):
        return torch.mul(a, b, out=out)

    def subtract(self, a, b, out):
        return torch.sub(a, b, out=out)

    def array(self, data):
        return np.array(data)

    def reshape(self, data, shape):
        out = torch.reshape(data, shape)
        assert (self.numpy_backend.reshape(data.numpy(), shape) == out.numpy()).all()
        return out

    def size(self, data):
        out = torch.numel(data)
        assert (self.numpy_backend.size(data.numpy()) == out)
        return out

    def astype(self, data, dtype):
        out = data.to(dtype_converted[dtype])
        assert (self.numpy_backend.astype(data.numpy(), torch_dtype_converted[dtype]) == out.numpy()).all()
        return out

    def arange(self, start, end=None, dtype=None):
        if end is None:
            end = start
            start = 0
        if dtype is not None:
            dtype = dtype_converted[dtype]
        out = torch.arange(start, end, dtype=dtype)
        assert (self.numpy_backend.arange(start, end, dtype) == out.numpy()).all()
        return out

    def unsqueeze(self, data, axis):
        out = torch.unsqueeze(data, dim=axis)
        assert (self.numpy_backend.unsqueeze(data.numpy(), axis) == out.numpy()).all()
        return out

    def cumsum(self, data, axis, out=None):
        out = torch.cumsum(data, dim=axis, out=out)
        assert (self.numpy_backend.cumsum(data.numpy(), axis, out) == out.numpy()).all()
        return out

    def right_shift(self, data, shift, out=None):

        if type(shift) is int:
            return torch.from_numpy(data.numpy() >> shift)
        else:
            return torch.from_numpy(data.numpy() >> shift.numpy())

    def flip(self, data, axis):

        out = torch.flip(data, dims=(axis,))
        assert (self.numpy_backend.flip(data.numpy(), axis) == out.numpy()).all()
        return out

    def bitwise_and(self, x, y, out=None):
        # TODO: make inplace!
        out = x & y
        assert (self.numpy_backend.bitwise_and(x.numpy(), y, None) == out.numpy()).all()
        return out

    def unsigned_gt(self, a, b):
        out = a > b
        out[(a < 0) & (b >= 0)] = True
        out[(a >= 0) & (b < 0)] = False

        assert (self.numpy_backend.unsigned_gt(a.numpy(), b.numpy()) == out.numpy()).all()

        return out

    def pad(self, data, pad, mode='constant', value=0):
        assert pad[0][0] == 0 and pad[0][1] == 0
        assert pad[1][0] == 0 and pad[1][1] == 0
        out = torch.nn.functional.pad(data, [pad[3][0], pad[3][1], pad[2][0], pad[2][1]], mode=mode, value=value)
        assert (self.numpy_backend.pad(data.numpy(), pad, mode=mode) == out.numpy()).all()
        return out

    def stack(self, data, axis=0):
        out = torch.stack(data, dim=axis)
        assert (self.numpy_backend.stack([tensor.numpy() for tensor in data], axis=axis) == out.numpy()).all()
        return out

    def mean(self, data, axis, keepdims=False):
        size = sum(data.shape[i] for i in axis)
        out = torch.sum(data, dim=axis, keepdim=keepdims) // size
        assert np.abs(self.numpy_backend.mean(data.numpy(), axis, keepdims) - out.numpy()).max() <= 1
        return out

    def sum(self, data, axis, keepdims=False):
        out = torch.sum(data, dim=axis, keepdim=keepdims)
        return out

    def ones_like(self, data):
        out = torch.ones_like(data)
        assert (self.numpy_backend.ones_like(data.numpy()) == out.numpy()).all()
        return out

    def zeros_like(self, data):
        out = torch.zeros_like(data)
        assert (self.numpy_backend.zeros_like(data.numpy()) == out.numpy()).all()
        return out

    def repeat(self, data, repeats):
        return data.repeat((1, 1, 1, 1, repeats))

    def permute(self, data, order):
        return data.permute(order)
if IS_TORCH_BACKEND:
    backend = TorchBackend()
else:
    backend = NumpyBackend()