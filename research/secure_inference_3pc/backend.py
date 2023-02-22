import torch
import numpy as np
from research.secure_inference_3pc.const import IS_TORCH_BACKEND
from research.secure_inference_3pc.timer import timer
dtype_converted = {np.int32: torch.int32, np.int64: torch.int64, torch.int8:torch.int8, torch.bool:torch.bool, torch.int32:torch.int32, torch.int64:torch.int64}
torch_dtype_converted = {torch.int32: np.int32, torch.int64: np.int64, torch.int8:np.int8, torch.bool:np.bool, np.int32:np.int32, np.int64:np.int64, np.int8:np.int8, np.bool:np.bool, None:None}
signed_dtype_to_unsigned = {np.dtype("int8"): np.uint8, np.dtype("int16").dtype: np.uint16, np.dtype("int32"): np.uint32, np.dtype("int64"): np.uint64}
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
        dtype = signed_dtype_to_unsigned[a.dtype]
        return a.astype(dtype, copy=False) > b.astype(dtype, copy=False)

    def pad(self, data, pad_width, mode):
        return np.pad(data, pad_width, mode=mode, constant_values=0)

    def stack(self, data, axis=0):
        return np.stack(data, axis=axis)

    def mean(self, data, axis, keepdims=False, dtype=None):
        size = sum(data.shape[i] for i in axis)
        return np.sum(data, axis=axis, keepdims=keepdims, dtype=dtype) // size

    def sum(self, data, axis, keepdims=False):
        out = np.sum(data, axis=axis, keepdims=keepdims, dtype=data.dtype)
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

    def put_on_device(self, data, device):
        assert device == "cpu"
        return data

    def subtract_module(self, x, y, P):
        ret = self.subtract(x, y, out=x)
        ret[ret < 0] += P
        return ret

    def greater(self, x, y, out=None):
        return np.greater(x, y, out=out)

    def equal(self, x, y, out=None):
        return np.equal(x, y, out=out)

    def any(self, x, axis=None, out=None):
        return np.any(x, axis=axis, out=out)

class TorchBackend:
    def __init__(self):
        self.int8 = torch.int8
        self.int32 = torch.int32
        self.bool = torch.bool
        self.numpy_backend = NumpyBackend()

    def zeros(self, shape, dtype):
        out = torch.zeros(size=shape, dtype=dtype_converted[dtype])
        return out

    def concatenate(self, tensors):
        out = torch.cat(tensors)
        return out

    def add(self, a, b, out):
        out = torch.add(a, b, out=out)
        return out

    def multiply(self, a, b, out):
        return torch.mul(a, b, out=out)

    def subtract(self, a, b, out):
        return torch.sub(a, b, out=out)

    def array(self, data):
        if type(data) is torch.Tensor:
            return np.array(data.cpu())
        elif type(data) is list:
            if type(data[0]) is torch.Tensor:
                data = [d.cpu() for d in data]
            return np.array(data)
        elif type(data) is np.ndarray:
            return torch.from_numpy(data)
        else:
            raise ValueError("Unsupported type")

    def reshape(self, data, shape):
        out = torch.reshape(data, shape)
        return out

    def size(self, data):
        out = torch.numel(data)
        return out

    def astype(self, data, dtype):
        out = data.to(dtype_converted[dtype])
        return out

    def arange(self, start, end=None, dtype=None):
        if end is None:
            end = start
            start = 0
        if dtype is not None:
            dtype = dtype_converted[dtype]
        out = torch.arange(start, end, dtype=dtype)
        return out

    def unsqueeze(self, data, axis):
        out = torch.unsqueeze(data, dim=axis)
        return out

    def cumsum(self, data, axis, out=None):
        out = torch.cumsum(data, dim=axis, out=out)
        return out

    def right_shift(self, data, shift, out=None):
        # TODO: is there a more efficient wat?
        sign = data.sign()
        shifted = data.abs() >> shift
        out = self.multiply(sign, shifted, out=shifted)
        sign = self.subtract(sign, 1, out=sign)
        sign = sign // 2
        out = self.add(out, sign, out=out)
        return out


    def flip(self, data, axis):

        out = torch.flip(data, dims=(axis,))
        return out

    def bitwise_and(self, x, y, out=None):
        # TODO: make inplace!
        out = x & y
        return out

    def unsigned_gt(self, a, b):
        # out = a > b
        # out = torch.bitwise_xor(out, b > 0, out=out)
        # out = torch.bitwise_xor(out, a > 0, out=out)
        # return out
        return (a > 0) ^ (b > 0) ^ (a > b)
        #
    def pad(self, data, pad, mode='constant', value=0):
        assert pad[0][0] == 0 and pad[0][1] == 0
        assert pad[1][0] == 0 and pad[1][1] == 0
        out = torch.nn.functional.pad(data, [pad[3][0], pad[3][1], pad[2][0], pad[2][1]], mode=mode, value=value)
        return out

    def stack(self, data, axis=0):
        out = torch.stack(data, dim=axis)
        return out

    def mean(self, data, axis, keepdims=False, dtype=None):
        size = sum(data.shape[i] for i in axis)
        out = torch.sum(data, dim=axis, keepdim=keepdims, dtype=dtype) // size
        return out

    def sum(self, data, axis, keepdims=False):
        out = torch.sum(data, dim=axis, keepdim=keepdims, dtype=data.dtype)
        return out

    def ones_like(self, data):
        out = torch.ones_like(data)
        return out

    def zeros_like(self, data):
        out = torch.zeros_like(data)
        return out

    def repeat(self, data, repeats):
        return data.repeat((1, 1, 1, 1, repeats))

    def permute(self, data, order):
        return data.permute(order)

    def put_on_device(self, data, device):
        return data.to(device)

    def subtract_module(self, x, y, P):
        ret = self.subtract(x, y, out=x)
        ret[ret < 0] += P
        return ret

    def greater(self, x, y, out=None):
        return torch.gt(x, y, out=out)

    def equal(self, x, y, out=None):
        return torch.eq(x, y, out=out)

    def any(self, x, axis=None, out=None):
        r = x.to(torch.bool)
        out[:] = torch.any(r, dim=axis, out=r)
        return out

if IS_TORCH_BACKEND:
    backend = TorchBackend()
else:
    backend = NumpyBackend()