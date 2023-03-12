from research.secure_inference_3pc.backend import backend
import torch
import numpy as np

class SecureMaxPool:
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        assert self.kernel_size == 3
        assert self.stride == 2
        assert self.padding == 1

    def foo(self, x):
        out_shape = (x.shape[2] + 1) // 2, (x.shape[3] + 1) // 2
        x = backend.pad(x, ((0, 0), (0, 0), (1, 2), (1, 2)), mode='edge')

        x = backend.stack([
            x[:, :, 0::2, 0::2][:, :, :out_shape[0], :out_shape[1]],
            x[:, :, 0::2, 1::2][:, :, :out_shape[0], :out_shape[1]],
            x[:, :, 0::2, 2::2][:, :, :out_shape[0], :out_shape[1]],
            x[:, :, 1::2, 0::2][:, :, :out_shape[0], :out_shape[1]],
            x[:, :, 1::2, 1::2][:, :, :out_shape[0], :out_shape[1]],
            x[:, :, 1::2, 2::2][:, :, :out_shape[0], :out_shape[1]],
            x[:, :, 2::2, 0::2][:, :, :out_shape[0], :out_shape[1]],
            x[:, :, 2::2, 1::2][:, :, :out_shape[0], :out_shape[1]],
            x[:, :, 2::2, 2::2][:, :, :out_shape[0], :out_shape[1]]])

        out_shape = x.shape[1:]
        x = x.reshape((x.shape[0], -1))

        max_ = x[0]
        for i in range(1, 9):
            w = x[i] - max_
            alpha = w > 0
            max_ = (1 - alpha) * max_ + (alpha) * x[i]

        ret = max_.reshape(out_shape)
        return ret


first_dim = 119
sec_dim = 112
tensor = torch.randn(1, 3, first_dim, sec_dim)
out_0 = torch.max_pool2d(tensor, 3, 2, 1).numpy()
print(out_0.shape[2:])
print((first_dim+1)//2, (sec_dim+1)//2)
out_1 = SecureMaxPool(3, 2, 1).foo(tensor.numpy())

print(np.abs(out_0 - out_1).mean())