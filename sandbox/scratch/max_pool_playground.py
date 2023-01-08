import numpy as np
import torch

x = np.float32(range(112 * 112)).reshape(1, 1, 112, 112)

def select_share(alpha, x, y):
    w = y - x
    c = alpha * w
    z = x + c
    return z

class MaxPool2dNumpy(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(MaxPool2dNumpy, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        assert self.kernel_size == 3
        assert self.stride == 2
        assert self.padding == 1

    def dReLU(self, x):
        return (x > 0).astype(np.float32)

    def mult(self, x, y):
        return x * y
    def forward(self, x):
        assert x.shape[2] == 112
        assert x.shape[3] == 112

        x = np.pad(x, ((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant')
        x = np.stack([x[:, :, 0:-1:2, 0:-1:2],
                      x[:, :, 0:-1:2, 1:-1:2],
                      x[:, :, 0:-1:2, 2::2],
                      x[:, :, 1:-1:2, 0:-1:2],
                      x[:, :, 1:-1:2, 1:-1:2],
                      x[:, :, 1:-1:2, 2::2],
                      x[:, :, 2::2, 0:-1:2],
                      x[:, :, 2::2, 1:-1:2],
                      x[:, :, 2::2, 2::2]])

        out_shape = x.shape[1:]
        x = x.reshape((x.shape[0], -1))

        max_ = x[0]
        for i in range(1, 9):
            w = x[i] - max_
            beta = self.dReLU(w)
            a = self.mult(beta, x[i])
            b = self.mult((1-beta), max_)
            max_ = a + b

        return max_.reshape(out_shape)


y = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(torch.from_numpy(x)).numpy()
y2 = MaxPool2dNumpy(kernel_size=3, stride=2, padding=1)(x)

print(np.all(y2 == y))