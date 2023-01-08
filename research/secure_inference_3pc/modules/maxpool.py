from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.modules.base import SecureModule


class SecureMaxPool(SecureModule):
    def __init__(self, kernel_size, stride, padding, dummy_max_pool, **kwargs):
        super(SecureMaxPool, self).__init__(**kwargs)
        self.dummy_max_pool = dummy_max_pool
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        assert self.kernel_size == 3
        assert self.stride == 2
        assert self.padding == 1

    def forward(self, x):

        assert x.shape[2] == 112, x.shape
        assert x.shape[3] == 112, x.shape

        x = backend.pad(x, ((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant')
        x = backend.stack([x[:, :, 0:-1:2, 0:-1:2],
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
            max_ = self.select_share(beta, max_, x[i])

        ret = max_.reshape(out_shape)
        return ret
