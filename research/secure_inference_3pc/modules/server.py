import numpy as np
import torch

from research.secure_inference_3pc.modules.base import PRFFetcherModule, SecureModule
from research.secure_inference_3pc.modules.conv2d import get_output_shape
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, P, MIN_VAL, MAX_VAL, SIGNED_DTYPE, NUM_BITS, NUM_OF_COMPARE_BITS
from research.secure_inference_3pc.timer import Timer
from research.bReLU import NumpySecureOptimizedBlockReLU

# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors
class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W, bias, stride, dilation, padding, groups, crypto_assets, network_assets, device="cpu"):
        super(PRFFetcherConv2D, self).__init__(crypto_assets, network_assets)

        self.W_shape = W.shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)

        out_shape = get_output_shape(X_share, self.W_shape, self.padding, self.dilation, self.stride)

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=X_share.dtype)

        return np.zeros(shape=out_shape, dtype=X_share.dtype)


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherPrivateCompare, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_0):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=1, high=67, size=[x_bits_0.shape[0]] + [NUM_OF_COMPARE_BITS], dtype=np.int32)


class PRFFetcherShareConvert(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherShareConvert, self).__init__(crypto_assets, network_assets)
        self.private_compare = PRFFetcherPrivateCompare(crypto_assets, network_assets)

    def forward(self, dummy_tensor):
        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=dummy_tensor.shape, dtype=np.int8)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)

        self.private_compare(dummy_tensor)

        return dummy_tensor


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherMultiplication, self).__init__(crypto_assets, network_assets)

    def forward(self, dummy_tensor):

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=dummy_tensor.dtype)

        return dummy_tensor

class PRFFetcherSelectShare(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherSelectShare, self).__init__(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)


    def forward(self, dummy_tensor):
        dtype = dummy_tensor.dtype
        shape = dummy_tensor.shape

        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=dtype)
        self.mult(dummy_tensor)
        return dummy_tensor


class PRFFetcherMSB(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherMSB, self).__init__(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)
        self.private_compare = PRFFetcherPrivateCompare(crypto_assets, network_assets)

    def forward(self, dummy_tensor):

        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=dummy_tensor.shape, dtype=np.int8)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=dummy_tensor.dtype)

        self.private_compare(dummy_tensor)
        self.mult(dummy_tensor)

        return dummy_tensor


class PRFFetcherDReLU(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherDReLU, self).__init__(crypto_assets, network_assets)

        self.share_convert = PRFFetcherShareConvert(crypto_assets, network_assets)
        self.msb = PRFFetcherMSB(crypto_assets, network_assets)

    def forward(self, dummy_tensor):
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=dummy_tensor.dtype)

        self.share_convert(dummy_tensor)
        self.msb(dummy_tensor)

        return dummy_tensor


class PRFFetcherReLU(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets, dummy_relu=False):
        super(PRFFetcherReLU, self).__init__(crypto_assets, network_assets)

        self.DReLU = PRFFetcherDReLU(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)
        self.dummy_relu = dummy_relu

    def forward(self, dummy_tensor):
        if self.dummy_relu:
            return dummy_tensor
        else:

            dummy_numpy = dummy_tensor
            dtype = dummy_numpy.dtype
            shape = dummy_numpy.shape
            self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=dtype)

            dummy_arr = dummy_numpy.flatten()
            self.DReLU(dummy_arr)
            self.mult(dummy_arr)
            return dummy_tensor

class PRFFetcherMaxPool(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets, kernel_size=3, stride=2, padding=1, dummy_max_pool=False):
        super(PRFFetcherMaxPool, self).__init__(crypto_assets, network_assets)

        self.dummy_max_pool = dummy_max_pool
        self.select_share = PRFFetcherSelectShare(crypto_assets, network_assets)
        self.dReLU = PRFFetcherDReLU(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)

    def forward(self, x):
        if self.dummy_max_pool:
            return x[:,:,::2, ::2]
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
            self.dReLU(max_)
            self.select_share(max_)

        ret = max_.reshape(out_shape).astype(SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)

        return ret

class PRFFetcherBlockReLU(SecureModule, NumpySecureOptimizedBlockReLU):
    def __init__(self, block_sizes, crypto_assets, network_assets, dummy_relu=False):
        SecureModule.__init__(self, crypto_assets=crypto_assets, network_assets=network_assets)
        NumpySecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.secure_DReLU = PRFFetcherDReLU(crypto_assets, network_assets)
        self.secure_mult = PRFFetcherMultiplication(crypto_assets, network_assets)

        self.dummy_relu = dummy_relu

    def mult(self, x, y):
        return self.secure_mult(x)

    def DReLU(self, activation):
        return self.secure_DReLU(activation)

    def forward(self, activation):
        if self.dummy_relu:
            return torch.zeros_like(activation)

        activation = NumpySecureOptimizedBlockReLU.forward(self, activation)
        activation = activation.astype(SIGNED_DTYPE)

        return activation

class PRFFetcherSecureModelSegmentation(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(PRFFetcherSecureModelSegmentation, self).__init__(crypto_assets, network_assets)
        self.model = model

    def forward(self, img):

        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=img.shape, dtype=SIGNED_DTYPE)
        out_0 = self.model.decode_head(self.model.backbone(np.zeros(shape=img.shape, dtype=SIGNED_DTYPE)))


class PRFFetcherSecureModelClassification(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(PRFFetcherSecureModelClassification, self).__init__( crypto_assets, network_assets)
        self.model = model

    def forward(self, img):

        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, dtype=SIGNED_DTYPE, size=img.shape)
        out = self.model.backbone(np.zeros(shape=img.shape, dtype=SIGNED_DTYPE))[0]
        out = self.model.neck(out)
        out_0 = self.model.head.fc(out)