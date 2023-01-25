from research.secure_inference_3pc.modules.base import PRFFetcherModule
from research.secure_inference_3pc.const import  NUM_OF_COMPARE_BITS, IGNORE_MSB_BITS

import torch
from research.secure_inference_3pc.backend import backend

from research.secure_inference_3pc.timer import timer, Timer
from research.secure_inference_3pc.base import P, module_67
from research.secure_inference_3pc.conv2d.conv2d_handler_factory import conv2d_handler_factory
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE
from research.secure_inference_3pc.conv2d.utils import get_output_shape
from research.bReLU import SecureOptimizedBlockReLU
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.modules.base import Decompose
from research.secure_inference_3pc.modules.base import DummyShapeTensor

import numpy as np
import torch

from numba import njit, prange
@njit('Tuple((int8[:, :], int64[:], int64[:]))(int64[:], int64[:], int64[:], int8[:,:], uint64[:], uint64[:], uint8, uint8)', parallel=True,  nogil=True, cache=True)
def processing_numba(x, x_1, x_bit_0_0, x_bits_0, x_uint64, x_1_uint64, bits, ignore_msb_bits):
    x_bits_1 = x_bits_0
    x_0 = x_1
    x_bit_0_1 = x

    # bits = bits - 1
    for i in prange(x_bits_1.shape[0]):
        for j in range(bits - ignore_msb_bits):
            x_bit = (x[i] >> (bits - 1 - j)) & 1  # x_bits

            if x_bit >= x_bits_0[i, j]:
                x_bits_1[i][j] = x_bit - x_bits_0[i, j]
            else:
                x_bits_1[i][j] = x_bit - x_bits_0[i, j] + P

        if x_uint64[i] < x_1_uint64[i]:
            x_0[i] = x[i] - x_1[i] - 1
        else:
            x_0[i] = x[i] - x_1[i]
        x_bit0 = x[i] % 2
        x_bit_0_1[i] = x_bit0 - x_bit_0_0[i]

    return x_bits_1, x_0, x_bit_0_1

@njit('(int64[:])(int8[:,:], int8[:,:])', parallel=True,  nogil=True, cache=True)
def numba_private_compare(d_bits_0, d_bits_1):
    out = np.zeros(shape=(d_bits_0.shape[0],), dtype=np.int64)
    for i in prange(d_bits_0.shape[0]):
        for j in range(d_bits_0.shape[1]):
            a = (d_bits_0[i, j] + d_bits_1[i, j])
            if a == 0 or a == 67:
                out[i] = 1
                break

    return out


class SecureConv2DCryptoProvider(SecureModule):
    def __init__(self, W_shape, stride, dilation, padding, groups,  **kwargs):
        super(SecureConv2DCryptoProvider, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

        self.conv2d_handler = conv2d_handler_factory.create(self.device)

    def forward(self, X_share):

        # out_shape = get_output_shape(X_share.shape, self.W_shape, self.padding, self.dilation, self.stride)
        # return backend.zeros(out_shape, dtype=X_share.dtype)
        # self.network_assets.sender_02.put(np.arange(10))
        # self.network_assets.receiver_12.get()
        out = self.forward_(X_share)

        # self.network_assets.sender_02.put(np.arange(10))
        # self.network_assets.receiver_12.get()

        return out

    @timer("SecureConv2DCryptoProvider")
    def forward_(self, X_share):

        A_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        A_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)

        A = backend.add(A_share_0, A_share_1, out=A_share_0)
        B = backend.add(B_share_0, B_share_1, out=B_share_0)

        C = self.conv2d_handler.conv2d(A, B, None, None, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
        C_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=C.shape, dtype=SIGNED_DTYPE)
        C_share_0 = backend.subtract(C, C_share_1, out=C)

        self.network_assets.sender_02.put(C_share_0)

        return C_share_0


class PrivateCompareCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareCryptoProvider, self).__init__(**kwargs)

    def forward(self):
        d_bits_0 = self.network_assets.receiver_02.get()
        d_bits_1 = self.network_assets.receiver_12.get()
        beta_p = numba_private_compare(d_bits_0, d_bits_1)
        # d = backend.add(d_bits_0, d_bits_1, out=d_bits_0)
        # d = d % P
        # beta_p = backend.astype((d == 0).any(axis=-1), SIGNED_DTYPE)

        return beta_p


class ShareConvertCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertCryptoProvider, self).__init__(**kwargs)
        self.private_compare = PrivateCompareCryptoProvider(**kwargs)
        self.decompose = Decompose(ignore_msb_bits=IGNORE_MSB_BITS, num_of_compare_bits=NUM_OF_COMPARE_BITS, dtype=SIGNED_DTYPE, **kwargs)

    # @timer("ShareConvertCryptoProvider")
    def forward(self, size):
        # with Timer("ShareConvertCryptoProvider: prf... "):
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=size+(NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS,), dtype=backend.int8)
        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)

        # with Timer("ShareConvertCryptoProvider: get... "):

        a_tild_0 = self.network_assets.receiver_02.get()
        a_tild_1 = self.network_assets.receiver_12.get()

        # with Timer("ShareConvertCryptoProvider: Processing - 0 "):
        x = backend.add(a_tild_0, a_tild_1, out=a_tild_1)
        x_bits = self.decompose(x)
        x_bits_1 = backend.subtract_module(x_bits, x_bits_0, P)
        x_bits_1 = backend.astype(x_bits_1, backend.int8)

        self.network_assets.sender_12.put(x_bits_1)

        diff = backend.subtract(a_tild_0, x, out=a_tild_0)
        delta = backend.greater(diff, 0, out=diff)

        delta_0 = self.sub_mode_L_minus_one(delta, delta_1)
        self.network_assets.sender_02.put(delta_0)

        eta_p = self.private_compare()

        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        eta_p_1 = self.sub_mode_L_minus_one(eta_p, eta_p_0)

        self.network_assets.sender_12.put(eta_p_1)

        return


class SecureMultiplicationCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationCryptoProvider, self).__init__(**kwargs)

    def forward(self, shape):
        A_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        B_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        C_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        A_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        B_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1

        C_share_0 = A * B - C_share_1

        self.network_assets.sender_02.put(C_share_0)


class SecureMSBCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBCryptoProvider, self).__init__(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)
        self.private_compare = PrivateCompareCryptoProvider(**kwargs)
        self.decompose = Decompose(ignore_msb_bits=IGNORE_MSB_BITS, num_of_compare_bits=NUM_OF_COMPARE_BITS, dtype=SIGNED_DTYPE, **kwargs)

    def forward(self, size):

        x = self.prf_handler[CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=(size[0], 20), dtype=backend.int8)
        x_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        x_bit_0_0 = self.prf_handler[CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=size, dtype=SIGNED_DTYPE)

        x_bits_1, x_0, x_bit_0_1 = processing_numba(x, x_1, x_bit_0_0, x_bits_0,
                                                    x.astype(np.uint64, copy=False),
                                                    x_1.astype(np.uint64, copy=False),
                                                    NUM_OF_COMPARE_BITS,
                                                    IGNORE_MSB_BITS)

        # x_bits = self.decompose(x)
        # x_bits_1 = backend.subtract_module(x_bits, x_bits_0, P)
        # x_0 = self.sub_mode_L_minus_one(x, x_1)
        # x_bit0 = np.bitwise_and(x, 1, out=x)  # x_bit0 = x % 2
        # x_bit_0_1 = backend.subtract(x_bit0, x_bit_0_0, out=x_bit0)

        self.network_assets.sender_02.put(x_0)
        self.network_assets.sender_02.put(x_bit_0_0)

        self.network_assets.sender_12.put(x_bits_1)
        self.network_assets.sender_12.put(x_bit_0_1)

        beta_p = self.private_compare()

        beta_p_0 = self.prf_handler[CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=size, dtype=SIGNED_DTYPE)
        beta_p_1 = beta_p - beta_p_0

        self.network_assets.sender_02.put(beta_p_0)
        self.network_assets.sender_12.put(beta_p_1)

        self.mult(size)
        return


class SecureDReLUCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureDReLUCryptoProvider, self).__init__(**kwargs)

        self.share_convert = ShareConvertCryptoProvider(**kwargs)
        self.msb = SecureMSBCryptoProvider(**kwargs)

    def forward(self, X_share):

        self.share_convert(X_share.shape)
        self.msb(X_share.shape)
        return X_share


class SecureReLUCryptoProvider(SecureModule):
    def __init__(self, dummy_relu=False, **kwargs):
        super(SecureReLUCryptoProvider, self).__init__(**kwargs)

        self.DReLU = SecureDReLUCryptoProvider(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        if self.dummy_relu:
            return X_share
        else:
            orig_shape = X_share.shape
            X_share = X_share.flatten()
            X_share = self.DReLU(X_share)
            self.mult(X_share.shape)
            return X_share.reshape(orig_shape)

class SecureBlockReLUCryptoProvider(SecureModule):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        self.secure_DReLU = SecureDReLUCryptoProvider(**kwargs)
        self.secure_mult = SecureMultiplicationCryptoProvider(**kwargs)
        self.block_sizes = block_sizes
        self.dummy_relu = dummy_relu
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def forward(self, activation):
        shape = activation.shape
        if self.dummy_relu:
            return activation

        if not np.all(self.block_sizes == [0, 1]):
            mean_tensor_shape = (int(sum(np.ceil(shape[2] / block_size[0]) * np.ceil(shape[3] / block_size[1]) for block_size in self.block_sizes if 0 not in block_size)),)
            mult_shape = shape[0], sum(~self.is_identity_channels), shape[2], shape[3]

            self.secure_DReLU(torch.zeros(size=mean_tensor_shape))
            self.secure_mult(mult_shape)

        return activation

class SecureSelectShareCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareCryptoProvider, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationCryptoProvider(**kwargs)

    def forward(self, share, dummy0=None, dummy1=None):

        self.secure_multiplication(share.shape)
        return share

class SecureMaxPoolCryptoProvider(SecureMaxPool):
    def __init__(self, kernel_size, stride, padding, dummy_max_pool, **kwargs):
        super(SecureMaxPoolCryptoProvider, self).__init__(kernel_size, stride, padding, dummy_max_pool, **kwargs)
        self.select_share = SecureSelectShareCryptoProvider(**kwargs)
        self.dReLU = SecureDReLUCryptoProvider(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)

    def forward(self, x):
        if self.dummy_max_pool:
            return x[:, :, ::2, ::2]
        return super(SecureMaxPoolCryptoProvider, self).forward(x)


class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W_shape, stride, dilation, padding, groups, **kwargs):
        super(PRFFetcherConv2D, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, shape):

        out_shape = get_output_shape(shape, self.W_shape, self.padding, self.dilation, self.stride)

        # return DummyShapeTensor(out_shape)

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=SIGNED_DTYPE)

        return DummyShapeTensor(out_shape)


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherPrivateCompare, self).__init__(**kwargs)

    def forward(self, shape):
        return 


class PRFFetcherShareConvert(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherShareConvert, self).__init__(**kwargs)
        self.private_compare = PRFFetcherPrivateCompare(**kwargs)

    def forward(self, shape):
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(shape) + [NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS], dtype=backend.int8)
        # self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)

        self.private_compare(shape)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)

        return


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherMultiplication, self).__init__(**kwargs)

    def forward(self, shape):

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        return shape

class PRFFetcherSelectShare(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherSelectShare, self).__init__(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)


    def forward(self, shape):

        self.mult(shape)
        return shape

class PRFFetcherMSB(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherMSB, self).__init__(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)
        self.private_compare = PRFFetcherPrivateCompare(**kwargs)

    def forward(self, shape):

        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(shape) + [NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS], dtype=backend.int8)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.private_compare(shape)
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.mult(shape)

        return shape


class PRFFetcherDReLU(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherDReLU, self).__init__(**kwargs)

        self.share_convert = PRFFetcherShareConvert(**kwargs)
        self.msb = PRFFetcherMSB(**kwargs)

    def forward(self, shape):

        self.share_convert(shape)
        self.msb(shape)

        return shape


class PRFFetcherReLU(PRFFetcherModule):
    def __init__(self, dummy_relu=False, **kwargs):
        super(PRFFetcherReLU, self).__init__(**kwargs)

        self.DReLU = PRFFetcherDReLU(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, shape):

        if not self.dummy_relu:
            self.DReLU((shape[0] * shape[1] * shape[2] * shape[3], ))
            self.mult((shape[0] * shape[1] * shape[2] * shape[3], ))
        return shape


class PRFFetcherMaxPool(PRFFetcherModule):
    def __init__(self, kernel_size=3, stride=2, padding=1, dummy_max_pool=False, **kwargs):
        super(PRFFetcherMaxPool, self).__init__(**kwargs)

        self.select_share = PRFFetcherSelectShare(**kwargs)
        self.dReLU = PRFFetcherDReLU(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)
        self.dummy_max_pool = dummy_max_pool

    def forward(self, shape):
        # activation = backend.zeros(shape=shape, dtype=SIGNED_DTYPE)

        if self.dummy_max_pool:
            return DummyShapeTensor((shape[0], shape[1], shape[2] // 2, shape[3] // 2))

            return activation[:, :, ::2, ::2]
        assert activation.shape[2] == 112
        assert activation.shape[3] == 112

        x = backend.pad(activation, ((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant')
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
        x = backend.astype(x.reshape((x.shape[0], -1)), SIGNED_DTYPE)

        max_ = x[0]
        for i in range(1, 9):
            self.dReLU(max_.shape)
            self.select_share(max_.shape)

        ret = backend.astype(max_.reshape(out_shape), SIGNED_DTYPE)

        return ret


class PRFFetcherBlockReLU(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.secure_DReLU = PRFFetcherDReLU(**kwargs)
        self.secure_mult = PRFFetcherMultiplication(**kwargs)

        self.dummy_relu = dummy_relu

    def forward(self, shape):

        if self.dummy_relu:
            return shape

        if not np.all(self.block_sizes == [0, 1]):
            mean_tensor_shape = (int(sum(np.ceil(shape[2] / block_size[0]) * np.ceil(shape[3] / block_size[1]) for block_size in self.block_sizes if 0 not in block_size)),)
            mult_shape = shape[0], sum(~self.is_identity_channels), shape[2], shape[3]

            self.secure_DReLU(mean_tensor_shape)
            self.secure_mult(mult_shape)

        return shape

class PRFFetcherSecureModelSegmentation(SecureModule):
    def __init__(self, model,  **kwargs):
        super(PRFFetcherSecureModelSegmentation, self).__init__( **kwargs)
        self.model = model

    def forward(self, img):
        shape = img.shape
        assert False
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        out_0 = self.model.decode_head(self.model.backbone(backend.zeros(shape=img.shape, dtype=SIGNED_DTYPE)))


class PRFFetcherSecureModelClassification(SecureModule):
    def __init__(self, model,  **kwargs):
        super(PRFFetcherSecureModelClassification, self).__init__( **kwargs)
        self.model = model

    def forward(self, img):
        shape = DummyShapeTensor(img.shape)
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        out = self.model.backbone(shape)[0]
        out = self.model.neck(out)
        out_0 = self.model.head.fc(out)


