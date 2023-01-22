from research.secure_inference_3pc.modules.base import PRFFetcherModule, SecureModule

# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors
from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.base import get_c_party_0, P, module_67
from research.secure_inference_3pc.conv2d.conv2d_handler_factory import conv2d_handler_factory
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, NUM_OF_COMPARE_BITS, IGNORE_MSB_BITS, TRUNC_BITS
from research.secure_inference_3pc.timer import timer, Timer
from research.secure_inference_3pc.conv2d.utils import get_output_shape
from research.secure_inference_3pc.modules.base import Decompose
from research.bReLU import SecureOptimizedBlockReLU
from research.secure_inference_3pc.modules.base import DummyShapeTensor

import torch
import numpy as np

class SecureConv2DClient(SecureModule):

    def __init__(self, W_shape, stride, dilation, padding, groups, **kwargs):
        super(SecureConv2DClient, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.conv2d_handler = conv2d_handler_factory.create(self.device)

    @timer("SecureConv2DClient")
    def forward(self, X_share):
        # self.network_assets.sender_01.put(np.arange(10))
        # self.network_assets.receiver_02.get()

        assert self.W_shape[2] == self.W_shape[3]
        assert (self.W_shape[1] == X_share.shape[1]) or self.groups > 1

        with Timer("SecureConv2DClient: preprocessing..."):
            W_share = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
            A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
            B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=W_share.shape, dtype=SIGNED_DTYPE)

            with Timer("SecureConv2DClient: subtractions..."):
                E_share = backend.subtract(X_share, A_share, out=A_share)
                F_share = backend.subtract(W_share, B_share, out=B_share)

            self.network_assets.sender_01.put(E_share)
            self.network_assets.sender_01.put(F_share)

        E_share_server = self.network_assets.receiver_01.get()
        F_share_server = self.network_assets.receiver_01.get()

        # E_share_server, F_share_server = share_server[:backend.size(E_share)].reshape(E_share.shape), share_server[backend.size(E_share):].reshape(F_share.shape)

        E = backend.add(E_share_server, E_share, out=E_share)
        F = backend.add(F_share_server, F_share, out=F_share)

        out = self.conv2d_handler.conv2d(X_share,
                                         F,
                                         E,
                                         W_share,
                                         padding=self.padding,
                                         stride=self.stride,
                                         dilation=self.dilation,
                                         groups=self.groups)
        C_share = self.network_assets.receiver_02.get()

        out = backend.add(out, C_share, out=out)
        out = backend.right_shift(out, TRUNC_BITS, out=out)

        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)

        out = backend.add(out, mu_0, out=out)
        # self.network_assets.sender_01.put(np.arange(10))
        # self.network_assets.receiver_02.get()

        return out


class PrivateCompareClient(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareClient, self).__init__(**kwargs)
        self.decompose = Decompose(ignore_msb_bits=IGNORE_MSB_BITS, num_of_compare_bits=NUM_OF_COMPARE_BITS, dtype=SIGNED_DTYPE, **kwargs)

    # @timer("PrivateCompareClient")
    def forward(self, x_bits_0, r, beta):

        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=P, size=x_bits_0.shape, dtype=backend.int32)
        r[backend.astype(beta, backend.bool)] += 1
        bits = self.decompose(r)
        c_bits_0 = get_c_party_0(x_bits_0, bits, beta)
        s = backend.multiply(s, c_bits_0, out=s)
        d_bits_0 = module_67(s)

        d_bits_0 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_0, axis=-1)

        self.network_assets.sender_02.put(d_bits_0)

        return


class ShareConvertClient(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertClient, self).__init__(**kwargs)
        self.private_compare = PrivateCompareClient(**kwargs)

    # @timer("post_compare")
    def post_compare(self, a_0, eta_pp, delta_0, alpha, beta_0, mu_0):
        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_0.shape, dtype=SIGNED_DTYPE)
        eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)
        t0 = eta_pp * eta_p_0
        t1 = self.add_mode_L_minus_one(t0, t0)
        t2 = self.sub_mode_L_minus_one(eta_pp, t1)
        eta_0 = self.add_mode_L_minus_one(eta_p_0, t2)

        t0 = self.add_mode_L_minus_one(delta_0, eta_0)
        t1 = self.sub_mode_L_minus_one(t0, backend.ones_like(t0))
        t2 = self.sub_mode_L_minus_one(t1, alpha)
        theta_0 = self.add_mode_L_minus_one(beta_0, t2)

        y_0 = self.sub_mode_L_minus_one(a_0, theta_0)
        y_0 = self.add_mode_L_minus_one(y_0, mu_0)

        return y_0

    def forward(self, a_0):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=backend.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=a_0.shape, dtype=SIGNED_DTYPE)

        # with Timer("ShareConvertClient - processing..."):

        alpha = backend.astype(0 < r_0 - r, SIGNED_DTYPE)

        a_tild_0 = a_0 + r_0
        beta_0 = backend.astype(0 < a_0 - a_tild_0, SIGNED_DTYPE)
        self.network_assets.sender_02.put(a_tild_0)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS], dtype=backend.int8)
        delta_0 = self.network_assets.receiver_02.get()

        self.private_compare(x_bits_0, r - 1, eta_pp)

        return self.post_compare(a_0, eta_pp, delta_0, alpha, beta_0, mu_0)


class SecureMultiplicationClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationClient, self).__init__(**kwargs)

    # @timer("SecureMultiplicationClient - exchange shares")
    def exchange_shares(self, E_share, F_share):
        E_share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(E_share)
        F_share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        return E_share_server, F_share_server

    # @timer("SecureMultiplicationClient")
    def forward(self, X_share, Y_share):

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        C_share = self.network_assets.receiver_02.get()

        E_share = X_share - A_share
        F_share = Y_share - B_share

        E_share_server, F_share_server = self.exchange_shares(E_share, F_share)

        E = E_share_server + E_share
        F = F_share_server + F_share

        out = X_share * F + Y_share * E + C_share
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)

        return out + mu_0


class SecureSelectShareClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareClient, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationClient(**kwargs)

    def forward(self, alpha, x, y):
        # if alpha == 0: return x else return 1
        shape = alpha.shape
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        w = y - x
        c = self.secure_multiplication(alpha, w)
        z = x + c
        return z + mu_0


class SecureMSBClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBClient, self).__init__(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.private_compare = PrivateCompareClient(**kwargs)

    # @timer("SecureMSBClient - post_compare")
    def post_compare(self, beta, x_bit_0_0, r_mode_2, mu_0):
        beta = backend.astype(beta, SIGNED_DTYPE)
        beta_p_0 = self.network_assets.receiver_02.get()

        gamma_0 = beta_p_0 - (2 * beta * beta_p_0)
        delta_0 = x_bit_0_0 - (2 * r_mode_2 * x_bit_0_0)

        theta_0 = self.mult(gamma_0, delta_0)
        alpha_0 = gamma_0 + delta_0 - 2 * theta_0
        alpha_0 = alpha_0 + mu_0

        return alpha_0

    def pre_compare(self, a_0):

        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=backend.int8)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS], dtype=backend.int8)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=a_0.dtype)

        x_0 = self.network_assets.receiver_02.get()
        x_bit_0_0 = self.network_assets.receiver_02.get()
        r_1 = self.network_assets.receiver_01.get()

        y_0 = self.add_mode_L_minus_one(a_0, a_0)
        r_0 = self.add_mode_L_minus_one(x_0, y_0)
        self.network_assets.sender_01.put(r_0)
        r = self.add_mode_L_minus_one(r_0, r_1)

        r_mode_2 = r % 2

        return x_bits_0, r, beta, x_bit_0_0, r_mode_2, mu_0



    # @timer("SecureMSBClient")
    def forward(self, a_0):
        x_bits_0, r, beta, x_bit_0_0, r_mode_2, mu_0 = self.pre_compare(a_0)

        self.private_compare(x_bits_0, r, beta)

        return self.post_compare(beta, x_bit_0_0, r_mode_2, mu_0)


class SecureDReLUClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureDReLUClient, self).__init__(**kwargs)

        self.share_convert = ShareConvertClient(**kwargs)
        self.msb = SecureMSBClient(**kwargs)

    # @timer("SecureDReLUClient")
    def forward(self, X_share):

        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)

        X0_converted = self.share_convert(X_share)
        MSB_0 = self.msb(X0_converted)

        return -MSB_0+mu_0


class SecureReLUClient(SecureModule):
    def __init__(self, dummy_relu=False, **kwargs):
        super(SecureReLUClient, self).__init__(**kwargs)

        self.DReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.dummy_relu = dummy_relu

    # @timer("SecureReLUClient")
    def forward(self, X_share):
        if self.dummy_relu:
            self.network_assets.sender_01.put(X_share)
            return backend.zeros_like(X_share)
        else:

            shape = X_share.shape
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

            X_share = X_share.flatten()
            MSB_0 = self.DReLU(X_share)
            ret = self.mult(X_share, MSB_0).reshape(shape)

            return ret + mu_0


class SecureMaxPoolClient(SecureMaxPool):
    def __init__(self, kernel_size, stride, padding, dummy_max_pool, **kwargs):
        super(SecureMaxPoolClient, self).__init__(kernel_size, stride, padding, dummy_max_pool, **kwargs)
        self.select_share = SecureSelectShareClient(**kwargs)
        self.dReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)

    def forward(self, x):
        if self.dummy_max_pool:
            self.network_assets.sender_01.put(x)
            return backend.zeros_like(x[:, :, ::2, ::2])

        ret = super(SecureMaxPoolClient, self).forward(x)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)

        return ret + mu_0


class SecureBlockReLUClient(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)

    def forward(self, activation):
        return SecureOptimizedBlockReLU.forward(self, activation)

class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W_shape, stride, dilation, padding, groups, **kwargs):
        super(PRFFetcherConv2D, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, shape):
        out_shape = get_output_shape(shape, self.W_shape, self.padding, self.dilation, self.stride)

        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=SIGNED_DTYPE)

        return DummyShapeTensor(out_shape)


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherPrivateCompare, self).__init__(**kwars)

    def forward(self, shape):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=1, high=P, size=[shape[0]] + [NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS], dtype=backend.int32)


class PRFFetcherShareConvert(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherShareConvert, self).__init__(**kwars)
        self.private_compare = PRFFetcherPrivateCompare(**kwars)

    def forward(self, shape):
        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=shape, dtype=backend.int8)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(shape) + [NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS], dtype=backend.int8)

        self.private_compare(shape)

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)

        return shape


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherMultiplication, self).__init__(**kwars)

    def forward(self, shape):

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)

        return shape


class PRFFetcherSelectShare(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherSelectShare, self).__init__(**kwars)
        self.mult = PRFFetcherMultiplication(**kwars)


    def forward(self, shape):

        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.mult(shape)
        return shape



class PRFFetcherMSB(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherMSB, self).__init__(**kwars)
        self.mult = PRFFetcherMultiplication(**kwars)
        self.private_compare = PRFFetcherPrivateCompare(**kwars)

    def forward(self, shape):

        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=shape, dtype=backend.int8)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(shape) + [NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS], dtype=backend.int8)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        self.private_compare(shape)
        self.mult(shape)

        return shape


class PRFFetcherDReLU(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherDReLU, self).__init__(**kwargs)

        self.share_convert = PRFFetcherShareConvert(**kwargs)
        self.msb = PRFFetcherMSB(**kwargs)

    def forward(self, shape):

        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

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
            self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

            self.DReLU((shape[0] * shape[1] * shape[2] * shape[3], ))
            self.mult((shape[0] * shape[1] * shape[2] * shape[3], ))
        return shape


class PRFFetcherMaxPool(PRFFetcherModule):
    def __init__(self, kernel_size=3, stride=2, padding=1, dummy_max_pool=False, **kwargs):
        super(PRFFetcherMaxPool, self).__init__(**kwargs)
        self.dummy_max_pool = dummy_max_pool
        self.select_share = PRFFetcherSelectShare(**kwargs)
        self.dReLU = PRFFetcherDReLU(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)

    def forward(self, shape):
        # activation = backend.zeros(shape=shape, dtype=SIGNED_DTYPE)

        if self.dummy_max_pool:

            return DummyShapeTensor((shape[0], shape[1], shape[2] // 2, shape[3] // 2))
            return activation[:, :, ::2, ::2]
        assert False
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
        x = x.reshape((x.shape[0], -1))

        max_ = x[0]
        for i in range(1, 9):
            self.dReLU(max_.shape)
            self.select_share(max_.shape)

        ret = backend.astype(max_.reshape(out_shape), SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)

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
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, dtype=SIGNED_DTYPE, size=shape)
        out_0 = self.model.decode_head(self.model.backbone(backend.zeros(shape=img, dtype=SIGNED_DTYPE)))


class PRFFetcherSecureModelClassification(SecureModule):
    def __init__(self, model, **kwargs):
        super(PRFFetcherSecureModelClassification, self).__init__(**kwargs)
        self.model = model

    def forward(self, img):

        shape = DummyShapeTensor(img.shape)
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, dtype=SIGNED_DTYPE, size=shape)
        out = self.model.backbone(shape)[0]
        out = self.model.neck(out)
        out_0 = self.model.head.fc(out)