from research.secure_inference_3pc.const import COMPARISON_NUM_BITS_IGNORED
from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.conv2d.conv2d_handler_factory import conv2d_handler_factory
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, \
    UNSIGNED_DTYPE, P
from research.secure_inference_3pc.conv2d.utils import get_output_shape
from research.bReLU import SecureOptimizedBlockReLU
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.modules.base import Decompose
from research.secure_inference_3pc.parties.crypto_provider.numba_methods import numba_private_compare, processing_numba
import numpy as np


class SecureConv2DCryptoProvider(SecureModule):
    def __init__(self, W_shape, stride, dilation, padding, groups, **kwargs):
        super(SecureConv2DCryptoProvider, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

        self.conv2d_handler = conv2d_handler_factory.create(self.device)
        self.is_dummy = False

    # @timer(name='provider_conv2d')
    def forward(self, X_share):
        if self.is_dummy:
            out_shape = get_output_shape(X_share.shape, self.W_shape, self.padding, self.dilation, self.stride)
            return backend.zeros(out_shape, dtype=X_share.dtype)
        # self.network_assets.sender_02.put(np.arange(10))
        # self.network_assets.receiver_12.get()
        out = self.forward_(X_share)

        # self.network_assets.sender_02.put(np.arange(10))
        # self.network_assets.receiver_12.get()

        return out

    def forward_(self, X_share):
        A_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape,
                                                                       dtype=SIGNED_DTYPE)
        B_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_shape,
                                                                       dtype=SIGNED_DTYPE)
        A_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape,
                                                                       dtype=SIGNED_DTYPE)
        B_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_shape,
                                                                       dtype=SIGNED_DTYPE)

        A = backend.add(A_share_0, A_share_1, out=A_share_0)
        B = backend.add(B_share_0, B_share_1, out=B_share_0)

        C = self.conv2d_handler.conv2d(A, B, None, None, padding=self.padding, stride=self.stride,
                                       dilation=self.dilation, groups=self.groups)
        C_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=C.shape,
                                                                       dtype=SIGNED_DTYPE)
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
        self.decompose = Decompose(num_bits_ignored=COMPARISON_NUM_BITS_IGNORED, dtype=SIGNED_DTYPE, **kwargs)

    def forward(self, size):
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=size + (
        64 - COMPARISON_NUM_BITS_IGNORED,), dtype=backend.int8)
        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)

        a_tild_0 = self.network_assets.receiver_02.get()
        a_tild_1 = self.network_assets.receiver_12.get()

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


from torch.nn import Module


class DepthToSpace(Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W, _ = x.shape
        x = x.reshape(N, C, H, W, self.block_size[0], self.block_size[1])
        x = backend.permute(x, (0, 1, 2, 4, 3, 5))
        x = x.reshape(N, C, H * self.block_size[0], W * self.block_size[1])
        return x


class SecurePostBReLUMultCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecurePostBReLUMultCryptoProvider, self).__init__(**kwargs)

    def post(self, activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes,
             active_block_sizes_to_channels):
        relu_map = backend.ones_like(activation)
        for i, block_size in enumerate(active_block_sizes):
            orig_shape = (1, active_block_sizes_to_channels[i].shape[0], pad_handlers[i].out_shape[0] // block_size[0],
                          pad_handlers[i].out_shape[1] // block_size[1], 1)
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shape)
            tensor = backend.repeat(sign_tensor, block_size[0] * block_size[1])
            cur_channels = active_block_sizes_to_channels[i]
            relu_map[:, cur_channels] = pad_handlers[i].unpad(DepthToSpace(active_block_sizes[i])(tensor))
        return relu_map

    def forward(self, activation, sign_tensors, cumsum_shapes, pad_handlers, is_identity_channels, active_block_sizes,
                active_block_sizes_to_channels):
        non_identity_activation = activation[:, ~is_identity_channels]

        A_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1,
                                                                       size=non_identity_activation.shape,
                                                                       dtype=SIGNED_DTYPE)
        B_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=sign_tensors.shape,
                                                                       dtype=SIGNED_DTYPE)
        C_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1,
                                                                       size=non_identity_activation.shape,
                                                                       dtype=SIGNED_DTYPE)
        A_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1,
                                                                       size=non_identity_activation.shape,
                                                                       dtype=SIGNED_DTYPE)
        B_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=sign_tensors.shape,
                                                                       dtype=SIGNED_DTYPE)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1

        B = self.post(activation, B, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels)[:,
            ~is_identity_channels]

        C_share_0 = A * B - C_share_1

        self.network_assets.sender_02.put(C_share_0)

        return


class SecureMultiplicationCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationCryptoProvider, self).__init__(**kwargs)

    def forward(self, shape):
        A_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape,
                                                                       dtype=SIGNED_DTYPE)
        B_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape,
                                                                       dtype=SIGNED_DTYPE)
        C_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape,
                                                                       dtype=SIGNED_DTYPE)
        A_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape,
                                                                       dtype=SIGNED_DTYPE)
        B_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape,
                                                                       dtype=SIGNED_DTYPE)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1

        C_share_0 = A * B - C_share_1

        self.network_assets.sender_02.put(C_share_0)


class SecureMSBCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBCryptoProvider, self).__init__(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)
        self.private_compare = PrivateCompareCryptoProvider(**kwargs)


    def forward(self, size):
        x = self.prf_handler[CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=(
        size[0], 64 - COMPARISON_NUM_BITS_IGNORED), dtype=backend.int8)
        x_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        x_bit_0_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=size,
                                                                       dtype=SIGNED_DTYPE)
        beta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=size,
                                                                      dtype=SIGNED_DTYPE)

        processing_numba(x, x_1, x_bit_0_0, x_bits_0,
                         x.astype(UNSIGNED_DTYPE, copy=False),
                         x_1.astype(UNSIGNED_DTYPE, copy=False),
                         COMPARISON_NUM_BITS_IGNORED)
        x_bits_1 = x_bits_0
        x_0 = x_1
        x_bit_0_1 = x
        # x_bits = self.decompose(x)
        # x_bits_1 = backend.subtract_module(x_bits, x_bits_0, P)
        # x_0 = self.sub_mode_L_minus_one(x, x_1)
        # x_bit0 = np.bitwise_and(x, 1, out=x)  # x_bit0 = x % 2
        # x_bit_0_1 = backend.subtract(x_bit0, x_bit_0_0, out=x_bit0)

        self.network_assets.sender_02.put(x_0)
        # self.network_assets.sender_02.put(x_bit_0_0)

        self.network_assets.sender_12.put(x_bits_1)
        self.network_assets.sender_12.put(x_bit_0_1)

        beta_p = self.private_compare()

        beta_p_1 = beta_p - beta_p_0

        # self.network_assets.sender_02.put(beta_p_0)
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
        # return X_share
        if self.dummy_relu:
            return X_share
        else:
            orig_shape = X_share.shape
            X_share = X_share.flatten()
            X_share = self.DReLU(X_share)
            self.mult(X_share.shape)
            return X_share.reshape(orig_shape)


class SecureBlockReLUCryptoProvider(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)

        self.secure_DReLU = SecureDReLUCryptoProvider(**kwargs)
        self.secure_mult = SecureMultiplicationCryptoProvider(**kwargs)
        self.block_sizes = block_sizes
        self.dummy_relu = dummy_relu
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])
        self.DReLU = SecureDReLUCryptoProvider(**kwargs)

        self.post_bReLU = SecurePostBReLUMultCryptoProvider(**kwargs)

    def forward(self, activation):
        if self.dummy_relu:
            return activation
        SecureOptimizedBlockReLU.forward(self, activation)
        return activation
        #
        #
        # shape = activation.shape
        # if self.dummy_relu:
        #     return activation
        #
        # if not np.all(self.block_sizes == [0, 1]):
        #     mean_tensor_shape = (int(sum(np.ceil(shape[2] / block_size[0]) * np.ceil(shape[3] / block_size[1]) for block_size in self.block_sizes if 0 not in block_size)),)
        #     mult_shape = shape[0], sum(~self.is_identity_channels), shape[2], shape[3]
        #
        #     self.secure_DReLU(torch.zeros(size=mean_tensor_shape))
        #     self.secure_mult(mult_shape)
        #
        # return activation


class SecureSelectShareCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareCryptoProvider, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationCryptoProvider(**kwargs)

    def forward(self, share, dummy0=None, dummy1=None):
        self.secure_multiplication(share.shape)
        return share


class SecureMaxPoolCryptoProvider(SecureMaxPool):
    def __init__(self, kernel_size=3, stride=2, padding=1, **kwargs):
        super(SecureMaxPoolCryptoProvider, self).__init__(kernel_size, stride, padding, **kwargs)
        self.select_share = SecureSelectShareCryptoProvider(**kwargs)
        self.dReLU = SecureDReLUCryptoProvider(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)

    def forward(self, x):
        return super(SecureMaxPoolCryptoProvider, self).forward(x)
