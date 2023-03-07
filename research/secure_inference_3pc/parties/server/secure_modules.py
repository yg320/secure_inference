from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.const import COMPARISON_NUM_BITS_IGNORED
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, \
    TRUNC_BITS
from research.secure_inference_3pc.modules.bReLU import SecureOptimizedBlockReLU, post_brelu
from research.secure_inference_3pc.parties.server.numba_methods import private_compare_numba_server, post_compare_numba, \
    mult_server_numba
from research.secure_inference_3pc.conv2d.numba_conv2d import Conv2DHandler as NumbaConv2DHandler

from research.secure_inference_3pc.conv2d.utils import get_output_shape

import numpy as np

class SecureConv2DServer(SecureModule):
    def __init__(self, W, bias, stride, dilation, padding, groups, **kwargs):
        super(SecureConv2DServer, self).__init__(**kwargs)

        self.W_plaintext = backend.put_on_device(W, self.device)
        self.bias = bias
        if self.bias is not None:
            self.bias = backend.reshape(self.bias, [1, -1, 1, 1])
            self.bias = backend.put_on_device(self.bias, self.device)
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.conv2d_handler = NumbaConv2DHandler()

        self.dummy = False

    def forward(self, X_share):

        if self.dummy:
            share_client = self.network_assets.receiver_01.get()
            recon = share_client + X_share
            # out_shape = get_output_shape(X_share.shape, self.W_plaintext.shape, self.padding, self.dilation, self.stride)
            # out = np.zeros(shape=out_shape, dtype=SIGNED_DTYPE)
            out = self.conv2d_handler.conv2d(recon, self.W_plaintext, None, None, self.padding, self.stride, self.dilation, self.groups)
            out = backend.right_shift(out, TRUNC_BITS, out=out)
            if self.bias is not None:
                out = backend.add(out, self.bias, out=out)

            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=out.shape, dtype=SIGNED_DTYPE)
            return out - mu_0

        assert self.W_plaintext.shape[2] == self.W_plaintext.shape[3]
        assert (self.W_plaintext.shape[1] == X_share.shape[1]) or self.groups > 1
        assert self.stride[0] == self.stride[1]

        W_client = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, size=self.W_plaintext.shape,
                                                             dtype=SIGNED_DTYPE)
        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape,
                                                                     dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_plaintext.shape,
                                                                     dtype=SIGNED_DTYPE)

        W_share = backend.subtract(self.W_plaintext, W_client, out=W_client)
        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(W_share, B_share, out=B_share)

        self.network_assets.sender_01.put(E_share)
        self.network_assets.sender_01.put(F_share)

        E_share_client = self.network_assets.receiver_01.get()
        F_share_client = self.network_assets.receiver_01.get()

        E = backend.add(E_share_client, E_share, out=E_share)
        F = backend.add(F_share_client, F_share, out=F_share)

        W_share = backend.subtract(W_share, F, out=W_share)

        out = self.conv2d_handler.conv2d(E, W_share, X_share, F, self.padding, self.stride, self.dilation, self.groups)

        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)

        out = backend.add(out, C_share, out=out)
        out = backend.right_shift(out, TRUNC_BITS, out=out)

        if self.bias is not None:
            out = backend.add(out, self.bias, out=out)

        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        out = backend.add(out, mu_1, out=out)

        return out


class PrivateCompareServer(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareServer, self).__init__(**kwargs)

    def forward(self, x_bits_1, r, beta):
        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_1.shape, dtype=backend.int8)
        d_bits_1 = private_compare_numba_server(s, r, x_bits_1, beta, COMPARISON_NUM_BITS_IGNORED)

        d_bits_1 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_1, axis=-1)

        self.network_assets.sender_12.put(d_bits_1)

        return


class ShareConvertServer(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertServer, self).__init__(**kwargs)
        self.private_compare = PrivateCompareServer(**kwargs)

    def forward(self, a_1):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=backend.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=SIGNED_DTYPE)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)

        r_1 = backend.subtract(r, r_0, out=r_0)
        a_tild_1 = backend.add(a_1, r_1, out=r_1)
        beta_1 = backend.astype(0 < a_1 - a_tild_1, SIGNED_DTYPE)  # TODO: Optimize this

        self.network_assets.sender_12.put(a_tild_1)

        x_bits_1 = self.network_assets.receiver_12.get()

        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_1.shape,
                                                                     dtype=SIGNED_DTYPE)

        r_minus_1 = backend.subtract(r, 1, out=r)
        self.private_compare(x_bits_1, r_minus_1, eta_pp)
        eta_p_1 = self.network_assets.receiver_12.get()

        return post_compare_numba(a_1, eta_pp, delta_1, beta_1, mu_0, eta_p_1)


class SecureMultiplicationServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationServer, self).__init__(**kwargs)

    def forward(self, X_share, Y_share):
        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=X_share.dtype)

        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(Y_share, B_share, out=B_share)

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        out = mult_server_numba(X_share, Y_share, C_share, mu_1, E_share, E_share_client, F_share, F_share_client)

        return out


class SecureMSBServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBServer, self).__init__(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)
        self.private_compare = PrivateCompareServer(**kwargs)

    def forward(self, a_1):
        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=backend.int8)
        x_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=a_1.dtype)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)

        x_bits_1 = self.network_assets.receiver_12.get()
        x_bit_0_1 = self.network_assets.receiver_12.get()

        y_1 = self.add_mode_L_minus_one(a_1, a_1)
        r_1 = self.add_mode_L_minus_one(x_1, y_1)

        self.network_assets.sender_01.put(r_1)
        r_0 = self.network_assets.receiver_01.get()

        r = self.add_mode_L_minus_one(r_0, r_1)
        r_mod_2 = r % 2

        self.private_compare(x_bits_1, r, beta)
        beta_p_1 = self.network_assets.receiver_12.get()

        beta = backend.astype(beta, SIGNED_DTYPE)  # TODO: Optimize this
        gamma_1 = beta_p_1 + beta - 2 * beta * beta_p_1  # TODO: Optimize this
        delta_1 = x_bit_0_1 + r_mod_2 - (2 * r_mod_2 * x_bit_0_1)  # TODO: Optimize this

        theta_1 = self.mult(gamma_1, delta_1)

        alpha_1 = gamma_1 + delta_1 - 2 * theta_1  # TODO: Optimize this
        alpha_1 = alpha_1 + mu_1  # TODO: Optimize this

        return alpha_1


class SecureDReLUServer(SecureModule):
    # counter = 0
    def __init__(self, **kwargs):
        super(SecureDReLUServer, self).__init__(**kwargs)

        self.share_convert = ShareConvertServer(**kwargs)
        self.msb = SecureMSBServer(**kwargs)

        self.dummy = False

    def forward(self, X_share):
        # SecureDReLUServer.counter += 1
        # np.save("/home/yakir/Data2/secure_activation_statistics/server/{}.npy".format(SecureDReLUServer.counter), X_share)
        if self.dummy:
            share_client = self.network_assets.receiver_01.get()
            recon = share_client + X_share
            value = backend.astype(recon > 0, recon.dtype)
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=value.shape, dtype=SIGNED_DTYPE)
            return value - mu_0
        else:
            mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
            backend.multiply(mu_1, -1, out=mu_1)

            X1_converted = self.share_convert(X_share)
            MSB_1 = self.msb(X1_converted)

            ret = backend.multiply(MSB_1, -1, out=MSB_1)
            ret = backend.add(ret, mu_1, out=ret)
            ret = backend.add(ret, 1, out=ret)
            return ret


class SecureReLUServer(SecureModule):
    # index = 0
    def __init__(self, dummy_relu=False, **kwargs):
        super(SecureReLUServer, self).__init__(**kwargs)

        self.DReLU = SecureDReLUServer(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)

    def forward(self, X_share):

        shape = X_share.shape
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        backend.multiply(mu_1, -1, out=mu_1)

        X_share = X_share.reshape(-1)
        MSB_0 = self.DReLU(X_share)
        ret = self.mult(X_share, MSB_0).reshape(shape)
        backend.add(ret, mu_1, out=ret)
        return ret


class SecurePostBReLUMultServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecurePostBReLUMultServer, self).__init__(**kwargs)

    def forward(self, activation, sign_tensors, cumsum_shapes, pad_handlers, is_identity_channels, active_block_sizes,
                active_block_sizes_to_channels, stacked_active_block_sizes_to_channels, offsets, channel_map):
        non_identity_activation = activation[:, ~is_identity_channels]

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=non_identity_activation.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=sign_tensors.shape, dtype=SIGNED_DTYPE)
        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=non_identity_activation.shape, dtype=SIGNED_DTYPE)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=activation.shape, dtype=activation.dtype)

        E_share = backend.subtract(non_identity_activation, A_share, out=A_share)
        F_share = backend.subtract(sign_tensors, B_share, out=B_share)

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        E = backend.add(E_share_client, E_share, out=E_share)
        F = backend.add(F_share_client, F_share, out=F_share)

        F = post_brelu(activation, F, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels, stacked_active_block_sizes_to_channels, offsets, is_identity_channels, channel_map)
        sign_tensors = post_brelu(activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels, stacked_active_block_sizes_to_channels, offsets, is_identity_channels, channel_map)

        out = - E * F + non_identity_activation * F + sign_tensors * E + C_share
        activation[:, ~is_identity_channels] = out
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        activation = activation + mu_1
        return activation


class SecureBlockReLUServer(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUServer(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)
        self.post_bReLU = SecurePostBReLUMultServer(**kwargs)

    def forward(self, activation):

        return SecureOptimizedBlockReLU.forward(self, activation)


class SecureSelectShareServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareServer, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationServer(**kwargs)

    def forward(self, alpha, x, y):
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=alpha.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        y = backend.subtract(y, x, out=y)

        c = self.secure_multiplication(alpha, y)
        x = backend.add(x, c, out=x)
        x = backend.add(x, mu_1, out=x)
        return x


class SecureMaxPoolServer(SecureMaxPool):
    def __init__(self, kernel_size=3, stride=2, padding=1, **kwargs):
        super(SecureMaxPoolServer, self).__init__(kernel_size, stride, padding, **kwargs)
        self.select_share = SecureSelectShareServer(**kwargs)
        self.dReLU = SecureDReLUServer(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)

    def forward(self, x):
        ret = super(SecureMaxPoolServer, self).forward(x)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        ret = backend.add(ret, mu_1, out=ret)
        return ret
