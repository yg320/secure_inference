from research.secure_inference_3pc.backend import backend

from research.secure_inference_3pc.modules.base import PRFFetcherModule
from research.secure_inference_3pc.conv2d.utils import get_output_shape
from research.secure_inference_3pc.const import NUM_OF_COMPARE_BITS, IGNORE_MSB_BITS

from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.base import get_c_party_1, module_67
from research.secure_inference_3pc.conv2d.conv2d_handler_factory import conv2d_handler_factory
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, TRUNC_BITS
from research.bReLU import SecureOptimizedBlockReLU, unpack_bReLU
from research.secure_inference_3pc.modules.base import Decompose
from research.secure_inference_3pc.modules.base import DummyShapeTensor
from research.secure_inference_3pc.timer import Timer, timer
from research.secure_inference_3pc.const import NUM_BITS, NUM_SPLIT_CONV_IN_CHANNEL, NUM_SPLIT_CONV_OUT_CHANNEL
import torch
import numpy as np

from numba import njit, prange, int64, uint64, int8, uint8, int32, uint32

NUMBA_INT_DTYPE = int64 if NUM_BITS == 64 else int32
NUMBA_UINT_DTYPE = uint64 if NUM_BITS == 64 else uint32

@njit((int8[:,:])(int8[:,:], NUMBA_INT_DTYPE[:], int8[:,:], int8[:],  uint8,  uint8), parallel=True, nogil=True, cache=True)
def private_compare_numba_server(s, r, x_bits_1, beta, bits, ignore_msb_bits):
    for i in prange(x_bits_1.shape[0]):
        r[i] = r[i] + beta[i]
        counter = 0
        for j in range(bits - ignore_msb_bits):
            multiplexer_bit = (r[i] >> (bits - 1 - j)) & 1

            w = -2 * multiplexer_bit * x_bits_1[i, j] + x_bits_1[i, j] + multiplexer_bit

            counter = counter + w
            w_cumsum = counter - w

            multiplexer_bit = multiplexer_bit - x_bits_1[i, j]
            multiplexer_bit = multiplexer_bit * (-2 * beta[i] + 1)
            multiplexer_bit = multiplexer_bit + 1
            w_cumsum = w_cumsum + multiplexer_bit

            s[i, j] = (s[i, j] * w_cumsum) % 67
    return s

@njit((NUMBA_INT_DTYPE[:])(NUMBA_INT_DTYPE[:], int8[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:]), parallel=True,  nogil=True, cache=True)
def post_compare_numba(a_1, eta_pp, delta_1,  beta_1, mu_0, eta_p_1):

    out = a_1
    for i in prange(a_1.shape[0],):
        # eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)
        eta_pp_t = NUMBA_INT_DTYPE(eta_pp[i])

        # t00 = backend.multiply(eta_pp, eta_p_1, out=eta_pp)
        t00 = eta_pp_t * eta_p_1[i]


        # t11 = self.add_mode_L_minus_one(t00, t00)
        t11 = t00 + t00
        if NUMBA_UINT_DTYPE(t00) > NUMBA_UINT_DTYPE(t11):
            t11 += 1
        if t11 == -1:
            t11 = 0


        # eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)
        if NUMBA_UINT_DTYPE(t11) > NUMBA_UINT_DTYPE(eta_p_1[i]):
            eta_1 = eta_p_1[i] - t11 - 1
        else:
            eta_1 = eta_p_1[i] - t11



        # t00 = self.add_mode_L_minus_one(delta_1, eta_1)
        t00 = delta_1[i] + eta_1
        if NUMBA_UINT_DTYPE(delta_1[i]) > NUMBA_UINT_DTYPE(t00):
            t00 += 1
        if t00 == -1:
            t00 = 0


        # theta_1 = self.add_mode_L_minus_one(beta_1, t00)
        theta_1 = beta_1[i] + t00
        if NUMBA_UINT_DTYPE(t00) > NUMBA_UINT_DTYPE(theta_1):
            theta_1 += 1
        if theta_1 == -1:
            theta_1 = 0


        # y_1 = self.sub_mode_L_minus_one(a_1, theta_1)
        if NUMBA_UINT_DTYPE(theta_1) > NUMBA_UINT_DTYPE(a_1[i]):
            y_1 = a_1[i] - theta_1 - 1
        else:
            y_1 = a_1[i] - theta_1


        # y_1 = self.add_mode_L_minus_one(y_1, mu_1)
        ret = y_1 - mu_0[i]
        if NUMBA_UINT_DTYPE(y_1) > NUMBA_UINT_DTYPE(ret):
            ret += 1
        if ret == -1:
            ret = 0

        out[i] = ret
    return out

@njit((NUMBA_INT_DTYPE[:])(NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:]), parallel=True,  nogil=True, cache=True)
def mult_server_flatten(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0],):
        e = (e0[i] + e1[i])
        f = (f0[i] + f1[i])
        f1[i] = - e * f + x[i] * f + y[i] * e + c[i] - m[i]
    return f1

@njit((NUMBA_INT_DTYPE[:, :, :, :])(NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :]), parallel=True,  nogil=True, cache=True)
def mult_server_non_flatten(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0],):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    e = (e0[i, j, k, l] + e1[i, j, k, l])
                    f = (f0[i, j, k, l] + f1[i, j, k, l])
                    f1[i, j, k, l] = - e * f + x[i, j, k, l] * f + y[i, j, k, l] * e + c[i, j, k, l] - m[i, j, k, l]
        # e = (e0[i] + e1[i])
        # f = (f0[i] + f1[i])
        # f1[i] = - e * f + x[i] * f + y[i] * e + c[i] - m[i]
    return f1

class SecureConv2DServer(SecureModule):
    def __init__(self, W, bias, stride, dilation, padding, groups,  **kwargs):
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
        self.conv2d_handler = conv2d_handler_factory.create(self.device)
        self.is_dummy = False

        self.num_split_in_channels = NUM_SPLIT_CONV_IN_CHANNEL
        self.num_split_out_channels = NUM_SPLIT_CONV_OUT_CHANNEL

        self.out_channels, self.in_channels = self.W_plaintext.shape[:2]

        self.in_channel_group_size = self.in_channels // self.num_split_in_channels
        self.out_channel_group_size = self.out_channels // self.num_split_out_channels

        self.in_channel_s = [self.in_channel_group_size * i for i in range(self.num_split_in_channels)]
        self.in_channel_e = self.in_channel_s[1:] + [None]

        self.out_channel_s = [self.out_channel_group_size * i for i in range(self.num_split_out_channels)]
        self.out_channel_e = self.out_channel_s[1:] + [None]


    def split_conv(self, E_share, F_share, W_share, X_share):

        E_share_splits = [E_share[:, s_in:e_in] for s_in, e_in in zip(self.in_channel_s, self.in_channel_e)]
        F_share_splits = [[F_share[s_out:e_out, s_in:e_in] for s_in, e_in in zip(self.in_channel_s, self.in_channel_e)] for s_out, e_out in zip(self.out_channel_s, self.out_channel_e)]
        X_share_splits = [X_share[:, s_in:e_in] for s_in, e_in in zip(self.in_channel_s, self.in_channel_e)]
        W_share_splits = [[W_share[s_out:e_out, s_in:e_in] for s_in, e_in in zip(self.in_channel_s, self.in_channel_e)] for s_out, e_out in zip(self.out_channel_s, self.out_channel_e)]

        for i in range(self.num_split_in_channels):
            self.network_assets.sender_01.put(E_share_splits[i])
            for j in range(self.num_split_out_channels):
                self.network_assets.sender_01.put(F_share_splits[j][i])

        outs_all = []
        for i in range(self.num_split_in_channels):
            E_share_server = self.network_assets.receiver_01.get()
            E = E_share_server + E_share_splits[i]
            outs = []
            for j in range(self.num_split_out_channels):
                F_share_server = self.network_assets.receiver_01.get()
                F = F_share_server + F_share_splits[j][i]
                cur_W_share = W_share_splits[j][i] - F
                outs.append(self.conv2d_handler.conv2d(X_share_splits[i],
                                                       F,
                                                       E,
                                                       cur_W_share,
                                                       padding=self.padding,
                                                       stride=self.stride,
                                                       dilation=self.dilation,
                                                       groups=self.groups))
            outs_all.append(np.concatenate(outs, axis=1))
        outs_all = np.stack(outs_all).sum(axis=0)
        return outs_all
        # div = self.num_split_weights
        # mid = W_share.shape[0] // div
        # self.network_assets.sender_01.put(E_share)
        # E_share_client = self.network_assets.receiver_01.get()
        # E = backend.add(E_share_client, E_share, out=E_share)
        #
        # for i in range(div):
        #
        #     start = i * mid
        #     end = (i + 1) * mid if i < div else None
        #     self.network_assets.sender_01.put(F_share[start:end])
        #
        # outs = []
        # for i in range(div):
        #
        #     start = i * mid
        #     end = (i + 1) * mid if i < div else None
        #
        #     F_share_client = self.network_assets.receiver_01.get()
        #
        #     F = F_share_client + F_share[start:end]
        #
        #     W_share_cur = W_share[start:end] - F
        #
        #
        #     outs.append(self.conv2d_handler.conv2d(E,
        #                                            W_share_cur,
        #                                            X_share,
        #                                            F,
        #                                            padding=self.padding,
        #                                            stride=self.stride,
        #                                            dilation=self.dilation,
        #                                            groups=self.groups))
        # out = np.concatenate(outs, axis=1)
        # return out

    # @timer(name='server_conv2d')
    def forward(self, X_share):
        if self.is_dummy:
            X_share_client = self.network_assets.receiver_01.get()
            X = X_share_client + X_share
            out = self.conv2d_handler.conv2d(X, self.W_plaintext, None, None, self.padding, self.stride, self.dilation, self.groups)
            out = backend.right_shift(out, TRUNC_BITS, out=out)
            if self.bias is not None:
                out = backend.add(out, self.bias, out=out)
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=out.shape, dtype=SIGNED_DTYPE)
            return out - mu_0

        assert self.W_plaintext.shape[2] == self.W_plaintext.shape[3]
        assert (self.W_plaintext.shape[1] == X_share.shape[1]) or self.groups > 1
        assert self.stride[0] == self.stride[1]

        W_client = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, size=self.W_plaintext.shape, dtype=SIGNED_DTYPE)
        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_plaintext.shape, dtype=SIGNED_DTYPE)

        W_share = backend.subtract(self.W_plaintext, W_client, out=W_client)
        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(W_share, B_share, out=B_share)


        if self.num_split_out_channels == 1 and self.num_split_in_channels == 1:

            self.network_assets.sender_01.put(E_share)
            self.network_assets.sender_01.put(F_share)

            E_share_client = self.network_assets.receiver_01.get()
            F_share_client = self.network_assets.receiver_01.get()

            E = backend.add(E_share_client, E_share, out=E_share)
            F = backend.add(F_share_client, F_share, out=F_share)

            W_share = backend.subtract(W_share, F, out=W_share)

            out = self.conv2d_handler.conv2d(E, W_share, X_share, F, self.padding, self.stride, self.dilation, self.groups)

        else:
            out = self.split_conv(E_share, F_share, W_share, X_share)

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
        self.decompose = Decompose(ignore_msb_bits=IGNORE_MSB_BITS, num_of_compare_bits=NUM_OF_COMPARE_BITS, dtype=SIGNED_DTYPE, **kwargs)

    def forward(self, x_bits_1, r, beta):

        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_1.shape, dtype=backend.int8)
        d_bits_1 = private_compare_numba_server(s, r, x_bits_1, beta, NUM_OF_COMPARE_BITS, IGNORE_MSB_BITS)

        # r[backend.astype(beta, backend.bool)] += 1
        # bits = self.decompose(r)
        # c_bits_1 = get_c_party_1(x_bits_1, bits, beta)
        # s = backend.multiply(s, c_bits_1, out=s)
        # d_bits_1 = module_67(s)

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

        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)

        r_minus_1 = backend.subtract(r, 1, out=r)
        self.private_compare(x_bits_1, r_minus_1, eta_pp)
        eta_p_1 = self.network_assets.receiver_12.get()

        return post_compare_numba(a_1, eta_pp, delta_1, beta_1, mu_0, eta_p_1)
        # mu_1 = backend.multiply(mu_0, -1, out=mu_0)
        # eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)
        # t00 = backend.multiply(eta_pp, eta_p_1, out=eta_pp)
        # t11 = self.add_mode_L_minus_one(t00, t00)
        # eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)
        # t00 = self.add_mode_L_minus_one(delta_1, eta_1)
        # theta_1 = self.add_mode_L_minus_one(beta_1, t00)
        # y_1 = self.sub_mode_L_minus_one(a_1, theta_1)
        # y_1 = self.add_mode_L_minus_one(y_1, mu_1)
        # return y_1


def mult_server_numba(x, y, c, m, e0, e1, f0, f1):
    if x.ndim == 1:
        return mult_server_flatten(x, y, c, m, e0, e1, f0, f1)
    else:
        return mult_server_non_flatten(x, y, c, m, e0, e1, f0, f1)


class SecurePostBReLUMultServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecurePostBReLUMultServer, self).__init__(**kwargs)


    def forward(self, activation, sign_tensors, cumsum_shapes,  pad_handlers, active_block_sizes, active_block_sizes_to_channels):

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=activation.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=sign_tensors.shape, dtype=SIGNED_DTYPE)
        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=activation.shape, dtype=SIGNED_DTYPE)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=activation.shape, dtype=activation.dtype)

        E_share = backend.subtract(activation, A_share, out=A_share)
        F_share = backend.subtract(sign_tensors, B_share, out=B_share)

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        E = backend.add(E_share_client, E_share, out=E_share)
        F = backend.add(F_share_client, F_share, out=F_share)

        F = unpack_bReLU(activation, F, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels)
        sign_tensors = unpack_bReLU(activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels)

        out = - E * F + activation * F + sign_tensors * E + C_share

        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        out = out + mu_1
        return out



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

        # E = backend.add(E_share_client, E_share, out=E_share)
        # F = backend.add(F_share_client, F_share, out=F_share)
        # out = - E * F + X_share * F + Y_share * E + C_share
        # mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        # out = out + mu_1

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
    def __init__(self,  **kwargs):
        super(SecureDReLUServer, self).__init__( **kwargs)

        self.share_convert = ShareConvertServer( **kwargs)
        self.msb = SecureMSBServer(**kwargs)

    def forward(self, X_share):
        # SecureDReLUServer.counter += 1
        # np.save("/home/yakir/Data2/secure_activation_statistics/server/{}.npy".format(SecureDReLUServer.counter), X_share)

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
    def __init__(self, dummy_relu=False,  **kwargs):
        super(SecureReLUServer, self).__init__( **kwargs)

        self.DReLU = SecureDReLUServer( **kwargs)
        self.mult = SecureMultiplicationServer( **kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        # return X_share
        if self.dummy_relu:
            share_client = self.network_assets.receiver_01.get()
            recon = share_client + X_share
            value = recon * (backend.astype(recon > 0, recon.dtype))
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=value.shape, dtype=SIGNED_DTYPE)
            return value - mu_0
        else:

            shape = X_share.shape
            mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
            backend.multiply(mu_1, -1, out=mu_1)

            X_share = X_share.reshape(-1)
            MSB_0 = self.DReLU(X_share)
            ret = self.mult(X_share, MSB_0).reshape(shape)
            backend.add(ret, mu_1, out=ret)
            return ret


class SecureBlockReLUServer(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes,  dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUServer(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)
        self.dummy_relu = dummy_relu
        self.post_bReLU = SecurePostBReLUMultServer(**kwargs)

    def forward(self, activation):
        if self.dummy_relu:
            return activation
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
    def __init__(self, kernel_size=3, stride=2, padding=1,  **kwargs):
        super(SecureMaxPoolServer, self).__init__(kernel_size, stride, padding,  **kwargs)
        self.select_share = SecureSelectShareServer( **kwargs)
        self.dReLU = SecureDReLUServer( **kwargs)
        self.mult = SecureMultiplicationServer( **kwargs)

    def forward(self, x):

        ret = super(SecureMaxPoolServer, self).forward(x)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        ret = backend.add(ret, mu_1, out=ret)
        return ret



# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors
class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W, bias, stride, dilation, padding, groups, device="cpu", **kwargs):
        super(PRFFetcherConv2D, self).__init__(**kwargs)

        self.W_shape = W.shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.is_dummy = False
    def forward(self, shape):
        if self.is_dummy:
            out_shape = get_output_shape(shape, self.W_shape, self.padding, self.dilation, self.stride)
            self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=out_shape, dtype=SIGNED_DTYPE)
            return DummyShapeTensor(out_shape)
        # out_shape = get_output_shape(shape, self.W_shape, self.padding, self.dilation, self.stride)
        # return DummyShapeTensor(out_shape)

        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)

        out_shape = get_output_shape(shape, self.W_shape, self.padding, self.dilation, self.stride)

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=SIGNED_DTYPE)

        return DummyShapeTensor(out_shape)


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherPrivateCompare, self).__init__(**kwargs)

    def forward(self, shape):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=1, high=67, size=[shape[0]] + [NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS], dtype=backend.int8)


class PRFFetcherShareConvert(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherShareConvert, self).__init__(**kwargs)
        self.private_compare = PRFFetcherPrivateCompare(**kwargs)

    def forward(self, shape):
        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=shape, dtype=backend.int8)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)

        self.private_compare(shape)

        return shape


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherMultiplication, self).__init__(**kwargs)

    def forward(self, shape):

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)

        return shape

class PRFFetcherSelectShare(PRFFetcherModule):
    def __init__(self,  **kwargs):
        super(PRFFetcherSelectShare, self).__init__( **kwargs)
        self.mult = PRFFetcherMultiplication( **kwargs)


    def forward(self, shape):

        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.mult(shape)
        return shape


class PRFFetcherMSB(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherMSB, self).__init__(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)
        self.private_compare = PRFFetcherPrivateCompare(**kwargs)

    def forward(self, shape):

        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=shape, dtype=backend.int8)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
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
        else:
            self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        return shape


class PRFFetcherMaxPool(PRFFetcherModule):
    def __init__(self, kernel_size=3, stride=2, padding=1,  **kwargs):
        super(PRFFetcherMaxPool, self).__init__( **kwargs)

        self.select_share = PRFFetcherSelectShare( **kwargs)
        self.dReLU = PRFFetcherDReLU( **kwargs)
        self.mult = PRFFetcherMultiplication( **kwargs)

    def forward(self, shape):

        assert shape[2] == 112
        assert shape[3] == 112
        shape = DummyShapeTensor((shape[0], shape[1], 56, 56))
        shape_2 = DummyShapeTensor((shape[0] * shape[1] * 56 * 56,))

        for i in range(1, 9):
            self.dReLU(shape_2)
            self.select_share(shape_2)

        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        return shape


class PRFFetcherBlockReLU(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False,  **kwargs):
        SecureModule.__init__(self,  **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.secure_DReLU = PRFFetcherDReLU( **kwargs)
        self.secure_mult = PRFFetcherMultiplication( **kwargs)

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
    def __init__(self, model,   **kwargs):
        super(PRFFetcherSecureModelSegmentation, self).__init__( **kwargs)
        self.model = model

    def forward(self, img):
        shape = DummyShapeTensor(img.shape)

        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        out_0 = self.model.decode_head(self.model.backbone(shape))


class PRFFetcherSecureModelClassification(SecureModule):
    def __init__(self, model,   **kwargs):
        super(PRFFetcherSecureModelClassification, self).__init__(  **kwargs)
        self.model = model

    def forward(self, img):
        shape = DummyShapeTensor(img.shape)
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, dtype=SIGNED_DTYPE, size=shape)
        out = self.model.backbone(shape)[0]
        out = self.model.neck(out)
        out_0 = self.model.head.fc(out)