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
from research.secure_inference_3pc.const import NUM_BITS, NUM_SPLIT_CONV_IN_CHANNEL, NUM_SPLIT_CONV_OUT_CHANNEL
import torch
import numpy as np
from numba import njit, prange, int64, uint64, int8, uint8, int32, uint32

NUMBA_INT_DTYPE = int64 if NUM_BITS == 64 else int32
NUMBA_UINT_DTYPE = uint64 if NUM_BITS == 64 else uint32

@njit((int8[:,:])(int8[:,:], NUMBA_INT_DTYPE[:], int8[:,:], int8[:],  uint8, uint8), parallel=True,  nogil=True, cache=True)
def private_compare_numba(s, r, x_bits_0, beta, bits, ignore_msb_bits):

    for i in prange(x_bits_0.shape[0]):
        r[i] = r[i] + beta[i]

        counter = 0

        for j in range(bits - ignore_msb_bits):
            decompose_bit = (r[i] >> (bits - 1 - j)) & 1
            decompose_bit = -2 * decompose_bit * x_bits_0[i, j] + x_bits_0[i, j]
            counter = counter + decompose_bit

            tmp = (counter - decompose_bit) + x_bits_0[i, j] * (2 * beta[i] - 1)
            s[i, j] = (tmp * s[i, j]) % 67

    return s

@njit((NUMBA_INT_DTYPE[:])(NUMBA_INT_DTYPE[:], int8[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:]), parallel=True,  nogil=True, cache=True)
def post_compare_numba(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0):

    out = a_0
    for i in prange(a_0.shape[0],):

        eta_pp_t = NUMBA_INT_DTYPE(eta_pp[i])
        t0 = eta_pp_t * eta_p_0[i]


        # t1 = add_mode_L_minus_one(t0, t0)
        t1 = t0 + t0
        if NUMBA_UINT_DTYPE(t0) > NUMBA_UINT_DTYPE(t1):
            t1 += 1
        if t1 == -1:
            t1 = 0


        # t2 = sub_mode_L_minus_one(eta_pp_t, t1)
        if NUMBA_UINT_DTYPE(t1) > NUMBA_UINT_DTYPE(eta_pp_t):
            t2 = eta_pp_t - t1 - 1
        else:
            t2 = eta_pp_t - t1



        # eta_0 = add_mode_L_minus_one(eta_p_0[i], t2)
        eta_0 = eta_p_0[i] + t2
        if NUMBA_UINT_DTYPE(eta_p_0[i]) > NUMBA_UINT_DTYPE(eta_0):
            eta_0 += 1
        if eta_0 == -1:
            eta_0 = 0



        # t0 = add_mode_L_minus_one(delta_0[i], eta_0)
        t0 = delta_0[i] + eta_0
        if NUMBA_UINT_DTYPE(delta_0[i]) > NUMBA_UINT_DTYPE(t0):
            t0 += 1
        if t0 == -1:
            t0 = 0




        # t1 = sub_mode_L_minus_one(t0, 1)
        if NUMBA_UINT_DTYPE(1) > NUMBA_UINT_DTYPE(t0):
            t1 = t0 - 1 - 1
        else:
            t1 = t0 - 1



        # t2 = sub_mode_L_minus_one(t1, alpha[i])
        if NUMBA_UINT_DTYPE(alpha[i]) > NUMBA_UINT_DTYPE(t1):
            t2 = t1 - alpha[i] - 1
        else:
            t2 = t1 - alpha[i]





        # theta_0 = add_mode_L_minus_one(beta_0[i], t2)
        theta_0 = beta_0[i] + t2
        if NUMBA_UINT_DTYPE(beta_0[i]) > NUMBA_UINT_DTYPE(theta_0):
            theta_0 += 1
        if theta_0 == -1:
            theta_0 = 0



        # y_0 = sub_mode_L_minus_one(a_0, theta_0)
        if NUMBA_UINT_DTYPE(theta_0) > NUMBA_UINT_DTYPE(a_0[i]):
            y_0 = a_0[i] - theta_0 - 1
        else:
            y_0 = a_0[i] - theta_0


        # y_0 = add_mode_L_minus_one(y_0, mu_0[i])
        ret = y_0 + mu_0[i]
        if NUMBA_UINT_DTYPE(y_0) > NUMBA_UINT_DTYPE(ret):
            ret += 1
        if ret == -1:
            ret = 0

        out[i] = ret
    return out

@njit((NUMBA_INT_DTYPE[:])(NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:]), parallel=True,  nogil=True, cache=True)
def mult_client_flatten(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0],):
        f1[i] = (x[i] * (f0[i] + f1[i]) + y[i] * (e0[i] + e1[i]) + c[i]) + m[i]
    return f1

@njit((NUMBA_INT_DTYPE[:, :, :, :])(NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :]), parallel=True,  nogil=True, cache=True)
def mult_client_non_flatten(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0],):
        f1[i] = (x[i] * (f0[i] + f1[i]) + y[i] * (e0[i] + e1[i]) + c[i]) + m[i]
    return f1


class SecureConv2DClient(SecureModule):

    def __init__(self, W_shape, stride, dilation, padding, groups, **kwargs):
        super(SecureConv2DClient, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.conv2d_handler = conv2d_handler_factory.create(self.device)
        self.is_dummy = False

        self.num_split_in_channels = NUM_SPLIT_CONV_IN_CHANNEL
        self.num_split_out_channels = NUM_SPLIT_CONV_OUT_CHANNEL

        self.out_channels, self.in_channels = self.W_shape[:2]

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

                outs.append(self.conv2d_handler.conv2d(X_share_splits[i],
                                                       F,
                                                       E,
                                                       W_share_splits[j][i],
                                                       padding=self.padding,
                                                       stride=self.stride,
                                                       dilation=self.dilation,
                                                       groups=self.groups))

            outs_all.append(np.concatenate(outs, axis=1))
        outs_all = np.stack(outs_all).sum(axis=0)
        return outs_all

        # outs = []
        # for i in range(div):
        #     start = i * mid
        #     end = (i + 1) * mid if i < div else None
        #
        #     F_share_server = self.network_assets.receiver_01.get()
        #     outs.append(self.conv2d_handler.conv2d
        #                 (X_share,
        #                  F_share_server + F_share[start:end],
        #                  E,
        #                  W_share[start:end],
        #                  padding=self.padding,
        #                  stride=self.stride,
        #                  dilation=self.dilation,
        #                  groups=self.groups))
        # out = np.concatenate(outs, axis=1)
        #
        #
        #
        # div = self.num_split_activation
        #
        # mid = W_share.shape[0] // div
        # self.network_assets.sender_01.put(E_share)
        # E_share_server = self.network_assets.receiver_01.get()
        # E = backend.add(E_share_server, E_share, out=E_share)
        #
        # for i in range(div):
        #     start = i * mid
        #     end = (i + 1) * mid if i < div else None
        #
        #     self.network_assets.sender_01.put(F_share[start:end])
        #
        # outs = []
        # for i in range(div):
        #     start = i * mid
        #     end = (i + 1) * mid if i < div else None
        #
        #     F_share_server = self.network_assets.receiver_01.get()
        #     outs.append(self.conv2d_handler.conv2d
        #                 (X_share,
        #                  F_share_server + F_share[start:end],
        #                  E,
        #                  W_share[start:end],
        #                  padding=self.padding,
        #                  stride=self.stride,
        #                  dilation=self.dilation,
        #                  groups=self.groups))
        # out = np.concatenate(outs, axis=1)
        # return out
    @timer(name='client_conv2d')
    def forward(self, X_share):
        if self.is_dummy:
            out_shape = get_output_shape(X_share.shape, self.W_shape, self.padding, self.dilation, self.stride)
            self.network_assets.sender_01.put(X_share)
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=out_shape, dtype=SIGNED_DTYPE)
            return mu_0

        assert self.W_shape[2] == self.W_shape[3]
        assert (self.W_shape[1] == X_share.shape[1]) or self.groups > 1

        W_share = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=W_share.shape, dtype=SIGNED_DTYPE)

        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(W_share, B_share, out=B_share)


        if self.num_split_out_channels == 1 and self.num_split_in_channels == 1:
            with Timer(name="Reconstruct"):
                self.network_assets.sender_01.put(E_share)
                self.network_assets.sender_01.put(F_share)

                E_share_server = self.network_assets.receiver_01.get()
                F_share_server = self.network_assets.receiver_01.get()

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
        else:
            out = self.split_conv(E_share, F_share, W_share, X_share)

        C_share = self.network_assets.receiver_02.get()
        out = backend.add(out, C_share, out=out)
        out = backend.right_shift(out, TRUNC_BITS, out=out)

        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)

        out = backend.add(out, mu_0, out=out)

        return out


class PrivateCompareClient(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareClient, self).__init__(**kwargs)
        self.decompose = Decompose(ignore_msb_bits=IGNORE_MSB_BITS, num_of_compare_bits=NUM_OF_COMPARE_BITS, dtype=SIGNED_DTYPE, **kwargs)

    def forward(self, x_bits_0, r, beta):

        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=P, size=x_bits_0.shape, dtype=backend.int8)
        d_bits_0 = private_compare_numba(s, r, x_bits_0, beta, NUM_OF_COMPARE_BITS, IGNORE_MSB_BITS)
        # r[backend.astype(beta, backend.bool)] += 1
        # bits = self.decompose(r)
        # c_bits_0 = get_c_party_0(x_bits_0, bits, beta)
        # s = backend.multiply(s, c_bits_0, out=s)
        # d_bits_0 = module_67(s)

        d_bits_0 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_0, axis=-1)

        self.network_assets.sender_02.put(d_bits_0)

        return



class ShareConvertClient(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertClient, self).__init__(**kwargs)
        self.private_compare = PrivateCompareClient(**kwargs)

    def post_compare(self, a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0):
        return post_compare_numba(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0)
        # eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)
        # t0 = eta_pp * eta_p_0
        # t1 = self.add_mode_L_minus_one(t0, t0)
        # t2 = self.sub_mode_L_minus_one(eta_pp, t1)
        # eta_0 = self.add_mode_L_minus_one(eta_p_0, t2)
        #
        # t0 = self.add_mode_L_minus_one(delta_0, eta_0)
        # t1 = self.sub_mode_L_minus_one(t0, backend.ones_like(t0))
        # t2 = self.sub_mode_L_minus_one(t1, alpha)
        # theta_0 = self.add_mode_L_minus_one(beta_0, t2)
        #
        # y_0 = self.sub_mode_L_minus_one(a_0, theta_0)
        # y_0 = self.add_mode_L_minus_one(y_0, mu_0)
        #
        # return y_0

    def forward(self, a_0):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=backend.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=a_0.shape, dtype=SIGNED_DTYPE)
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS], dtype=backend.int8)

        alpha = backend.astype(0 < r_0 - r, SIGNED_DTYPE)
        a_tild_0 = a_0 + r_0
        self.network_assets.sender_02.put(a_tild_0)

        beta_0 = backend.astype(0 < a_0 - a_tild_0, SIGNED_DTYPE)
        delta_0 = self.network_assets.receiver_02.get()

        self.private_compare(x_bits_0, r - 1, eta_pp)
        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_0.shape, dtype=SIGNED_DTYPE)

        return self.post_compare(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0)


def mult_client_numba(x, y, c, m, e0, e1, f0, f1):
    if x.ndim == 1:
        return mult_client_flatten(x, y, c, m, e0, e1, f0, f1)
    else:
        return mult_client_non_flatten(x, y, c, m, e0, e1, f0, f1)

class SecureMultiplicationClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationClient, self).__init__(**kwargs)

    def exchange_shares(self, E_share, F_share):

        self.network_assets.sender_01.put(E_share)
        E_share_server = self.network_assets.receiver_01.get()

        self.network_assets.sender_01.put(F_share)
        F_share_server = self.network_assets.receiver_01.get()

        return E_share_server, F_share_server

    def forward(self, X_share, Y_share):

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        E_share_server, F_share_server = self.exchange_shares(E_share, F_share)

        C_share = self.network_assets.receiver_02.get()
        out = mult_client_numba(X_share, Y_share, C_share, mu_0, E_share, E_share_server, F_share, F_share_server)
        # E = E_share_server + E_share
        # F = F_share_server + F_share
        #
        # out = X_share * F + Y_share * E + C_share
        # out = out + mu_0
        return out


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


    def forward(self, a_0):
        x_bits_0, r, beta, x_bit_0_0, r_mode_2, mu_0 = self.pre_compare(a_0)

        self.private_compare(x_bits_0, r, beta)

        return self.post_compare(beta, x_bit_0_0, r_mode_2, mu_0)


class SecureDReLUClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureDReLUClient, self).__init__(**kwargs)

        self.share_convert = ShareConvertClient(**kwargs)
        self.msb = SecureMSBClient(**kwargs)

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

    def forward(self, X_share):
        if self.dummy_relu:
            self.network_assets.sender_01.put(X_share)
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
            return mu_0
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

        self.dummy_relu = dummy_relu
    def forward(self, activation):
        if self.dummy_relu:
            return activation
        return SecureOptimizedBlockReLU.forward(self, activation)

class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W_shape, stride, dilation, padding, groups, **kwargs):
        super(PRFFetcherConv2D, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.is_dummy = False

    def forward(self, shape):
        if self.is_dummy:
            out_shape = get_output_shape(shape, self.W_shape, self.padding, self.dilation, self.stride)
            self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=out_shape, dtype=SIGNED_DTYPE)
            return DummyShapeTensor(out_shape)
        out_shape = get_output_shape(shape, self.W_shape, self.padding, self.dilation, self.stride)

        # return DummyShapeTensor(out_shape)

        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=SIGNED_DTYPE)

        return DummyShapeTensor(out_shape)


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherPrivateCompare, self).__init__(**kwars)

    def forward(self, shape):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=1, high=P, size=[shape[0]] + [NUM_OF_COMPARE_BITS - IGNORE_MSB_BITS], dtype=backend.int8)


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
        else:
            self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

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
        shape = DummyShapeTensor(img.shape)

        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, dtype=SIGNED_DTYPE, size=shape)
        out_0 = self.model.decode_head(self.model.backbone(shape))


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