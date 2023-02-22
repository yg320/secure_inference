import numpy as np
from numba import njit, prange, int64, uint64, int8, uint8, int32, uint32, int16, uint16

from research.secure_inference_3pc.modules.base import PRFFetcherModule
from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.base import P
from research.secure_inference_3pc.conv2d.conv2d_handler_factory import conv2d_handler_factory
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, TRUNC_BITS, DTYPE_TO_BITS
from research.secure_inference_3pc.conv2d.utils import get_output_shape
from research.secure_inference_3pc.modules.base import DummyShapeTensor
from research.secure_inference_3pc.const import NUM_BITS, NUM_SPLIT_CONV_IN_CHANNEL, NUM_SPLIT_CONV_OUT_CHANNEL

from research.bReLU import SecureOptimizedBlockReLU, unpack_bReLU



# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors

NUMBA_INT_DTYPE = int64 if NUM_BITS == 64 else int32
NUMBA_UINT_DTYPE = uint64 if NUM_BITS == 64 else uint32


@njit((int8[:,:])(int8[:,:], int64[:], int8[:,:], int8[:],  uint8, uint8), parallel=True,  nogil=True, cache=True)
def private_compare_numba_64_bits(s, r, x_bits_0, beta):
    for i in prange(x_bits_0.shape[0]):
        r[i] = r[i] + beta[i]

        counter = 0

        for j in range(64):
            decompose_bit = (r[i] >> (63 - j)) & 1
            decompose_bit = -2 * decompose_bit * x_bits_0[i, j] + x_bits_0[i, j]
            counter = counter + decompose_bit

            tmp = (counter - decompose_bit) + x_bits_0[i, j] * (2 * beta[i] - 1)
            s[i, j] = (tmp * s[i, j]) % 67

    return s


@njit((int8[:,:])(int8[:,:], int16[:], int8[:,:], int8[:],  uint8, uint8), parallel=True,  nogil=True, cache=True)
def private_compare_numba_16_bits(s, r, x_bits_0, beta):

    for i in prange(x_bits_0.shape[0]):
        r[i] = r[i] + beta[i]

        counter = 0

        for j in range(16):
            decompose_bit = (r[i] >> (15 - j)) & 1
            decompose_bit = -2 * decompose_bit * x_bits_0[i, j] + x_bits_0[i, j]
            counter = counter + decompose_bit

            tmp = (counter - decompose_bit) + x_bits_0[i, j] * (2 * beta[i] - 1)
            s[i, j] = (tmp * s[i, j]) % 67

    return s


@njit((int64[:])(int64[:], int8[:], int64[:], int64[:], int64[:], int64[:], int64[:]), parallel=True,  nogil=True, cache=True)
def post_compare_numba_64_bits(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0):

    out = a_0
    for i in prange(a_0.shape[0],):

        eta_pp_t = NUMBA_INT_DTYPE(eta_pp[i])
        t0 = eta_pp_t * eta_p_0[i]


        # t1 = add_mode_L_minus_one(t0, t0)
        t1 = t0 + t0
        if uint64(t0) > uint64(t1):
            t1 += 1
        if t1 == -1:
            t1 = 0


        # t2 = sub_mode_L_minus_one(eta_pp_t, t1)
        if uint64(t1) > uint64(eta_pp_t):
            t2 = eta_pp_t - t1 - 1
        else:
            t2 = eta_pp_t - t1



        # eta_0 = add_mode_L_minus_one(eta_p_0[i], t2)
        eta_0 = eta_p_0[i] + t2
        if uint64(eta_p_0[i]) > uint64(eta_0):
            eta_0 += 1
        if eta_0 == -1:
            eta_0 = 0



        # t0 = add_mode_L_minus_one(delta_0[i], eta_0)
        t0 = delta_0[i] + eta_0
        if uint64(delta_0[i]) > uint64(t0):
            t0 += 1
        if t0 == -1:
            t0 = 0




        # t1 = sub_mode_L_minus_one(t0, 1)
        if uint64(1) > uint64(t0):
            t1 = t0 - 1 - 1
        else:
            t1 = t0 - 1



        # t2 = sub_mode_L_minus_one(t1, alpha[i])
        if uint64(alpha[i]) > uint64(t1):
            t2 = t1 - alpha[i] - 1
        else:
            t2 = t1 - alpha[i]





        # theta_0 = add_mode_L_minus_one(beta_0[i], t2)
        theta_0 = beta_0[i] + t2
        if uint64(beta_0[i]) > uint64(theta_0):
            theta_0 += 1
        if theta_0 == -1:
            theta_0 = 0



        # y_0 = sub_mode_L_minus_one(a_0, theta_0)
        if uint64(theta_0) > uint64(a_0[i]):
            y_0 = a_0[i] - theta_0 - 1
        else:
            y_0 = a_0[i] - theta_0


        # y_0 = add_mode_L_minus_one(y_0, mu_0[i])
        ret = y_0 + mu_0[i]
        if uint64(y_0) > uint64(ret):
            ret += 1
        if ret == -1:
            ret = 0

        out[i] = ret
    return out


@njit((int16[:])(int16[:], int8[:], int16[:], int16[:], int16[:], int16[:], int16[:]), parallel=True,  nogil=True, cache=True)
def post_compare_numba_16_bits(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0):

    out = a_0
    for i in prange(a_0.shape[0],):

        eta_pp_t = NUMBA_INT_DTYPE(eta_pp[i])
        t0 = eta_pp_t * eta_p_0[i]


        # t1 = add_mode_L_minus_one(t0, t0)
        t1 = t0 + t0
        if uint16(t0) > uint16(t1):
            t1 += 1
        if t1 == -1:
            t1 = 0


        # t2 = sub_mode_L_minus_one(eta_pp_t, t1)
        if uint16(t1) > uint16(eta_pp_t):
            t2 = eta_pp_t - t1 - 1
        else:
            t2 = eta_pp_t - t1



        # eta_0 = add_mode_L_minus_one(eta_p_0[i], t2)
        eta_0 = eta_p_0[i] + t2
        if uint16(eta_p_0[i]) > uint16(eta_0):
            eta_0 += 1
        if eta_0 == -1:
            eta_0 = 0



        # t0 = add_mode_L_minus_one(delta_0[i], eta_0)
        t0 = delta_0[i] + eta_0
        if uint16(delta_0[i]) > uint16(t0):
            t0 += 1
        if t0 == -1:
            t0 = 0




        # t1 = sub_mode_L_minus_one(t0, 1)
        if uint16(1) > uint16(t0):
            t1 = t0 - 1 - 1
        else:
            t1 = t0 - 1



        # t2 = sub_mode_L_minus_one(t1, alpha[i])
        if uint16(alpha[i]) > uint16(t1):
            t2 = t1 - alpha[i] - 1
        else:
            t2 = t1 - alpha[i]





        # theta_0 = add_mode_L_minus_one(beta_0[i], t2)
        theta_0 = beta_0[i] + t2
        if uint16(beta_0[i]) > uint16(theta_0):
            theta_0 += 1
        if theta_0 == -1:
            theta_0 = 0



        # y_0 = sub_mode_L_minus_one(a_0, theta_0)
        if uint16(theta_0) > uint16(a_0[i]):
            y_0 = a_0[i] - theta_0 - 1
        else:
            y_0 = a_0[i] - theta_0


        # y_0 = add_mode_L_minus_one(y_0, mu_0[i])
        ret = y_0 + mu_0[i]
        if uint16(y_0) > uint16(ret):
            ret += 1
        if ret == -1:
            ret = 0

        out[i] = ret
    return out


@njit((int64[:])(int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:]), parallel=True,  nogil=True, cache=True)
def mult_client_flatten_64_bits(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0],):
        f1[i] = (x[i] * (f0[i] + f1[i]) + y[i] * (e0[i] + e1[i]) + c[i]) + m[i]
    return f1


@njit((int64[:, :, :, :])(int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :]), parallel=True,  nogil=True, cache=True)
def mult_client_non_flatten_64_bits(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0],):
        f1[i] = (x[i] * (f0[i] + f1[i]) + y[i] * (e0[i] + e1[i]) + c[i]) + m[i]
    return f1


@njit((int16[:])(int16[:], int16[:], int16[:], int16[:], int16[:], int16[:], int16[:], int16[:]), parallel=True,  nogil=True, cache=True)
def mult_client_flatten_16_bits(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0],):
        f1[i] = (x[i] * (f0[i] + f1[i]) + y[i] * (e0[i] + e1[i]) + c[i]) + m[i]
    return f1


@njit((int16[:, :, :, :])(int16[:, :, :, :], int16[:, :, :, :], int16[:, :, :, :], int16[:, :, :, :], int16[:, :, :, :], int16[:, :, :, :], int16[:, :, :, :], int16[:, :, :, :]), parallel=True,  nogil=True, cache=True)
def mult_client_non_flatten_16_bits(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0],):
        f1[i] = (x[i] * (f0[i] + f1[i]) + y[i] * (e0[i] + e1[i]) + c[i]) + m[i]
    return f1


dtype_to_post_compare = {
    np.dtype('int16'): post_compare_numba_16_bits,
    np.dtype('int64'): post_compare_numba_64_bits,
}

dtype_to_private_compare = {
    np.dtype('int16'): private_compare_numba_16_bits,
    np.dtype('int64'): private_compare_numba_64_bits,
}

dtype_to_mult_non_flatten = {
    np.dtype('int16'): mult_client_non_flatten_16_bits,
    np.dtype('int64'): mult_client_non_flatten_64_bits,
}

dtype_to_mult_flatten = {
    np.dtype('int16'): mult_client_flatten_16_bits,
    np.dtype('int64'): mult_client_flatten_64_bits,
}


def mult_client_numba(x, y, c, m, e0, e1, f0, f1):
    """Efficiently compute:
    E = E_share_server + E_share
    F = F_share_server + F_share
    out = X_share * F + Y_share * E + C_share
    out = out + mu_0
    """

    if x.ndim == 1:
        return dtype_to_mult_flatten[x.dtype](x, y, c, m, e0, e1, f0, f1)
    else:
        return dtype_to_mult_non_flatten[x.dtype](x, y, c, m, e0, e1, f0, f1)


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
    # @timer(name='client_conv2d')
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
        # self.decompose = Decompose(ignore_msb_bits=IGNORE_MSB_BITS, num_of_compare_bits=NUM_OF_COMPARE_BITS, dtype=SIGNED_DTYPE, **kwargs)

    def forward(self, x_bits_0, r, beta):

        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=P, size=x_bits_0.shape, dtype=backend.int8)
        d_bits_0 = dtype_to_private_compare[r.dtype](s, r, x_bits_0, beta)
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
        return dtype_to_post_compare[a_0.dtype](a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0)
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

        dtype = a_0.dtype
        shape = a_0.shape
        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max
        num_bits = DTYPE_TO_BITS[dtype]

        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=shape, dtype=backend.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(min_val, max_val + 1, size=shape, dtype=dtype)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(min_val, max_val + 1, size=shape, dtype=dtype)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(min_val, max_val, size=shape, dtype=dtype)
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [num_bits], dtype=backend.int8)

        alpha = backend.astype(0 < r_0 - r, dtype)
        a_tild_0 = a_0 + r_0
        self.network_assets.sender_02.put(a_tild_0)

        beta_0 = backend.astype(0 < a_0 - a_tild_0, dtype)
        delta_0 = self.network_assets.receiver_02.get()

        self.private_compare(x_bits_0, r - 1, eta_pp)
        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(min_val, max_val, size=shape, dtype=dtype)

        return self.post_compare(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0)


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

        dtype = X_share.dtype
        shape = X_share.shape
        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(min_val, max_val + 1, size=shape, dtype=dtype)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(min_val, max_val + 1, size=shape, dtype=dtype)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(min_val, max_val, size=shape, dtype=dtype)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        E_share_server, F_share_server = self.exchange_shares(E_share, F_share)

        C_share = self.network_assets.receiver_02.get()
        out = mult_client_numba(X_share, Y_share, C_share, mu_0, E_share, E_share_server, F_share, F_share_server)

        return out


class SecureMSBClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBClient, self).__init__(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.private_compare = PrivateCompareClient(**kwargs)

    def post_compare(self, beta, x_bit_0_0, r_mode_2, mu_0):

        dtype = mu_0.dtype
        shape = mu_0.shape
        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max

        beta = backend.astype(beta, dtype)
        beta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(min_val, max_val + 1, size=shape, dtype=dtype)

        gamma_0 = beta_p_0 - (2 * beta * beta_p_0)
        delta_0 = x_bit_0_0 - (2 * r_mode_2 * x_bit_0_0)

        theta_0 = self.mult(gamma_0, delta_0)

        alpha_0 = gamma_0 + delta_0 - 2 * theta_0
        alpha_0 = alpha_0 + mu_0

        return alpha_0

    def pre_compare(self, a_0):

        dtype = a_0.dtype
        shape = a_0.shape
        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max
        num_bits = DTYPE_TO_BITS[dtype]

        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=shape, dtype=backend.int8)
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(shape) + [num_bits], dtype=backend.int8)
        x_bit_0_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(min_val, max_val + 1, size=shape, dtype=dtype)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(min_val, max_val + 1, size=shape, dtype=dtype)

        x_0 = self.network_assets.receiver_02.get()
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
    # counter = 0
    def __init__(self, **kwargs):
        super(SecureDReLUClient, self).__init__(**kwargs)

        self.share_convert = ShareConvertClient(**kwargs)
        self.msb = SecureMSBClient(**kwargs)

    def forward(self, X_share):
        # SecureDReLUClient.counter += 1
        # np.save("/home/yakir/Data2/secure_activation_statistics/client/{}.npy".format(SecureDReLUClient.counter), X_share)

        dtype = X_share.dtype
        shape = X_share.shape
        min_val = np.iinfo(dtype).min
        max_val = np.iinfo(dtype).max

        mu_0 = self.prf_handler[CLIENT, SERVER].integers(min_val, max_val + 1, size=shape, dtype=dtype)

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
            dtype = X_share.dtype
            shape = X_share.shape
            min_val = np.iinfo(dtype).min
            max_val = np.iinfo(dtype).max

            mu_0 = self.prf_handler[CLIENT, SERVER].integers(min_val, max_val + 1, size=shape, dtype=dtype)

            X_share = X_share.flatten()
            MSB_0 = self.DReLU(X_share)
            ret = self.mult(X_share, MSB_0).reshape(shape)

            return ret + mu_0


class SecurePostBReLUMultClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecurePostBReLUMultClient, self).__init__(**kwargs)

    def forward(self, activation, sign_tensors, cumsum_shapes,  pad_handlers, active_block_sizes, active_block_sizes_to_channels):

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=activation.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=sign_tensors.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=activation.shape, dtype=SIGNED_DTYPE)

        E_share = activation - A_share
        F_share = sign_tensors - B_share

        self.network_assets.sender_01.put(E_share)
        E_share_server = self.network_assets.receiver_01.get()

        self.network_assets.sender_01.put(F_share)
        F_share_server = self.network_assets.receiver_01.get()

        C_share = self.network_assets.receiver_02.get()
        E = E_share_server + E_share
        F = F_share_server + F_share

        F = unpack_bReLU(activation, F, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels)
        sign_tensors = unpack_bReLU(activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels)

        out = activation * F + sign_tensors * E + C_share

        out = out + mu_0

        return out


class SecureBlockReLUClient(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.post_bReLU = SecurePostBReLUMultClient(**kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, activation):
        if self.dummy_relu:
            return activation
        return SecureOptimizedBlockReLU.forward(self, activation)




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


class SecureMaxPoolClient(SecureMaxPool):
    def __init__(self, kernel_size=3, stride=2, padding=1,  **kwargs):
        super(SecureMaxPoolClient, self).__init__(kernel_size, stride, padding,  **kwargs)
        self.select_share = SecureSelectShareClient(**kwargs)
        self.dReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)

    def forward(self, x):

        ret = super(SecureMaxPoolClient, self).forward(x)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)

        return ret + mu_0











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


class PRFFetcherPostBReLUMultiplication(SecureModule):
    def __init__(self, **kwargs):
        super(PRFFetcherPostBReLUMultiplication, self).__init__(**kwargs)

    def forward(self, activation_shape, sign_tensors_shape):
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=activation_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=sign_tensors_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=activation_shape, dtype=SIGNED_DTYPE)

        return


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherMultiplication, self).__init__(**kwars)

    def forward(self, shape):

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=shape, dtype=SIGNED_DTYPE)

        return


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
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        self.private_compare(shape)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
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
        super(PRFFetcherMaxPool, self).__init__(**kwargs)
        self.select_share = PRFFetcherSelectShare(**kwargs)
        self.dReLU = PRFFetcherDReLU(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)

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
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.secure_DReLU = PRFFetcherDReLU(**kwargs)
        self.secure_mult = PRFFetcherPostBReLUMultiplication(**kwargs)

        self.dummy_relu = dummy_relu

    def forward(self, shape):
        if self.dummy_relu:
            return shape

        if not np.all(self.block_sizes == [0, 1]):
            mean_tensor_shape = (int(sum(np.ceil(shape[2] / block_size[0]) * np.ceil(shape[3] / block_size[1]) for block_size in self.block_sizes if 0 not in block_size)),)
            mult_shape = shape[0], sum(~self.is_identity_channels), shape[2], shape[3]

            self.secure_DReLU(mean_tensor_shape)
            self.secure_mult(mult_shape, mean_tensor_shape)

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