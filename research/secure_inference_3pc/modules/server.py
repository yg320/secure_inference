from research.secure_inference_3pc.backend import backend

from research.secure_inference_3pc.modules.base import PRFFetcherModule
from research.secure_inference_3pc.conv2d.utils import get_output_shape
from research.secure_inference_3pc.const import NUM_OF_COMPARE_BITS, IGNORE_MSB_BITS

from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.base import get_c_party_1, module_67
from research.secure_inference_3pc.conv2d.conv2d_handler_factory import conv2d_handler_factory
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, TRUNC_BITS
from research.bReLU import SecureOptimizedBlockReLU
from research.secure_inference_3pc.modules.base import Decompose
from research.secure_inference_3pc.modules.base import DummyShapeTensor
from research.secure_inference_3pc.timer import Timer, timer

import torch
import numpy as np

# from numba import njit, prange, int64, int32
# from research.secure_inference_3pc.const import SIGNED_DTYPE, NUM_BITS
# NUMBA_DTYPE = int64 if NUM_BITS == 64 else int32
#
# @njit(NUMBA_DTYPE[:, :, :](NUMBA_DTYPE[:, :, :, :],
#                            NUMBA_DTYPE[:, :, :, :],
#                            NUMBA_DTYPE[:, :, :, :],
#                            NUMBA_DTYPE[:, :, :, :],
#                            int32,
#                            int32,
#                            int32[:],
#                            int32[:]), parallel=True, nogil=True, cache=True)
# def numba_double_conv2d(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):
#     stride_row, stride_col = stride
#     dilation_row, dilation_col = dilation
#     res = np.zeros((B.shape[0], nb_rows_out, nb_cols_out), dtype=SIGNED_DTYPE)
#
#     for out_channel in prange(B.shape[0]):
#         for in_channel in range(A.shape[1]):
#             for i in range(res.shape[1]):
#                 for j in range(res.shape[2]):
#                     for k_0 in range(B.shape[2]):
#                         for k_1 in range(B.shape[3]):
#                             x = stride_row * i + k_0 * dilation_row
#                             y = stride_col * j + k_1 * dilation_col
#
#                             res[out_channel, i, j] += \
#                                 B[out_channel, in_channel, k_0, k_1] * \
#                                 A[0, in_channel, x, y]
#
#                             res[out_channel, i, j] += \
#                                 D[out_channel, in_channel, k_0, k_1] * \
#                                 C[0, in_channel, x, y]
#
#     return res
#
#
# def double_conv_2d(A, B, C, D, padding, stride, dilation, groups=1):
#     with Timer("conv2d - preprocessing"):
#
#         nb_rows_in = A.shape[2]
#         nb_cols_in = A.shape[3]
#         nb_rows_kernel = B.shape[2]
#         nb_cols_kernel = B.shape[3]
#         nb_rows_out = np.int32(
#             ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0]) + 1
#         )
#         nb_cols_out = np.int32(
#             ((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1]) + 1
#         )
#         A_ = np.pad(A, ((0, 0), (0, 0), padding, padding), mode='constant')
#         C_ = np.pad(C, ((0, 0), (0, 0), padding, padding), mode='constant')
#         stride = np.int32(stride)
#     dilation = np.int32(dilation)
#
#     out = numba_double_conv2d(A_, B, C_, D, nb_rows_out, nb_cols_out, stride, dilation)
#     return out[np.newaxis]

from numba import njit, prange, int64, uint64


@njit('(int8[:,:])(int8[:,:], int64[:], int8[:,:], int8[:],  uint8,  uint8)', parallel=True, nogil=True,
      cache=True)
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

    @timer("SecureConv2DServer")
    def forward(self, X_share):

        # out_shape = get_output_shape(X_share.shape, self.W_plaintext.shape, self.padding, self.dilation, self.stride)
        # return backend.zeros(out_shape, dtype=X_share.dtype)
        # self.network_assets.sender_12.put(np.arange(10))
        # self.network_assets.receiver_01.get()

        with Timer("conv2d - preprocessing"):
            assert self.W_plaintext.shape[2] == self.W_plaintext.shape[3]
            assert (self.W_plaintext.shape[1] == X_share.shape[1]) or self.groups > 1
            assert self.stride[0] == self.stride[1]
            with Timer("conv2d - preprocessing - prf"):

                W_client = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, size=self.W_plaintext.shape, dtype=SIGNED_DTYPE)
                A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
                B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_plaintext.shape, dtype=SIGNED_DTYPE)

            with Timer("conv2d - preprocessing - subtract"):

                W_share = backend.subtract(self.W_plaintext, W_client, out=W_client)
                E_share = backend.subtract(X_share, A_share, out=A_share)
                F_share = backend.subtract(W_share, B_share, out=B_share)
            with Timer("conv2d - preprocessing -put"):
                self.network_assets.sender_01.put(E_share)
                self.network_assets.sender_01.put(F_share)

            with Timer("conv2d - preprocessing -get"):

                E_share_client = self.network_assets.receiver_01.get()
                F_share_client = self.network_assets.receiver_01.get()

            with Timer("conv2d - preprocessing -add"):
                E = backend.add(E_share_client, E_share, out=E_share)
                F = backend.add(F_share_client, F_share, out=F_share)

                W_share = backend.subtract(W_share, F, out=W_share)

        with Timer("conv2d - main"):
            out = self.conv2d_handler.conv2d(E, W_share, X_share, F, self.padding, self.stride, self.dilation, self.groups)
        with Timer("conv2d - post"):

            C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)

            out = backend.add(out, C_share, out=out)
            out = backend.right_shift(out, TRUNC_BITS, out=out)

            if self.bias is not None:
                out = backend.add(out, self.bias, out=out)

            mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)
            mu_1 = backend.multiply(mu_1, -1, out=mu_1)
            out = backend.add(out, mu_1, out=out)

        # self.network_assets.sender_12.put(np.arange(10))
        # self.network_assets.receiver_01.get()

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


@njit('(int64[:])(int64[:], int8[:], int64[:], int64[:], int64[:], int64[:])', parallel=True,  nogil=True, cache=True)
def post_compare_numba(a_1, eta_pp, delta_1,  beta_1, mu_0, eta_p_1):

    out = a_1
    for i in prange(a_1.shape[0],):
        # eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)
        eta_pp_t = int64(eta_pp[i])

        # t00 = backend.multiply(eta_pp, eta_p_1, out=eta_pp)
        t00 = eta_pp_t * eta_p_1[i]


        # t11 = self.add_mode_L_minus_one(t00, t00)
        t11 = t00 + t00
        if uint64(t00) > uint64(t11):
            t11 += 1
        if t11 == -1:
            t11 = 0


        # eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)
        if uint64(t11) > uint64(eta_p_1[i]):
            eta_1 = eta_p_1[i] - t11 - 1
        else:
            eta_1 = eta_p_1[i] - t11



        # t00 = self.add_mode_L_minus_one(delta_1, eta_1)
        t00 = delta_1[i] + eta_1
        if uint64(delta_1[i]) > uint64(t00):
            t00 += 1
        if t00 == -1:
            t00 = 0


        # theta_1 = self.add_mode_L_minus_one(beta_1, t00)
        theta_1 = beta_1[i] + t00
        if uint64(t00) > uint64(theta_1):
            theta_1 += 1
        if theta_1 == -1:
            theta_1 = 0


        # y_1 = self.sub_mode_L_minus_one(a_1, theta_1)
        if uint64(theta_1) > uint64(a_1[i]):
            y_1 = a_1[i] - theta_1 - 1
        else:
            y_1 = a_1[i] - theta_1


        # y_1 = self.add_mode_L_minus_one(y_1, mu_1)
        ret = y_1 - mu_0[i]
        if uint64(y_1) > uint64(ret):
            ret += 1
        if ret == -1:
            ret = 0

        out[i] = ret
    return out


class ShareConvertServer(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertServer, self).__init__(**kwargs)
        self.private_compare = PrivateCompareServer(**kwargs)

    def forward(self, a_1):
        # with Timer("ShareConvertServer - processing..."):
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

        with Timer("post compare server"):
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

@njit('(int64[:])(int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], int64[:])', parallel=True,  nogil=True, cache=True)
def mult_server_flatten(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0],):
        e = (e0[i] + e1[i])
        f = (f0[i] + f1[i])
        f1[i] = - e * f + x[i] * f + y[i] * e + c[i] - m[i]
    return f1

@njit('(int64[:, :, :, :])(int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :])', parallel=True,  nogil=True, cache=True)
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

def mult_server_numba(x, y, c, m, e0, e1, f0, f1):
    if x.ndim == 1:
        return mult_server_flatten(x, y, c, m, e0, e1, f0, f1)
    else:
        return mult_server_non_flatten(x, y, c, m, e0, e1, f0, f1)

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
    def __init__(self,  **kwargs):
        super(SecureDReLUServer, self).__init__( **kwargs)

        self.share_convert = ShareConvertServer( **kwargs)
        self.msb = SecureMSBServer(**kwargs)

    def forward(self, X_share):
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
        if self.dummy_relu:
            share_client = self.network_assets.receiver_01.get()
            recon = share_client + X_share
            value = recon * (backend.astype(recon > 0, recon.dtype))
            return value
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
    def __init__(self, kernel_size, stride, padding, dummy_max_pool,  **kwargs):
        super(SecureMaxPoolServer, self).__init__(kernel_size, stride, padding, dummy_max_pool,  **kwargs)
        self.select_share = SecureSelectShareServer( **kwargs)
        self.dReLU = SecureDReLUServer( **kwargs)
        self.mult = SecureMultiplicationServer( **kwargs)

    def forward(self, x):
        if self.dummy_max_pool:
            x_client = self.network_assets.receiver_01.get()
            x_rec = x_client + x
            if type(x) is torch.Tensor:
                return torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)(x_rec.to(torch.float64)).to(x.dtype)
            else:
                return torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)(torch.from_numpy(x_rec).to(torch.float64)).numpy().astype(x.dtype)
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

    def forward(self, shape):

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
        return shape


class PRFFetcherMaxPool(PRFFetcherModule):
    def __init__(self, kernel_size=3, stride=2, padding=1, dummy_max_pool=False, **kwargs):
        super(PRFFetcherMaxPool, self).__init__( **kwargs)

        self.dummy_max_pool = dummy_max_pool
        self.select_share = PRFFetcherSelectShare( **kwargs)
        self.dReLU = PRFFetcherDReLU( **kwargs)
        self.mult = PRFFetcherMultiplication( **kwargs)

    def forward(self, shape):
        if self.dummy_max_pool:
            print(shape, shape, shape)
            return DummyShapeTensor((shape[0], shape[1], shape[2] // 2, shape[3] // 2))

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
        assert False, "send img"
        shape = img.shape
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=shape, dtype=SIGNED_DTYPE)
        out_0 = self.model.decode_head(self.model.backbone(backend.zeros(shape=img, dtype=SIGNED_DTYPE)))


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