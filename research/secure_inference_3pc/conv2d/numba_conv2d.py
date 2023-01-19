import numpy as np
import torch
import time
from research.secure_inference_3pc.timer import Timer
from numba import njit, prange, int64, int32
from research.secure_inference_3pc.const import SIGNED_DTYPE, NUM_BITS
NUMBA_MATMUL = "NUMBA_MATMUL"
NUMBA_CONV = "NUMBA_CONV"
NUMBA_DTYPE = int64 if NUM_BITS == 64 else int32
from research.secure_inference_3pc.conv2d.numba_functions import numba_functions


@njit(NUMBA_DTYPE[:, :, :](NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_double_conv2d_3x3(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((B.shape[0], nb_rows_out, nb_cols_out), dtype=SIGNED_DTYPE)

    for out_channel in prange(B.shape[0]):
        for in_channel in range(A.shape[1]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    for k_0 in range(3):
                        for k_1 in range(3):
                            x = stride_row * i + k_0 * dilation_row
                            y = stride_col * j + k_1 * dilation_col

                            res[out_channel, i, j] += \
                                B[out_channel, in_channel, k_0, k_1] * \
                                A[0, in_channel, x, y]

                            res[out_channel, i, j] += \
                                D[out_channel, in_channel, k_0, k_1] * \
                                C[0, in_channel, x, y]

    return res


@njit(NUMBA_DTYPE[:, :, :](NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True, nogil=True, cache=True)
def numba_double_separable_conv2d_3x3(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((B.shape[0], nb_rows_out, nb_cols_out), dtype=SIGNED_DTYPE)

    for out_channel in prange(B.shape[0]):
        for i in range(res.shape[1]):
            for j in range(res.shape[2]):
                for k_0 in range(3):
                    for k_1 in range(3):
                        x = stride_row * i + k_0 * dilation_row
                        y = stride_col * j + k_1 * dilation_col

                        res[out_channel, i, j] += \
                            B[out_channel, 0, k_0, k_1] * \
                            A[0, out_channel, x, y]

                        res[out_channel, i, j] += \
                            D[out_channel, 0, k_0, k_1] * \
                            C[0, out_channel, x, y]

    return res


@njit(NUMBA_DTYPE[:, :, :](NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True, nogil=True, cache=True)
def numba_single_conv2d_3x3(A, B, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((B.shape[0], nb_rows_out, nb_cols_out), dtype=SIGNED_DTYPE)

    for out_channel in prange(B.shape[0]):
        for in_channel in range(A.shape[1]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    for k_0 in range(3):
                        for k_1 in range(3):
                            x = stride_row * i + k_0 * dilation_row
                            y = stride_col * j + k_1 * dilation_col

                            res[out_channel, i, j] += \
                                B[out_channel, in_channel, k_0, k_1] * \
                                A[0, in_channel, x, y]


    return res


@njit(NUMBA_DTYPE[:, :, :](NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True, nogil=True, cache=True)
def numba_single_separable_conv2d_3x3(A, B, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((B.shape[0], nb_rows_out, nb_cols_out), dtype=SIGNED_DTYPE)

    for out_channel in prange(B.shape[0]):
        for i in range(res.shape[1]):
            for j in range(res.shape[2]):
                for k_0 in range(3):
                    for k_1 in range(3):
                        x = stride_row * i + k_0 * dilation_row
                        y = stride_col * j + k_1 * dilation_col

                        res[out_channel, i, j] += \
                            B[out_channel, 0, k_0, k_1] * \
                            A[0, out_channel, x, y]


    return res


@njit(NUMBA_DTYPE[:, :, :](NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], int32, int32, int32[:]), parallel=True, nogil=True, cache=True)
def numba_double_conv2d_1x1(A, B, C, D, nb_rows_out, nb_cols_out, stride):
    stride_row, stride_col = stride
    res = np.zeros((B.shape[0], nb_rows_out, nb_cols_out), dtype=SIGNED_DTYPE)

    for out_channel in prange(B.shape[0]):
        for in_channel in range(A.shape[1]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    x = stride_row * i
                    y = stride_col * j

                    res[out_channel, i, j] += \
                        B[out_channel, in_channel, 0, 0] * \
                        A[0, in_channel, x, y]

                    res[out_channel, i, j] += \
                        D[out_channel, in_channel, 0, 0] * \
                        C[0, in_channel, x, y]

    return res


@njit(NUMBA_DTYPE[:, :, :](NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], int32, int32, int32[:]), parallel=True, nogil=True, cache=True)
def numba_single_conv2d_1x1(A, B, nb_rows_out, nb_cols_out, stride):
    stride_row, stride_col = stride
    res = np.zeros((B.shape[0], nb_rows_out, nb_cols_out), dtype=SIGNED_DTYPE)

    for out_channel in prange(B.shape[0]):
        for in_channel in range(A.shape[1]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    x = stride_row * i
                    y = stride_col * j

                    res[out_channel, i, j] += \
                        B[out_channel, in_channel, 0, 0] * \
                        A[0, in_channel, x, y]

    return res


@njit(NUMBA_DTYPE[:, :, :](NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True, nogil=True, cache=True)
def numba_double_conv2d(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((B.shape[0], nb_rows_out, nb_cols_out), dtype=SIGNED_DTYPE)

    for out_channel in prange(B.shape[0]):
        for in_channel in range(A.shape[1]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    for k_0 in range(B.shape[2]):
                        for k_1 in range(B.shape[3]):
                            x = stride_row * i + k_0 * dilation_row
                            y = stride_col * j + k_1 * dilation_col

                            res[out_channel, i, j] += \
                                B[out_channel, in_channel, k_0, k_1] * \
                                A[0, in_channel, x, y]

                            res[out_channel, i, j] += \
                                D[out_channel, in_channel, k_0, k_1] * \
                                C[0, in_channel, x, y]

    return res


@njit(NUMBA_DTYPE[:, :, :](NUMBA_DTYPE[:, :, :, :], NUMBA_DTYPE[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True, nogil=True, cache=True)
def numba_single_conv2d(A, B, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((B.shape[0], nb_rows_out, nb_cols_out), dtype=SIGNED_DTYPE)

    for out_channel in prange(B.shape[0]):
        for in_channel in range(A.shape[1]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    for k_0 in range(B.shape[2]):
                        for k_1 in range(B.shape[3]):
                            x = stride_row * i + k_0 * dilation_row
                            y = stride_col * j + k_1 * dilation_col

                            res[out_channel, i, j] += \
                                B[out_channel, in_channel, k_0, k_1] * \
                                A[0, in_channel, x, y]


    return res


def double_conv_2d(A, B, C, D, padding, stride, dilation, groups=1):
    nb_rows_in = A.shape[2]
    nb_cols_in = A.shape[3]
    nb_rows_kernel = B.shape[2]
    nb_cols_kernel = B.shape[3]
    nb_rows_out = np.int32(
        ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0]) + 1
    )
    nb_cols_out = np.int32(
        ((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1]) + 1
    )

    A_ = np.pad(A, ((0, 0), (0, 0), padding, padding), mode='constant')
    C_ = np.pad(C, ((0, 0), (0, 0), padding, padding), mode='constant')
    stride = np.int32(stride)
    dilation = np.int32(dilation)
    if False:
        out = numba_functions[(A_.shape[1], B.shape[0], B.shape[2])](A_, B, C_, D, nb_rows_out, nb_cols_out, stride, dilation)[np.newaxis]
        return out

    if B.shape[2] == B.shape[3] == 3:
        if groups == 1:
            out = numba_double_conv2d_3x3(A_, B, C_, D, nb_rows_out, nb_cols_out, stride, dilation)
        else:
            out = numba_double_separable_conv2d_3x3(A_, B, C_, D, nb_rows_out, nb_cols_out, stride, dilation)

    elif B.shape[2] == B.shape[3] == 1:
        out = numba_double_conv2d_1x1(A_, B, C_, D, nb_rows_out, nb_cols_out, stride)
    else:
        # assert False
        out = numba_double_conv2d(A_, B, C_, D, nb_rows_out, nb_cols_out, stride, dilation)
    return out[np.newaxis]

x = set()
def single_conv_2d(A, B, padding, stride, dilation, groups=1):
    nb_rows_in = A.shape[2]
    nb_cols_in = A.shape[3]
    nb_rows_kernel = B.shape[2]
    nb_cols_kernel = B.shape[3]
    nb_rows_out = np.int32(
        ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0]) + 1
    )
    nb_cols_out = np.int32(
        ((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1]) + 1
    )

    A_ = np.pad(A, ((0, 0), (0, 0), padding, padding), mode='constant')
    stride = np.int32(stride)
    dilation = np.int32(dilation)
    if B.shape[2] == B.shape[3] == 3:
        if groups == 1:
            out = numba_single_conv2d_3x3(A_, B, nb_rows_out, nb_cols_out, stride, dilation)
        else:
            out = numba_single_separable_conv2d_3x3(A_, B, nb_rows_out, nb_cols_out, stride, dilation)
    elif B.shape[2] == B.shape[3] == 1:
        out = numba_single_conv2d_1x1(A_, B, nb_rows_out, nb_cols_out, stride)
    else:
        # assert False
        out = numba_single_conv2d(A_, B, nb_rows_out, nb_cols_out, stride, dilation)
    return out[np.newaxis]




class Conv2DHandler:
    def __init__(self):
        pass

    def conv2d(self, A, B, C=None, D=None, padding=(1, 1), stride=(1, 1), dilation=(1, 1), groups=1, method=NUMBA_CONV):
        if type(padding) is int:
            padding = (padding, padding)
        if type(stride) is int:
            stride = (stride, stride)
        if type(dilation) is int:
            dilation = (dilation, dilation)

        if C is None:
            out = single_conv_2d(A, B, padding, stride, dilation, groups=groups)
        else:
            out = double_conv_2d(A, B, C, D, padding, stride, dilation, groups=groups)

        return out

# # TODO: put in init
# def compile_numba_funcs():
#     conv2d_handler = Conv2DHandler()
#     in_channel = 512
#     out_channel = 128
#     dilation = (1, 1)
#     padding = (1, 1)
#     stride = (1, 1)
#     nb_rows = 24
#
#     A = np.random.randint(np.iinfo(SIGNED_DTYPE).min, np.iinfo(SIGNED_DTYPE).max, size=(1, in_channel, nb_rows, nb_rows))
#     B_3x3 = np.random.randint(np.iinfo(SIGNED_DTYPE).min, np.iinfo(SIGNED_DTYPE).max, size=(out_channel, in_channel, 3, 3))
#     B_1x1 = np.random.randint(np.iinfo(SIGNED_DTYPE).min, np.iinfo(SIGNED_DTYPE).max, size=(out_channel, in_channel, 1, 1))
#     C = np.random.randint(np.iinfo(SIGNED_DTYPE).min, np.iinfo(SIGNED_DTYPE).max, size=(1, in_channel, nb_rows, nb_rows))
#     D_3x3 = np.random.randint(np.iinfo(SIGNED_DTYPE).min, np.iinfo(SIGNED_DTYPE).max, size=(out_channel, in_channel, 3, 3))
#     D_1x1 = np.random.randint(np.iinfo(SIGNED_DTYPE).min, np.iinfo(SIGNED_DTYPE).max, size=(out_channel, in_channel, 1, 1))
#
#     conv2d_handler.conv_2d(A, B_3x3, C, D_3x3, padding=padding, stride=stride, dilation=dilation, method=NUMBA_CONV)
#     conv2d_handler.conv_2d(A, B_3x3, padding=padding, stride=stride, dilation=dilation, method=NUMBA_CONV)
#     conv2d_handler.conv_2d(A, B_3x3[:, :1], C, D_3x3, padding=padding, stride=stride, dilation=dilation, method=NUMBA_CONV, groups=B_3x3.shape[0])
#     conv2d_handler.conv_2d(A, B_3x3[:, :1], padding=padding, stride=stride, dilation=dilation, method=NUMBA_CONV, groups=B_3x3.shape[0])
#     conv2d_handler.conv_2d(A, B_1x1, C, D_1x1, padding=padding, stride=stride, dilation=dilation, method=NUMBA_CONV)
#     conv2d_handler.conv_2d(A, B_1x1, padding=padding, stride=stride, dilation=dilation, method=NUMBA_CONV)
#     conv2d_handler.conv_2d(A, B_3x3, C, D_3x3, padding=padding, stride=stride, dilation=dilation, method=NUMBA_MATMUL)
#     conv2d_handler.conv_2d(A, B_3x3, padding=padding, stride=stride, dilation=dilation, method=NUMBA_MATMUL)
#
#
# # TODO: allocate one array and reuse it
# # TODO: unravel stuff
#
# if __name__ == "__main__":
#
#     compile_numba_funcs()
