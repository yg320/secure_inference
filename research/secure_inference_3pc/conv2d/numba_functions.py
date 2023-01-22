import numpy as np
from numba import njit, prange, int64, int32


numba_functions = dict()

# for input_channels, output_channels, kernel_size in [(64, 256, 1), (256, 64, 1), (1024, 2048, 1), (1024, 256, 1), (2048, 512, 1), (128, 128, 3), (512, 512, 3), (256, 256, 3), (64, 64, 3), (512, 1024, 1), (1024, 512, 1), (512, 2048, 1), (64, 64, 1), (256, 128, 1), (128, 512, 1), (256, 512, 1), (512, 128, 1), (512, 256, 1), (256, 1024, 1), (3, 64, 7), (2048, 1000, 1)]:
@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_0(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride
    res = np.zeros((256, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(256):
        for in_channel in range(64):
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
numba_functions[(64, 256, 1)] = numba_func_0


@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_1(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride
    res = np.zeros((64, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(64):
        for in_channel in range(256):
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
numba_functions[(256, 64, 1)] = numba_func_1

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_2(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    res = np.zeros((2048, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(2048):
        for in_channel in range(1024):
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
numba_functions[(1024, 2048, 1)] = numba_func_2

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_3(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    res = np.zeros((256, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(256):
        for in_channel in range(1024):
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
numba_functions[(1024, 256, 1)] = numba_func_3

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_4(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    res = np.zeros((512, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(512):
        for in_channel in range(2048):
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
numba_functions[(2048, 512, 1)] = numba_func_4

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_5(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((128, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(128):
        for in_channel in range(128):
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
numba_functions[(128, 128, 3)] = numba_func_5
#
@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_6(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((512, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(512):
        for in_channel in range(512):
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
numba_functions[(512, 512, 3)] = numba_func_6

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_7(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((256, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(256):
        for in_channel in range(256):
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
numba_functions[(256, 256, 3)] = numba_func_7

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_8(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((64, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(64):
        for in_channel in range(64):
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
numba_functions[(64, 64, 3)] = numba_func_8
#
@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_9(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride

    res = np.zeros((1024, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(1024):
        for in_channel in range(512):
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
numba_functions[ (512, 1024, 1)] = numba_func_9

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_10(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride

    res = np.zeros((512, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(512):
        for in_channel in range(1024):
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
numba_functions[(1024, 512, 1)] = numba_func_10

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_11(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride

    res = np.zeros((2048, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(2048):
        for in_channel in range(512):
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
numba_functions[(512, 2048, 1)] = numba_func_11

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_12(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride

    res = np.zeros((64, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(64):
        for in_channel in range(64):
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
numba_functions[(64, 64, 1)] = numba_func_12

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_13(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride

    res = np.zeros((128, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(128):
        for in_channel in range(256):
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
numba_functions[(256, 128, 1)] = numba_func_13

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_14(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride

    res = np.zeros((512, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(512):
        for in_channel in range(128):
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
numba_functions[ (128, 512, 1)] = numba_func_14

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_15(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    res = np.zeros((512, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(512):
        for in_channel in range(256):
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
numba_functions[(256, 512, 1)] = numba_func_15

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_16(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride

    res = np.zeros((128, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(128):
        for in_channel in range(512):
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
numba_functions[(512, 128, 1)] = numba_func_16

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_17(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):
    stride_row, stride_col = stride
    res = np.zeros((256, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(256):
        for in_channel in range(512):
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
numba_functions[(512, 256, 1)] = numba_func_17

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_18(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride

    res = np.zeros((1024, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(1024):
        for in_channel in range(256):
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
numba_functions[(256, 1024, 1)] = numba_func_18

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_19(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride
    dilation_row, dilation_col = dilation
    res = np.zeros((64, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(64):
        for in_channel in range(3):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    for k_0 in range(7):
                        for k_1 in range(7):
                            x = stride_row * i + k_0 * dilation_row
                            y = stride_col * j + k_1 * dilation_col

                            res[out_channel, i, j] += \
                                B[out_channel, in_channel, k_0, k_1] * \
                                A[0, in_channel, x, y]

                            res[out_channel, i, j] += \
                                D[out_channel, in_channel, k_0, k_1] * \
                                C[0, in_channel, x, y]

    return res
numba_functions[(3, 64, 7)] = numba_func_19

@njit(int64[:, :, :](int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int64[:, :, :, :], int32, int32, int32[:], int32[:]), parallel=True,  nogil=True, cache=True)
def numba_func_20(A, B, C, D, nb_rows_out, nb_cols_out, stride, dilation):

    stride_row, stride_col = stride
    res = np.zeros((1000, nb_rows_out, nb_cols_out), dtype=np.int64)

    for out_channel in prange(1000):
        for in_channel in range(2048):
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
numba_functions[(2048, 1000, 1)] = numba_func_20
