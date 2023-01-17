import numpy as np
import torch
import time
from numba import njit, prange, int64, int32
from research.secure_inference_3pc.const import SIGNED_DTYPE, NUM_BITS
NUMBA_MATMUL = "NUMBA_MATMUL"
NUMBA_CONV = "NUMBA_CONV"
NUMBA_DTYPE = int64 if NUM_BITS == 64 else int32

# TODO: replace redundancy in code
def get_output_shape(A, B, padding, dilation, stride):
    nb_rows_in = A.shape[2]
    nb_cols_in = A.shape[3]
    nb_rows_kernel = B.shape[2]
    nb_cols_kernel = B.shape[3]

    nb_rows_out = int(
        ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0]) + 1
    )
    nb_cols_out = int(
        ((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1]) + 1
    )

    return 1, B.shape[0], nb_rows_out, nb_cols_out


def pre_conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    This is a block of local computation done at the beginning of the convolution. It
    basically does the matrix unrolling to be able to do the convolution as a simple
    matrix multiplication.

    Because all the computation are local, we add the @allow_command and run it directly
    on each share of the additive sharing tensor, when running mpc computations
    """
    # print(input.shape, weight.shape, stride, padding, dilation, groups)
    assert len(input.shape) == 4
    assert len(weight.shape) == 4

    # Change to tuple if not one
    stride = (stride, stride) if type(stride) is int else stride
    padding = (padding, padding) if type(padding) is int else padding
    dilation = (dilation, dilation) if type(dilation) is int else dilation

    # Extract a few useful values
    batch_size, nb_channels_in, nb_rows_in, nb_cols_in = input.shape
    nb_channels_out, nb_channels_kernel, nb_rows_kernel, nb_cols_kernel = weight.shape

    if bias is not None:
        assert len(bias) == nb_channels_out

    # Check if inputs are coherent
    assert nb_channels_in == nb_channels_kernel * groups
    assert nb_channels_in % groups == 0
    assert nb_channels_out % groups == 0

    # Compute output shape
    nb_rows_out = int(
        ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0]) + 1
    )
    nb_cols_out = int(
        ((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1]) + 1
    )

    # Apply padding to the input
    if padding != (0, 0):
        padding_mode = "constant"
        input = np.pad(input, ((0, 0), (0, 0), (padding[1], padding[1]), (padding[0], padding[0])), mode='constant')
        # Update shape after padding
        nb_rows_in += 2 * padding[0]
        nb_cols_in += 2 * padding[1]

    # We want to get relative positions of values in the input tensor that are used
    # by one filter convolution.
    # It basically is the position of the values used for the top left convolution.
    pattern_ind = []
    for ch in range(nb_channels_in):
        for r in range(nb_rows_kernel):
            for c in range(nb_cols_kernel):
                pixel = r * nb_cols_in * dilation[0] + c * dilation[1]
                pattern_ind.append(pixel + ch * nb_rows_in * nb_cols_in)

    # The image tensor is reshaped for the matrix multiplication:
    # on each row of the new tensor will be the input values used for each filter convolution
    # We will get a matrix [[in values to compute out value 0],
    #                       [in values to compute out value 1],
    #                       ...
    #                       [in values to compute out value nb_rows_out*nb_cols_out]]
    pattern_ind = np.array(pattern_ind)
    im_flat = input.reshape(batch_size, -1)
    im_reshaped = []
    for cur_row_out in range(nb_rows_out):
        for cur_col_out in range(nb_cols_out):
            # For each new output value, we just need to shift the receptive field
            offset = cur_row_out * stride[0] * nb_cols_in + cur_col_out * stride[1]
            tmp = offset + pattern_ind
            im_reshaped.append(im_flat[:, tmp])
    im_reshaped = np.stack(im_reshaped).transpose(1, 0, 2)

    # The convolution kernels are also reshaped for the matrix multiplication
    # We will get a matrix [[weights for out channel 0],
    #                       [weights for out channel 1],
    #                       ...
    #                       [weights for out channel nb_channels_out]].TRANSPOSE()
    weight_reshaped = weight.reshape(nb_channels_out // groups, -1).T

    return (
        im_reshaped.copy(),
        weight_reshaped.copy(),
        batch_size,
        nb_channels_out,
        nb_rows_out,
        nb_cols_out,
    )


def post_conv(bias, res, batch_size, nb_channels_out, nb_rows_out, nb_cols_out):
    """
    This is a block of local computation done at the end of the convolution. It
    basically reshape the matrix back to the shape it should have with a regular
    convolution.

    Because all the computation are local, we add the @allow_command and run it directly
    on each share of the additive sharing tensor, when running mpc computations
    """
    # batch_size, nb_channels_out, nb_rows_out, nb_cols_out = (
    #     batch_size.item(),
    #     nb_channels_out.item(),
    #     nb_rows_out.item(),
    #     nb_cols_out.item(),
    # )
    # Add a bias if needed
    if bias is not None:
        if bias.is_wrapper and res.is_wrapper:
            res += bias
        elif bias.is_wrapper:
            res += bias.child
        else:
            res += bias

    # ... And reshape it back to an image
    res = (
        res.transpose(0, 2, 1)
        .reshape(batch_size, nb_channels_out, nb_rows_out, nb_cols_out)
    )

    return res


@njit(NUMBA_DTYPE[:, :](NUMBA_DTYPE[:, :], NUMBA_DTYPE[:, :], NUMBA_DTYPE[:, :], NUMBA_DTYPE[:, :]), parallel=True,  nogil=True, cache=True)
def numba_double_mat_mult(a, b, c, d):
    assert a.shape[1] == b.shape[0]
    res = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)

    for i in prange(a.shape[0]):
        for k in range(a.shape[1]):
            for j in range(b.shape[1]):
                res[i, j] += a[i, k] * b[k, j]
                res[i, j] += c[i, k] * d[k, j]
    return res


@njit(NUMBA_DTYPE[:, :](NUMBA_DTYPE[:, :], NUMBA_DTYPE[:, :]), parallel=True,  nogil=True, cache=True)
def numba_single_mat_mult(a, b):
    assert a.shape[1] == b.shape[0]
    res = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)

    for i in prange(a.shape[0]):
        for k in range(a.shape[1]):
            for j in range(b.shape[1]):
                res[i, j] += a[i, k] * b[k, j]
    return res


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


def double_conv_2d_mat_mul(A, B, C, D, padding, stride, dilation):
    A, B, batch_size, nb_channels_out, nb_rows_out, nb_cols_out = pre_conv(A, B, padding=padding, stride=stride,
                                                                           dilation=dilation)
    C, D, *_ = pre_conv(C, D, padding=padding, stride=stride, dilation=dilation)
    E = numba_double_mat_mult(A[0], B, C[0], D)[np.newaxis]
    E = post_conv(None, E, batch_size, nb_channels_out, nb_rows_out, nb_cols_out)

    return E


def single_conv_2d_mat_mul(A, B, padding, stride, dilation):
    A, B, batch_size, nb_channels_out, nb_rows_out, nb_cols_out = pre_conv(A, B, padding=padding, stride=stride,
                                                                           dilation=dilation)
    E = numba_single_mat_mult(A[0], B)[np.newaxis]
    E = post_conv(None, E, batch_size, nb_channels_out, nb_rows_out, nb_cols_out)

    return E

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
        if type(A) is not np.ndarray:
            is_torch = True
        else:
            is_torch = False

        if is_torch:
            A = A.numpy()
            B = B.numpy()
            if C is not None:
                C = C.numpy()
                D = D.numpy()

        if groups == 1:
            if method == NUMBA_CONV:
                if C is None:
                    out = single_conv_2d(A, B, padding, stride, dilation)
                else:
                    out = double_conv_2d(A, B, C, D, padding, stride, dilation)
            else:
                if C is None:
                    out =  single_conv_2d_mat_mul(A, B, padding, stride, dilation)
                else:
                    out = double_conv_2d_mat_mul(A, B, C, D, padding, stride, dilation)
        else:
            assert method == NUMBA_CONV
            assert groups == B.shape[0]
            if C is None:
                out = single_conv_2d(A, B, padding, stride, dilation, groups=groups)
            else:
                out = double_conv_2d(A, B, C, D, padding, stride, dilation, groups=groups)

        if is_torch:
            out = torch.from_numpy(out)
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
