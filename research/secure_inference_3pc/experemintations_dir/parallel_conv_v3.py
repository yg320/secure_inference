import numpy as np

import time
from numba import njit, prange



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



@njit(parallel=True)
def mat_mult(A, B, C, D):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]))

    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i, j] += A[i, k] * B[k, j]
                res[i, j] += C[i, k] * D[k, j]
    return res

@njit(parallel=True)
def conv(A, B, C, D):

    res = np.zeros((B.shape[0], A.shape[2] - 2, A.shape[3] - 2), dtype=np.int64)

    for out_channel in prange(B.shape[0]):
        for in_channel in range(A.shape[1]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    res[out_channel, i, j] += (
                            (B[out_channel, in_channel, 0, 0] * A[0, in_channel, i + 0, j + 0]) +
                            (B[out_channel, in_channel, 0, 1] * A[0, in_channel, i + 0, j + 1]) +
                            (B[out_channel, in_channel, 0, 2] * A[0, in_channel, i + 0, j + 2]) +
                            (B[out_channel, in_channel, 1, 0] * A[0, in_channel, i + 1, j + 0]) +
                            (B[out_channel, in_channel, 1, 1] * A[0, in_channel, i + 1, j + 1]) +
                            (B[out_channel, in_channel, 1, 2] * A[0, in_channel, i + 1, j + 2]) +
                            (B[out_channel, in_channel, 2, 0] * A[0, in_channel, i + 2, j + 0]) +
                            (B[out_channel, in_channel, 2, 1] * A[0, in_channel, i + 2, j + 1]) +
                            (B[out_channel, in_channel, 2, 2] * A[0, in_channel, i + 2, j + 2]) +
                            (D[out_channel, in_channel, 0, 0] * C[0, in_channel, i + 0, j + 0]) +
                            (D[out_channel, in_channel, 0, 1] * C[0, in_channel, i + 0, j + 1]) +
                            (D[out_channel, in_channel, 0, 2] * C[0, in_channel, i + 0, j + 2]) +
                            (D[out_channel, in_channel, 1, 0] * C[0, in_channel, i + 1, j + 0]) +
                            (D[out_channel, in_channel, 1, 1] * C[0, in_channel, i + 1, j + 1]) +
                            (D[out_channel, in_channel, 1, 2] * C[0, in_channel, i + 1, j + 2]) +
                            (D[out_channel, in_channel, 2, 0] * C[0, in_channel, i + 2, j + 0]) +
                            (D[out_channel, in_channel, 2, 1] * C[0, in_channel, i + 2, j + 1]) +
                            (D[out_channel, in_channel, 2, 2] * C[0, in_channel, i + 2, j + 2])
                    )

    return res




for i in range(10):
    A = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(1, 640, 24, 24))
    B = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(128, 640, 3, 3))
    C = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(1, 640, 24, 24))
    D = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(128, 640, 3, 3))

    t0 = time.time()
    A = np.pad(A, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
    C = np.pad(C, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
    res = conv(A, B, C, D)
    print(time.time() - t0)

    # A, B, batch_size, nb_channels_out, nb_rows_out, nb_cols_out = pre_conv(A, B, padding=(1, 1))
    # C, D, *_ = pre_conv(C, D, padding=(1, 1))
    # E = mat_mult(A[0], B, C[0], D)
    # E = post_conv(None, E[np.newaxis], batch_size, nb_channels_out, nb_rows_out, nb_cols_out)
    #
    # print(time.time() - t0)
