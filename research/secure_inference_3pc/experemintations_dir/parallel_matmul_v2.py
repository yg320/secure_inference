import numpy as np
import time
from numba import njit, prange

@njit(parallel=True)
def mat_mult(a, b, c, d):
    assert a.shape[1] == b.shape[0]
    res = np.zeros((a.shape[0], b.shape[1]), )

    for i in prange(a.shape[0]):
        for k in range(a.shape[1]):
            for j in range(B.shape[1]):
                res[i, j] += a[i, k] * b[k, j]
                res[i, j] += c[i, k] * d[k, j]
    return res


def pre_conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    This is a block of local computation done at the beginning of the convolution. It
    basically does the matrix unrolling to be able to do the convolution as a simple
    matrix multiplication.

    Because all the computation are local, we add the @allow_command and run it directly
    on each share of the additive sharing tensor, when running mpc computations
    """
    assert len(input.shape) == 4
    assert len(weight.shape) == 4

    # Change to tuple if not one
    stride = (stride, stride)
    padding = (padding, padding)
    dilation = (dilation, dilation)

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
        input = np.pad(input, ((0,0), (0, 0), (padding[1], padding[1]), (padding[0], padding[0])), mode='constant')
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
        im_reshaped,
        weight_reshaped,
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

A = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(1, 512, 64, 64))
B = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(512, 512, 3, 3))
C = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(1, 512, 64, 64))
D = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(512, 512, 3, 3))
t0 = time.time()
A_, B_, b, c, h, w = pre_conv(A, B, padding=1)
C_, D_, b, c, h, w = pre_conv(A, B, padding=1)
print(time.time() - t0)
A_ = A_[0]
C_ = C_[0]

print(A_.shape, B_.shape, A_.dtype)
m, n, c = 4096, 4608, 512

A = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(m, n))
B = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(n, c))
C = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(m, n))
D = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(n, c))
print(A.dtype, A.shape, B.dtype, B.shape, C.dtype, C.shape, D.dtype, D.shape)
print(A_.dtype, A_.shape, B_.dtype, B_.shape, C_.dtype, C_.shape, D_.dtype, D_.shape)
A_ = A_.copy()
B_ = B_.copy()
C_ = C_.copy()
D_ = D_.copy()

# A = A_
# B = B_
# C = C_
# D = D_
t0 = time.time()
E_ = mat_mult(A_, B_, C_, D_)
print(time.time() - t0)
t0 = time.time()
E = mat_mult(A, B, C, D)
print(time.time() - t0)


# t0 = time.time()
# post_conv(None, C_, b, c, h, w)
# print(time.time() - t0)