import numpy as np
import time
from scipy.signal import convolve2d

m, n, c = 4096, 4608, 512
A = np.random.randint(np.iinfo(np.int64).min // 2, np.iinfo(np.int64).max // 2, size=(m, n))
B = np.random.randint(np.iinfo(np.int64).min // 2, np.iinfo(np.int64).max // 2, size=(n, c))
print(A.shape, A.dtype)
print(B.shape, B.dtype)
t0 = time.time()
# res = A @ B
print(time.time() - t0)

import torch

cur_type_numpy = np.int64
max_val_numpy = np.iinfo(cur_type_numpy).max
min_val_numpy = np.iinfo(cur_type_numpy).min

cur_type_torch = torch.int64
max_val_torch = torch.iinfo(cur_type_torch).max
min_val_torch = torch.iinfo(cur_type_torch).min
#
# t0 = time.time()
# arr_0 = np.random.randint(min_val_numpy, max_val_numpy+1, size=(10000000,))
# arr_1 = np.random.randint(min_val_numpy, max_val_numpy+1, size=(10000000,))
# c = arr_1 * arr_0
# print(time.time() - t0)
#
# t0 = time.time()
# arr_0 = np.random.random(size=(10000000,))
# arr_1 = np.random.random(size=(10000000,))
# c = arr_1 * arr_0
#
# print(time.time() - t0)
#
# t0 = time.time()
# arr_0 = torch.rand(size=(100000000,),dtype=torch.float32)
# arr_1 = torch.rand(size=(100000000,),dtype=torch.float32)
# c = arr_1 * arr_0
# print(time.time() - t0)
#
#
# t0 = time.time()
# arr_0 = torch.randint(
#     low=min_val_torch // 2,
#     high=max_val_torch // 2 + 1,
#     size=(100000000,),
#     dtype=cur_type_torch
# )
#
# arr_1 = torch.randint(
#     low=min_val_torch // 2,
#     high=max_val_torch // 2 + 1,
#     size=(100000000,),
#     dtype=cur_type_torch
# )
# c = arr_1 * arr_0
# print(time.time() - t0)
#
#


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
    stride = torch.nn.modules.utils._pair(stride)
    padding = torch.nn.modules.utils._pair(padding)
    dilation = torch.nn.modules.utils._pair(dilation)

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
        input = torch.nn.functional.pad(
            input, (padding[1], padding[1], padding[0], padding[0]), padding_mode
        )
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
    im_flat = input.reshape(batch_size, -1)
    im_reshaped = []
    for cur_row_out in range(nb_rows_out):
        for cur_col_out in range(nb_cols_out):
            # For each new output value, we just need to shift the receptive field
            offset = cur_row_out * stride[0] * nb_cols_in + cur_col_out * stride[1]
            tmp = [ind + offset for ind in pattern_ind]
            im_reshaped.append(im_flat[:, tmp])
    im_reshaped = torch.stack(im_reshaped).permute(1, 0, 2)

    # The convolution kernels are also reshaped for the matrix multiplication
    # We will get a matrix [[weights for out channel 0],
    #                       [weights for out channel 1],
    #                       ...
    #                       [weights for out channel nb_channels_out]].TRANSPOSE()
    weight_reshaped = weight.reshape(nb_channels_out // groups, -1).t()

    return (
        im_reshaped,
        weight_reshaped,
        torch.tensor(batch_size),
        torch.tensor(nb_channels_out),
        torch.tensor(nb_rows_out),
        torch.tensor(nb_cols_out),
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
        res.permute(0, 2, 1)
        .reshape(batch_size, nb_channels_out, nb_rows_out, nb_cols_out)
        .contiguous()
    )

    return res



# A = torch.rand(size=(1,512, 32, 32),dtype=torch.float64)
# B = torch.rand(size=(512,512, 3, 3),dtype=torch.float64)
# t0 = time.time()
# C = torch.conv2d(A, B, bias=None, stride=1, padding=1, dilation=1, groups=1)
# print(time.time() - t0)

A = torch.randint(
    low=min_val_torch // 2,
    high=max_val_torch // 2 + 1,
    size=(1, 512, 64, 64),
    dtype=cur_type_torch
)

B = torch.randint(
    low=min_val_torch // 2,
    high=max_val_torch // 2 + 1,
    size=(512, 512, 3, 3),
    dtype=cur_type_torch
)

# A = A.numpy()
# B = B.numpy()
#
# c = [[convolve2d(A[0,i], B[i,j], mode="same") for i in range(512)] for j in range(512)]
# # t0 = time.time()
A_, B_, b, c, h, w = pre_conv(A, B, padding=1)
# A_ = A_.to(torch.float64)
# B_ = B_.to(torch.float64)
# print(A_.shape)
# print(B_.shape)
# assert False
# t0 = time.time()
# C_ = A_ @ B_
# print(time.time() - t0)

A_numpy = A_.numpy()[0]#.astype(np.float64)
B_numpy = B_.numpy()#.astype(np.float64)
print("Start")
print(A_numpy.shape, A_numpy.dtype)
print(B_numpy.shape, B_numpy.dtype)

t0 = time.time()
C_numpy = A_numpy @ B_numpy
print(time.time() - t0)
