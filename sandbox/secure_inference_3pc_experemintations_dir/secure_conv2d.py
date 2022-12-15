import torch
import numpy as np


# Copied as is from syft.frameworks.torch.nn.functional.py
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


i = 3
o = 64
f = 7
m = 256
padding = (f - 1) // 2
X0 = torch.rand(size=(1, i, m, m))
X1 = torch.rand(size=(1, i, m, m))

Y0 = torch.rand(size=(o,i, f, f))
Y1 = torch.rand(size=(o,i, f, f))

X = X0 + X1
Y = Y0 + Y1


A0 = torch.rand(size=(1, i, m, m))
A1 = torch.rand(size=(1, i, m, m))

B0 = torch.rand(size=(o, i, f, f))
B1 = torch.rand(size=(o, i, f, f))

C0 = torch.rand(size=(1, o, m, m))

A = A0 + A1
B = B0 + B1

# A_prime, B_prime, *_ = pre_conv(A, B, padding=3)
# C1 = post_conv(bias=None, res=A_prime @ B_prime, batch_size=1, nb_channels_out=1, nb_rows_out=256, nb_cols_out=256) - C0
C1 = torch.conv2d(A, B, bias=None, stride=1, padding=padding, dilation=1, groups=1) - C0

E0 = X0 - A0
E1 = X1 - A1
F0 = Y0 - B0
F1 = Y1 - B1

E = E0 + E1
F = F0 + F1


# E_prime, F_prime, *_ = pre_conv(E, F, padding=3)
# X0_prime, Y0_prime, *_ = pre_conv(X0, Y0, padding=3)
# X1_prime, Y1_prime, *_ = pre_conv(X1, Y1, padding=3)
#
# Z0 = post_conv(bias=None, res= 0 * E_prime @ F_prime + X0_prime @ F_prime + E_prime @ Y0_prime, batch_size=1, nb_channels_out=1, nb_rows_out=256, nb_cols_out=256) + C0
# Z1 = post_conv(bias=None, res=-1 * E_prime @ F_prime + X1_prime @ F_prime + E_prime @ Y1_prime, batch_size=1, nb_channels_out=1, nb_rows_out=256, nb_cols_out=256) + C1

Z0 = 0 * torch.conv2d(E, F,  bias=None, stride=1, padding=padding, dilation=1, groups=1) + \
         torch.conv2d(X0, F, bias=None, stride=1, padding=padding, dilation=1, groups=1) + \
         torch.conv2d(E, Y0, bias=None, stride=1, padding=padding, dilation=1, groups=1) + C0

Z1 = -1 * torch.conv2d(E, F,  bias=None, stride=1, padding=padding, dilation=1, groups=1) + \
          torch.conv2d(X1, F, bias=None, stride=1, padding=padding, dilation=1, groups=1) + \
          torch.conv2d(E, Y1, bias=None, stride=1, padding=padding, dilation=1, groups=1) + C1

Z = Z0 + Z1
Z_normal = torch.conv2d(X, Y, bias=None, stride=1, padding=padding, dilation=1, groups=1)
print('fds')