import numpy as np
import time

def pre_conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    This is a block of local computation done at the beginning of the convolution. It
    basically does the matrix unrolling to be able to do the convolution as a simple
    matrix multiplication.

    Because all the computation are local, we add the @allow_command and run it directly
    on each share of the additive sharing tensor, when running mpc computations
    """
    t0 = time.time()

    print(input.shape, weight.shape, stride, padding, dilation, groups)
    assert len(input.shape) == 4
    assert len(weight.shape) == 4

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

    t1 = time.time()
    pattern_ind = []
    for ch in range(nb_channels_in):
        for r in range(nb_rows_kernel):
            for c in range(nb_cols_kernel):
                pixel = r * nb_cols_in * dilation[0] + c * dilation[1]
                pattern_ind.append(pixel + ch * nb_rows_in * nb_cols_in)


    pattern_ind = np.array(pattern_ind)
    t2 = time.time()
    im_flat = input.reshape(batch_size, -1)
    im_reshaped = []
    for cur_row_out in range(nb_rows_out):
        for cur_col_out in range(nb_cols_out):
            # For each new output value, we just need to shift the receptive field
            offset = cur_row_out * stride[0] * nb_cols_in + cur_col_out * stride[1]
            tmp = offset + pattern_ind
            im_reshaped.append(im_flat[:, tmp])
    im_reshaped = np.stack(im_reshaped).transpose(1, 0, 2)

    weight_reshaped = weight.reshape(nb_channels_out // groups, -1).T

    t3 = time.time()
    print(t1 - t0)
    print(t2 - t1)
    print(t3 - t2)
    return (
        im_reshaped,
        weight_reshaped,
        batch_size,
        nb_channels_out,
        nb_rows_out,
        nb_cols_out,
    )


input = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(1, 512, 24, 24))
weight = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(512, 512, 3, 3))
stride = (1, 1)
padding = (1, 1)
dilation = 1
groups = 1

for _ in range(10):
    t0 = time.time()
    pre_conv(input, weight, None, stride, padding, dilation, groups)
    t1 = time.time()
    print(t1 - t0)
