import torch
import numpy as np

from research.communication.utils import Sender, Receiver
import time

num_bit_to_dtype = {
    8: np.ubyte,
    16: np.ushort,
    32: np.uintc,
    64: np.ulonglong
}

num_bit_to_sign_dtype = {
    32: np.int32,
    64: np.int64
}

num_bit_to_torch_dtype = {
    32: torch.int32,
    64: torch.int64
}

class Addresses:
    def __init__(self):
        self.port_01 = 12474
        self.port_10 = 12475
        self.port_02 = 12476
        self.port_20 = 12477
        self.port_12 = 12478
        self.port_21 = 12479

class NetworkAssets:
    def __init__(self, sender_01, sender_02, sender_12, receiver_01, receiver_02, receiver_12):
        # TODO: transfer only port
        self.receiver_12 = receiver_12
        self.receiver_02 = receiver_02
        self.receiver_01 = receiver_01
        self.sender_12 = sender_12
        self.sender_02 = sender_02
        self.sender_01 = sender_01

        if self.receiver_12:
            self.receiver_12.start()
        if self.receiver_02:
            self.receiver_02.start()
        if self.receiver_01:
            self.receiver_01.start()
        if self.sender_12:
            self.sender_12.start()
        if self.sender_02:
            self.sender_02.start()
        if self.sender_01:
            self.sender_01.start()


NUM_BITS = 64
TRUNC = 10000
dtype = num_bit_to_dtype[NUM_BITS]
powers = np.arange(NUM_BITS, dtype=num_bit_to_dtype[NUM_BITS])[np.newaxis]
moduli = (2 ** powers)
P = 67
def decompose(value):
    orig_shape = list(value.shape)
    value_bits = (value.reshape(-1, 1) & moduli) >> powers
    return value_bits.reshape(orig_shape + [NUM_BITS])

def sub_mode_p(x, y):
    mask = y > x
    ret = x - y
    ret_2 = x + (P - y)
    ret[mask] = ret_2[mask]
    return ret

class CryptoAssets:
    def __init__(self, prf_01_numpy, prf_02_numpy, prf_12_numpy, prf_01_torch, prf_02_torch, prf_12_torch):

        self.prf_12_torch = prf_12_torch
        self.prf_02_torch = prf_02_torch
        self.prf_01_torch = prf_01_torch
        self.prf_12_numpy = prf_12_numpy
        self.prf_02_numpy = prf_02_numpy
        self.prf_01_numpy = prf_01_numpy

        self.private_prf_numpy = np.random.default_rng(seed=31243)
        self.private_prf_torch = torch.Generator().manual_seed(31243)

        self.numpy_dtype = num_bit_to_dtype[NUM_BITS]
        self.torch_dtype = num_bit_to_torch_dtype[NUM_BITS]
        self.trunc = TRUNC

        self.numpy_max_val = np.iinfo(self.numpy_dtype).max
    def get_random_tensor_over_L(self, shape, prf):
        return torch.randint(
            low=torch.iinfo(self.torch_dtype).min // 2,
            high=torch.iinfo(self.torch_dtype).max // 2 + 1,
            size=shape,
            dtype=self.torch_dtype,
            generator=prf
        )



class SecureModule(torch.nn.Module):
    def __init__(self, crypto_assets, network_assets):

        super(SecureModule, self).__init__()

        self.crypto_assets = crypto_assets
        self.network_assets = network_assets

        self.trunc = TRUNC
        self.torch_dtype = num_bit_to_torch_dtype[NUM_BITS]
        self.dtype = num_bit_to_dtype[NUM_BITS]

        self.min_val = np.iinfo(self.dtype).min
        self.max_val = np.iinfo(self.dtype).max
        self.L_minus_1 = 2 ** NUM_BITS - 1
        self.signed_type = num_bit_to_sign_dtype[NUM_BITS]

    def add_mode_L_minus_one(self, a, b):
        ret = a + b
        ret[ret < a] += self.dtype(1)
        ret[ret == self.L_minus_1] = self.dtype(0)
        return ret

    def sub_mode_L_minus_one(self, a, b):
        ret = a - b
        ret[b > a] -= self.dtype(1)
        return ret



def fuse_conv_bn(conv_module, batch_norm_module):
    # TODO: this was copied from somewhere
    fusedconv = torch.nn.Conv2d(
        conv_module.in_channels,
        conv_module.out_channels,
        kernel_size=conv_module.kernel_size,
        stride=conv_module.stride,
        padding=conv_module.padding,
        bias=True
    )
    fusedconv.weight.requires_grad = False
    fusedconv.bias.requires_grad = False
    w_conv = conv_module.weight.clone().view(conv_module.out_channels, -1)
    w_bn = torch.diag(
        batch_norm_module.weight.div(torch.sqrt(batch_norm_module.eps + batch_norm_module.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
    if conv_module.bias is not None:
        b_conv = conv_module.bias
    else:
        b_conv = torch.zeros(conv_module.weight.size(0))
    b_bn = batch_norm_module.bias - batch_norm_module.weight.mul(batch_norm_module.running_mean).div(
        torch.sqrt(batch_norm_module.running_var + batch_norm_module.eps))
    fusedconv.bias.copy_(torch.matmul(w_bn, b_conv) + b_bn)

    W, B = fusedconv.weight, fusedconv.bias

    return W, B


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


from numba import njit, prange

@njit(parallel=True)
def mat_mult(a, b, c, d):
    assert a.shape[1] == b.shape[0]
    res = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)

    for i in prange(a.shape[0]):
        for k in range(a.shape[1]):
            for j in range(b.shape[1]):
                res[i,j] += a[i,k] * b[k,j]
                res[i,j] += c[i,k] * d[k,j]
    return res

@njit(parallel=True)
def mat_mult_single(a, b):
    assert a.shape[1] == b.shape[0]
    res = np.zeros((a.shape[0], b.shape[1]), dtype=a.dtype)

    for i in prange(a.shape[0]):
        for k in range(a.shape[1]):
            for j in range(b.shape[1]):
                res[i,j] += a[i,k] * b[k,j]
    return res


def get_c(x_bits, r_bits, t_bits, beta, j):
    x_bits = x_bits.astype(np.int32)
    r_bits = r_bits.astype(np.int32)
    t_bits = t_bits.astype(np.int32)
    beta = beta.astype(np.int32)
    j = j.astype(np.int32)
    one = np.int32(1)
    two = np.int32(2)
    beta = beta[..., np.newaxis]
    multiplexer_bits = r_bits * (one-beta) + t_bits * beta
    w = x_bits + j * multiplexer_bits - 2 * multiplexer_bits * x_bits
    rrr = w[..., ::-1].cumsum(axis=-1)[..., ::-1] - w

    zzz = j + (one - two*beta) * (j * multiplexer_bits - x_bits)
    return ((rrr + zzz) % P).astype(np.uint64)

# def get_c_case_0(x_bits, r_bits, j):
#     x_bits = x_bits.astype(np.int32)
#     r_bits = r_bits.astype(np.int32)
#     j = j.astype(np.int32)
#
#     w = x_bits + j * r_bits - 2 * r_bits * x_bits
#     rrr = w[..., ::-1].cumsum(axis=-1)[..., ::-1] - w
#     zzz = j + j * r_bits  - x_bits
#     return ((rrr + zzz) % P).astype(np.uint64)
#
#
# def get_c_case_1(x_bits, t_bits, j):
#     x_bits = x_bits.astype(np.int32)
#     t_bits = t_bits.astype(np.int32)
#     j = j.astype(np.int32)
#
#     w = x_bits + j * t_bits - 2 * t_bits * x_bits
#     rrr = w[..., ::-1].cumsum(axis=-1)[..., ::-1] - w
#     zzz = j - j * t_bits + x_bits
#
#     return (zzz+rrr) % P

def get_c_case_2(u, j):
    c = (P + 1 - j) * (u + 1) + (P-j) * u
    c[..., 0] = u[...,0] * (P-1) ** j
    return c % P