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


NUM_BITS = 32
TRUNC = 10000

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

        self.torch_dtype = num_bit_to_torch_dtype[NUM_BITS]
        self.trunc = TRUNC

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


