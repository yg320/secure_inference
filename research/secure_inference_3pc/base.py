import torch
import numpy as np

from research.secure_inference_3pc.communication.utils import Sender, Receiver
import time
from research.secure_inference_3pc.prf import MultiPartyPRFHandler
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER
from numba import njit, prange
from research.secure_inference_3pc.const import TRUNC, NUM_BITS, UNSIGNED_DTYPE, SIGNED_DTYPE, P, TORCH_DTYPE, NUM_OF_COMPARE_BITS, IGNORE_MSB_BITS


class Addresses:
    def __init__(self):
        self.port_01 = 18041
        self.port_10 = 18042
        self.port_02 = 18043
        self.port_20 = 18044
        self.port_12 = 18045
        self.port_21 = 18046


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

    def done(self):
        if self.sender_12:
            self.sender_12.put(None)
        if self.sender_02:
            self.sender_02.put(None)
        if self.sender_01:
            self.sender_01.put(None)


def get_assets(party, repeat, simulated_bandwidth=None):

    addresses = Addresses()

    if party == 0:
        crypto_assets = MultiPartyPRFHandler(
            party=0, seeds={
                (CLIENT, SERVER): 0,
                (CLIENT, CRYPTO_PROVIDER): 1,
                # (SERVER, CRYPTO_PROVIDER): None,
                CLIENT: 3,
                # SERVER: None,
                # CRYPTO_PROVIDER: None,
            })
        network_assets = NetworkAssets(
            sender_01=Sender(addresses.port_01, simulated_bandwidth=simulated_bandwidth),
            sender_02=Sender(addresses.port_02, simulated_bandwidth=simulated_bandwidth),
            sender_12=None,
            receiver_01=Receiver(addresses.port_10),
            receiver_02=Receiver(addresses.port_20),
            receiver_12=None
        )

    if party == 1:
        crypto_assets = MultiPartyPRFHandler(
            party=1, seeds={
                (CLIENT, SERVER): 0,
                # (CLIENT, CRYPTO_PROVIDER): None,
                (SERVER, CRYPTO_PROVIDER): 2,
                # CLIENT: None,
                SERVER: 4,
                # CRYPTO_PROVIDER: None,
            })
        network_assets = NetworkAssets(
            sender_01=Sender(addresses.port_10, simulated_bandwidth=simulated_bandwidth),
            sender_02=None,
            sender_12=Sender(addresses.port_12, simulated_bandwidth=simulated_bandwidth),
            receiver_01=Receiver(addresses.port_01),
            receiver_02=None,
            receiver_12=Receiver(addresses.port_21),
        )

    if party == 2:
        crypto_assets = MultiPartyPRFHandler(
            party=2, seeds={
                # (CLIENT, SERVER): None,
                (CLIENT, CRYPTO_PROVIDER): 1,
                (SERVER, CRYPTO_PROVIDER): 2,
                # CLIENT: None,
                # SERVER: None,
                CRYPTO_PROVIDER: 5,
            })

        network_assets = NetworkAssets(
            sender_01=None,
            sender_02=Sender(addresses.port_20, simulated_bandwidth=simulated_bandwidth),
            sender_12=Sender(addresses.port_21, simulated_bandwidth=simulated_bandwidth),
            receiver_01=None,
            receiver_02=Receiver(addresses.port_02),
            receiver_12=Receiver(addresses.port_12),
        )

    return crypto_assets, network_assets


powers = np.arange(NUM_BITS, dtype=UNSIGNED_DTYPE)[np.newaxis][:, ::-1]
powers_torch_cuda_0 = torch.from_numpy(powers.astype(np.int64)).to("cuda:0")
powers_torch_cuda_1 = torch.from_numpy(powers.astype(np.int64)).to("cuda:1")


min_org_shit = -283206
max_org_shit = 287469
org_shit = (np.arange(min_org_shit, max_org_shit + 1) % P).astype(np.uint8)


def module_67(xxx):
    orig_shape = xxx.shape
    xxx = xxx.reshape(-1)
    np.subtract(xxx, min_org_shit, out=xxx)
    return org_shit[xxx].reshape(orig_shape)

def decompose(value):
    orig_shape = list(value.shape)
    value = value.reshape(-1, 1)
    end = None if IGNORE_MSB_BITS == 0 else -IGNORE_MSB_BITS
    r_shift = value.astype(np.uint64) >> powers[:, NUM_BITS - NUM_OF_COMPARE_BITS-IGNORE_MSB_BITS:end]
    value_bits = np.zeros(shape=(value.shape[0], NUM_OF_COMPARE_BITS), dtype=np.int8)
    np.bitwise_and(r_shift, np.int8(1), out=value_bits)
    ret = value_bits.reshape(orig_shape + [NUM_OF_COMPARE_BITS])
    return ret

def decompose_torch_0(value):
    orig_shape = list(value.shape)
    value = value.reshape(-1, 1)
    end = None if IGNORE_MSB_BITS == 0 else -IGNORE_MSB_BITS

    r_shift = value >> powers_torch_cuda_0[:,NUM_BITS - NUM_OF_COMPARE_BITS-IGNORE_MSB_BITS:end]
    value_bits = r_shift & 1

    ret = value_bits.to(torch.int8).reshape(orig_shape + [NUM_OF_COMPARE_BITS])
    return ret

def decompose_torch_1(value):
    orig_shape = list(value.shape)
    value = value.reshape(-1, 1)
    end = None if IGNORE_MSB_BITS == 0 else -IGNORE_MSB_BITS

    r_shift = value >> powers_torch_cuda_1[:,NUM_BITS - NUM_OF_COMPARE_BITS-IGNORE_MSB_BITS:end]
    value_bits = r_shift & 1

    ret = value_bits.to(torch.int8).reshape(orig_shape + [NUM_OF_COMPARE_BITS])
    return ret

def sub_mode_p(x, y):
    mask = y > x
    ret = x - y
    ret_2 = x + (P - y)
    ret[mask] = ret_2[mask]
    return ret


def fuse_conv_bn(conv_module, batch_norm_module):
    # TODO: this was copied from somewhere
    if conv_module.groups > 1:
        assert conv_module.in_channels == conv_module.out_channels == conv_module.groups
        assert conv_module.bias is None

        Ws = []
        Bs = []
        for i in range(conv_module.groups):
            fusedconv = torch.nn.Conv2d(
                1,
                1,
                kernel_size=conv_module.kernel_size,
                stride=conv_module.stride,
                padding=conv_module.padding,
                bias=True
            )
            fusedconv.weight.requires_grad = False
            fusedconv.bias.requires_grad = False
            w_conv = conv_module.weight[i:i+1].clone().view(1, -1)
            w_bn = torch.diag(
                batch_norm_module.weight[i:i+1].div(torch.sqrt(batch_norm_module.eps + batch_norm_module.running_var[i:i+1])))
            fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
            b_bn = batch_norm_module.bias - batch_norm_module.weight.mul(batch_norm_module.running_mean).div(
                torch.sqrt(batch_norm_module.running_var + batch_norm_module.eps))

            fusedconv.bias.copy_(b_bn[i:i+1])

            W, B = fusedconv.weight, fusedconv.bias
            Ws.append(W)
            Bs.append(B)
        W = torch.cat(Ws, dim=0)
        B = torch.cat(Bs, dim=0)
        return W, B
    else:

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


def get_c_party_0(x_bits, multiplexer_bits, beta):
    beta = beta[..., np.newaxis]
    beta = 2 * beta  # Not allowed to change beta inplace
    np.subtract(beta, 1, out=beta)
    np.multiply(multiplexer_bits, x_bits, out=multiplexer_bits)
    np.multiply(multiplexer_bits, -2, out=multiplexer_bits)
    np.add(multiplexer_bits, x_bits, out=multiplexer_bits)

    w_cumsum = multiplexer_bits.astype(np.int32)
    np.cumsum(w_cumsum, axis=-1, out=w_cumsum)
    np.subtract(w_cumsum, multiplexer_bits, out=w_cumsum)
    np.multiply(x_bits, beta, out=x_bits)
    np.add(w_cumsum, x_bits, out=w_cumsum)

    return w_cumsum


def get_c_party_0_torch(x_bits, multiplexer_bits, beta):

    beta = beta.unsqueeze(-1)
    beta = 2 * beta  # Not allowed to change beta inplace
    torch.sub(beta, 1, out=beta)
    torch.mul(multiplexer_bits, x_bits, out=multiplexer_bits)
    torch.mul(multiplexer_bits, -2, out=multiplexer_bits)
    torch.add(multiplexer_bits, x_bits, out=multiplexer_bits)

    w_cumsum = multiplexer_bits.to(torch.int32)
    torch.cumsum(w_cumsum, dim=-1, out=w_cumsum)
    torch.sub(w_cumsum, multiplexer_bits, out=w_cumsum)
    torch.mul(x_bits, beta, out=x_bits)
    torch.add(w_cumsum, x_bits, out=w_cumsum)

    return w_cumsum

def get_c_party_1_torch(x_bits, multiplexer_bits, beta):
    beta = beta.unsqueeze(-1)
    beta = -2 * beta  # Not allowed to change beta inplace
    torch.add(beta, 1, out=beta)

    w = multiplexer_bits * x_bits
    torch.mul(w, -2, out=w)
    torch.add(w, x_bits, out=w)
    torch.add(w, multiplexer_bits, out=w)

    w_cumsum = w.to(torch.int32)
    torch.cumsum(w_cumsum, dim=-1, out=w_cumsum)
    torch.sub(w_cumsum, w, out=w_cumsum)

    torch.sub(multiplexer_bits, x_bits, out=multiplexer_bits)
    torch.mul(multiplexer_bits, beta, out=multiplexer_bits)
    torch.add(multiplexer_bits, 1, out=multiplexer_bits)
    torch.add(w_cumsum, multiplexer_bits, out=w_cumsum)

    return w_cumsum

def get_c_party_1(x_bits, multiplexer_bits, beta):
    beta = beta[..., np.newaxis]
    beta = -2 * beta  # Not allowed to change beta inplace
    np.add(beta, 1, out=beta)

    w = multiplexer_bits * x_bits
    np.multiply(w, -2, out=w)
    np.add(w, x_bits, out=w)
    np.add(w, multiplexer_bits, out=w)

    w_cumsum = w.astype(np.int32)
    np.cumsum(w_cumsum, axis=-1, out=w_cumsum)
    np.subtract(w_cumsum, w, out=w_cumsum)

    np.subtract(multiplexer_bits, x_bits, out=multiplexer_bits)
    np.multiply(multiplexer_bits, beta, out=multiplexer_bits)
    np.add(multiplexer_bits, 1, out=multiplexer_bits)
    np.add(w_cumsum, multiplexer_bits, out=w_cumsum)

    return w_cumsum


def get_c(x_bits, multiplexer_bits, beta, j):
    beta = beta[..., np.newaxis]
    w = x_bits + j * multiplexer_bits - 2 * multiplexer_bits * x_bits
    w_cumsum = w.astype(np.int32)
    np.cumsum(w_cumsum, axis=-1, out=w_cumsum)
    np.subtract(w_cumsum, w, out=w_cumsum)
    rrr = w_cumsum
    zzz = j + (1 - 2 * beta) * (j * multiplexer_bits - x_bits)
    ret = rrr + zzz.astype(np.int32)

    return ret


def get_c_case_2(u, j):
    c = (P + 1 - j) * (u + 1) + (P-j) * u
    c[..., 0] = u[...,0] * (P-1) ** j
    return c % P



class TypeConverter:
    trunc = TRUNC
    int_dtype = TORCH_DTYPE
    float_dtype = torch.float32

    @staticmethod
    def f2i(data):
        if type(data) in [torch.Tensor, torch.nn.Parameter]:
            return ((data * TypeConverter.trunc).round().to(TypeConverter.int_dtype)).numpy()
        else:
            return ((torch.from_numpy(data) * TypeConverter.trunc).round().to(TypeConverter.int_dtype)).numpy()
    @staticmethod
    def i2f(data):
        return torch.from_numpy(data).to(TypeConverter.float_dtype) / TypeConverter.trunc