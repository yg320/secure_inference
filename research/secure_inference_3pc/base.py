import torch
from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.timer import timer

from research.secure_inference_3pc.communication.utils import Sender, Receiver
import time
from research.secure_inference_3pc.prf import MultiPartyPRFHandler
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER
from numba import njit, prange
from research.secure_inference_3pc.const import TRUNC, NUM_BITS, UNSIGNED_DTYPE, SIGNED_DTYPE, P, TORCH_DTYPE


class Addresses:
    def __init__(self):
        self.base_port = 2455
        self.port_01 = 10 * self.base_port + 0
        self.port_10 = 10 * self.base_port + 1
        self.port_02 = 10 * self.base_port + 2
        self.port_20 = 10 * self.base_port + 3
        self.port_12 = 10 * self.base_port + 4
        self.port_21 = 10 * self.base_port + 5

        self.ip_client = "localhost"
        self.ip_server = "localhost"
        self.ip_cryptoprovider = "localhost"

        self.ip_client_private = "localhost"
        self.ip_server_private = "localhost"
        self.ip_cryptoprovider_private = "localhost"

        # self.ip_client = "3.253.5.10"
        # self.ip_server = "3.253.18.156"
        # self.ip_cryptoprovider = "34.244.70.5"
        #
        # self.ip_client_private = "172.31.40.44"
        # self.ip_server_private = "172.31.47.91"
        # self.ip_cryptoprovider_private = "172.31.47.192"

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


def get_assets(party, device, simulated_bandwidth=None):

    addresses = Addresses()

    if party == 0:
        crypto_assets = MultiPartyPRFHandler(
            party=0, device=device, seeds={
                (CLIENT, SERVER): 0,
                (CLIENT, CRYPTO_PROVIDER): 1,
                # (SERVER, CRYPTO_PROVIDER): None,
                CLIENT: 3,
                # SERVER: None,
                # CRYPTO_PROVIDER: None,
            })
        network_assets = NetworkAssets(
            sender_01=Sender(ip=addresses.ip_server, port=addresses.port_01, simulated_bandwidth=simulated_bandwidth),
            sender_02=Sender(ip=addresses.ip_cryptoprovider, port=addresses.port_02, simulated_bandwidth=simulated_bandwidth),
            sender_12=None,
            receiver_01=Receiver(ip=addresses.ip_client_private, port=addresses.port_10, device=device),
            receiver_02=Receiver(ip=addresses.ip_client_private, port=addresses.port_20, device=device),
            receiver_12=None
        )

    if party == 1:
        crypto_assets = MultiPartyPRFHandler(
            party=1, device=device, seeds={
                (CLIENT, SERVER): 0,
                # (CLIENT, CRYPTO_PROVIDER): None,
                (SERVER, CRYPTO_PROVIDER): 2,
                # CLIENT: None,
                SERVER: 4,
                # CRYPTO_PROVIDER: None,
            })
        network_assets = NetworkAssets(
            sender_01=Sender(ip=addresses.ip_client, port=addresses.port_10, simulated_bandwidth=simulated_bandwidth),
            sender_02=None,
            sender_12=Sender(ip=addresses.ip_cryptoprovider, port=addresses.port_12, simulated_bandwidth=simulated_bandwidth),
            receiver_01=Receiver(ip=addresses.ip_server_private, port=addresses.port_01, device=device),
            receiver_02=None,
            receiver_12=Receiver(ip=addresses.ip_server_private, port=addresses.port_21, device=device),
        )

    if party == 2:
        crypto_assets = MultiPartyPRFHandler(
            party=2, device=device, seeds={
                # (CLIENT, SERVER): None,
                (CLIENT, CRYPTO_PROVIDER): 1,
                (SERVER, CRYPTO_PROVIDER): 2,
                # CLIENT: None,
                # SERVER: None,
                CRYPTO_PROVIDER: 5,
            })

        network_assets = NetworkAssets(
            sender_01=None,
            sender_02=Sender(ip=addresses.ip_client, port=addresses.port_20, simulated_bandwidth=simulated_bandwidth),
            sender_12=Sender(ip=addresses.ip_server, port=addresses.port_21, simulated_bandwidth=simulated_bandwidth),
            receiver_01=None,
            receiver_02=Receiver(ip=addresses.ip_cryptoprovider_private, port=addresses.port_02, device=device),
            receiver_12=Receiver(ip=addresses.ip_cryptoprovider_private, port=addresses.port_12, device=device),
        )

    return crypto_assets, network_assets

min_org_shit = -283206
max_org_shit = 287469
org_shit = backend.astype(backend.arange(min_org_shit, max_org_shit + 1) % P, backend.int8)

def module_67(xxx):
    if IS_TORCH_BACKEND:
    # TODO: fix this
        return xxx % 67
    else:
        orig_shape = xxx.shape
        xxx = xxx.reshape(-1)
        backend.subtract(xxx, min_org_shit, out=xxx)
        return org_shit[backend.astype(xxx, SIGNED_DTYPE)].reshape(orig_shape)


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
    beta = backend.unsqueeze(beta, -1)
    beta = 2 * beta  # Not allowed to change beta inplace
    backend.subtract(beta, 1, out=beta)
    backend.multiply(multiplexer_bits, x_bits, out=multiplexer_bits)
    backend.multiply(multiplexer_bits, -2, out=multiplexer_bits)
    backend.add(multiplexer_bits, x_bits, out=multiplexer_bits)

    w_cumsum = backend.astype(multiplexer_bits, backend.int32)
    backend.cumsum(w_cumsum, axis=-1, out=w_cumsum)
    backend.subtract(w_cumsum, multiplexer_bits, out=w_cumsum)
    backend.multiply(x_bits, beta, out=x_bits)
    backend.add(w_cumsum, x_bits, out=w_cumsum)

    return w_cumsum

def get_c_party_1(x_bits, multiplexer_bits, beta):
    beta = backend.unsqueeze(beta, -1)
    beta = -2 * beta  # Not allowed to change beta inplace
    backend.add(beta, 1, out=beta)

    w = multiplexer_bits * x_bits
    backend.multiply(w, -2, out=w)
    backend.add(w, x_bits, out=w)
    backend.add(w, multiplexer_bits, out=w)

    w_cumsum = backend.astype(w, backend.int32)
    backend.cumsum(w_cumsum, axis=-1, out=w_cumsum)
    backend.subtract(w_cumsum, w, out=w_cumsum)

    backend.subtract(multiplexer_bits, x_bits, out=multiplexer_bits)
    backend.multiply(multiplexer_bits, beta, out=multiplexer_bits)
    backend.add(multiplexer_bits, 1, out=multiplexer_bits)
    backend.add(w_cumsum, multiplexer_bits, out=w_cumsum)

    return w_cumsum


# def get_c(x_bits, multiplexer_bits, beta, j):
#     beta = beta[..., backend.newaxis]
#     w = x_bits + j * multiplexer_bits - 2 * multiplexer_bits * x_bits
#     w_cumsum = w.astype(backend.int32)
#     backend.cumsum(w_cumsum, axis=-1, out=w_cumsum)
#     backend.subtract(w_cumsum, w, out=w_cumsum)
#     rrr = w_cumsum
#     zzz = j + (1 - 2 * beta) * (j * multiplexer_bits - x_bits)
#     ret = rrr + zzz.astype(backend.int32)
#
#     return ret
#
#
# def get_c_case_2(u, j):
#     c = (P + 1 - j) * (u + 1) + (P-j) * u
#     c[..., 0] = u[...,0] * (P-1) ** j
#     return c % P


from research.secure_inference_3pc.const import IS_TORCH_BACKEND
class TypeConverter:
    trunc = TRUNC
    int_dtype = TORCH_DTYPE
    float_dtype = torch.float32

    @staticmethod
    def f2i(data):
        if type(data) in [torch.Tensor, torch.nn.Parameter]:
            if IS_TORCH_BACKEND:
                return ((data * TypeConverter.trunc).round().to(TypeConverter.int_dtype))
            else:
                return ((data * TypeConverter.trunc).round().to(TypeConverter.int_dtype)).numpy()  # NUMPY_CONVERSION

        else:
            if IS_TORCH_BACKEND:
                return ((data * TypeConverter.trunc).round().to(TypeConverter.int_dtype))
            else:
                return ((torch.from_numpy(data) * TypeConverter.trunc).round().to(TypeConverter.int_dtype)).numpy()  # NUMPY_CONVERSION
    @staticmethod
    def i2f(data):
        if IS_TORCH_BACKEND:
            return (data.to(TypeConverter.float_dtype) / TypeConverter.trunc)
        else:
            return torch.from_numpy(data).to(TypeConverter.float_dtype) / TypeConverter.trunc  # NUMPY_CONVERSION