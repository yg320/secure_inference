import torch
from research.communication.utils import send, recv
import numpy as np

num_bit_to_dtype = {
    8: np.ubyte,
    16: np.ushort,
    32: np.uintc,
    64: np.ulonglong
}

class CryptoProvider:
    def __init__(self, server_provider_prf, client_provider_prf, client_provider_socket, provider_client_socket,
                 server_provider_socket, provider_server_socket):
        self.server_provider_prf = server_provider_prf
        self.client_provider_prf = client_provider_prf

        self.client_provider_socket = client_provider_socket
        self.provider_client_socket = provider_client_socket

        self.server_provider_socket = server_provider_socket
        self.provider_server_socket = provider_server_socket

        self.numpy_server_provider_prf = np.random.default_rng(seed=125)
        self.numpy_client_provider_prf = np.random.default_rng(seed=124)

        self.private_prf = np.random.default_rng(seed=1234)
        self.num_bits = 64
        self.dtype = num_bit_to_dtype[self.num_bits]
        self.torch_dtype = torch.int64

        self.min_val = np.iinfo(self.dtype).min
        self.max_val = np.iinfo(self.dtype).max

        self.L_minus_1 = 2 ** self.num_bits - 1

    def add_mode_L_minus_one(self, a, b):
        ret = a + b
        ret[ret < a] += self.dtype(1)
        ret[ret == self.L_minus_1] = self.dtype(0)
        return ret

    def sub_mode_L_minus_one(self, a, b):
        ret = a - b
        ret[b > a] -= self.dtype(1)
        return ret


    def conv2d(self, X_share_shape, Y_share_shape, stride=1):

        b, i, m, _ = X_share_shape
        m = m // stride
        o, _, _, f = Y_share_shape
        output_shape = (b, o, m, m)
        padding = (f - 1) // 2

        # TODO: remove //2
        A_share_server = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=X_share_shape, dtype=self.torch_dtype, generator=self.server_provider_prf)
        B_share_server = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=Y_share_shape, dtype=self.torch_dtype, generator=self.server_provider_prf)
        C_share_server = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=output_shape,  dtype=self.torch_dtype, generator=self.server_provider_prf)
        A_share_client = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=X_share_shape, dtype=self.torch_dtype, generator=self.client_provider_prf)
        B_share_client = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=Y_share_shape, dtype=self.torch_dtype, generator=self.client_provider_prf)

        A = A_share_client + A_share_server
        B = B_share_client + B_share_server

        C_share_client = torch.conv2d(A, B, bias=None, stride=stride, padding=padding, dilation=1, groups=1) - C_share_server
        send(socket=self.provider_client_socket, data=C_share_client)

    def mult(self, shape):

        A_share_server = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
        B_share_server = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
        C_share_server = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)

        A_share_client = self.numpy_client_provider_prf.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
        B_share_client = self.numpy_client_provider_prf.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)


        A = A_share_client + A_share_server
        B = B_share_client + B_share_server

        C_share_client = A * B - C_share_server
        send(socket=self.provider_client_socket, data=C_share_client)

    def share_convert(self, size):

        a_tild_0 = recv(self.client_provider_socket)
        a_tild_1 = recv(self.server_provider_socket)

        x = (a_tild_0 + a_tild_1)
        delta = (x < a_tild_0).astype(self.dtype)

        delta_0 = self.private_prf.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        delta_1 = self.sub_mode_L_minus_one(delta, delta_0)

        send(self.provider_client_socket, delta_0)
        send(self.provider_server_socket, delta_1)

        r = recv(self.server_provider_socket)
        eta_pp = recv(self.server_provider_socket)
        eta_p = eta_pp ^ (x > (r - 1))

        eta_p_0 = self.private_prf.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        eta_p_1 = self.sub_mode_L_minus_one(eta_p, eta_p_0)

        send(self.provider_client_socket, eta_p_0)
        send(self.provider_server_socket, eta_p_1)

        return

    def msb(self, size):
        x = self.private_prf.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        x_0 = self.private_prf.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        x_1 = self.sub_mode_L_minus_one(x, x_0)

        x_bit0 = x % 2
        x_bit_0_0 = self.private_prf.integers(self.min_val, self.max_val + 1, size=size, dtype=self.dtype)
        x_bit_0_1 = x_bit0 - x_bit_0_0

        send(self.provider_client_socket, x_0)
        send(self.provider_client_socket, x_bit_0_0)

        send(self.provider_server_socket, x_1)
        send(self.provider_server_socket, x_bit_0_1)

        r = recv(self.server_provider_socket)
        beta = recv(self.server_provider_socket)

        beta_p = beta ^ (x > r)
        beta_p_0 = self.private_prf.integers(self.min_val, self.max_val + 1, size=size, dtype=self.dtype)
        beta_p_1 = beta_p - beta_p_0

        send(self.provider_client_socket, beta_p_0)
        send(self.provider_server_socket, beta_p_1)

        self.mult(size)
        return

    def drelu(self, size):
        crypt_provider.share_convert(size)
        crypt_provider.msb(size)

    def relu(self, size):
        self.drelu(size)
        self.mult(size)

if __name__ == "__main__":

    share_convert_check = False
    conv_2d_check = False
    mult_check = False
    msb_check = False
    msb_share_check = False
    relu_check = False
    stem_check = True

    crypt_provider = CryptoProvider(
        server_provider_prf=torch.Generator().manual_seed(3),
        client_provider_prf=torch.Generator().manual_seed(2),
        client_provider_socket=45125,
        provider_client_socket=22126,
        server_provider_socket=23127,
        provider_server_socket=24128
    )
    if share_convert_check:
        crypt_provider.share_convert(size=1000)

    if conv_2d_check:
        crypt_provider.conv2d((1, 3, 256, 256), (32, 3, 3, 3), stride=2)
        crypt_provider.conv2d((1, 32, 128, 128), (32, 32, 3, 3))
        crypt_provider.conv2d((1, 32, 128, 128), (64, 32, 3, 3))

    if mult_check:
        crypt_provider.mult(1000)

    if msb_check:
        crypt_provider.msb(1000)

    if msb_share_check:

        crypt_provider.share_convert(1000)
        crypt_provider.msb(1000)

    if relu_check:
        crypt_provider.relu(1000)

    if stem_check:
        crypt_provider.conv2d((1, 3, 64, 64), (64, 3, 7, 7), stride=2)
        crypt_provider.relu(size=(1, 64, 32, 32))

        crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
        crypt_provider.relu(size=(1, 64, 32, 32))

        crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
        crypt_provider.relu(size=(1, 64, 32, 32))

        crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
        crypt_provider.relu(size=(1, 64, 32, 32))

        crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
        crypt_provider.relu(size=(1, 64, 32, 32))

        # crypt_provider.conv2d((1, 3, 64, 64), (64, 3, 7, 7), stride=2)
        # crypt_provider.relu(size=(1, 64, 32, 32))
        #
        # crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
        # crypt_provider.relu(size=(1, 64, 32, 32))
        #
        # crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
        # crypt_provider.relu(size=(1, 64, 32, 32))