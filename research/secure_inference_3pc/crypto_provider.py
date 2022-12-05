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

        self.numpy_client_server_prf = np.random.default_rng(seed=123)
        self.num_bits = 32
        self.dtype = num_bit_to_dtype[self.num_bits]

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
        A_share_server = torch.rand(size=X_share_shape, generator=self.server_provider_prf)
        B_share_server = torch.rand(size=Y_share_shape, generator=self.server_provider_prf)
        C_share_server = torch.rand(size=output_shape, generator=self.server_provider_prf)

        A_share_client = torch.rand(size=X_share_shape, generator=self.client_provider_prf)
        B_share_client = torch.rand(size=Y_share_shape, generator=self.client_provider_prf)

        A = A_share_client + A_share_server
        B = B_share_client + B_share_server

        C_share_client = torch.conv2d(A, B, bias=None, stride=stride, padding=padding, dilation=1, groups=1) - C_share_server
        send(socket=self.provider_client_socket, data=C_share_client)

    def share_convert(self, size):

        eta_pp = self.numpy_client_server_prf.integers(0, 2, size=size, dtype=self.dtype)

        r = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=size, dtype=self.dtype)
        r_0 = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=size, dtype=self.dtype)
        r_1 = r - r_0


        a_tild_0 = recv(self.client_provider_socket)
        a_tild_1 = recv(self.server_provider_socket)

        x = (a_tild_0 + a_tild_1)
        delta = (x < a_tild_0).astype(self.dtype)

        delta_0 = self.numpy_client_server_prf.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        delta_1 = self.sub_mode_L_minus_one(delta, delta_0)

        send(self.provider_client_socket, delta_0)
        send(self.provider_server_socket, delta_1)

        eta_p = eta_pp ^ (x > (r - 1))

        eta_p_0 = self.numpy_client_server_prf.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        eta_p_1 = self.sub_mode_L_minus_one(eta_p, eta_p_0)

        send(self.provider_client_socket+1, eta_p_0)
        send(self.provider_server_socket+1, eta_p_1)

        return


if __name__ == "__main__":

    crypt_provider = CryptoProvider(
        server_provider_prf=torch.Generator().manual_seed(3),
        client_provider_prf=torch.Generator().manual_seed(2),
        client_provider_socket=21125,
        provider_client_socket=22126,
        server_provider_socket=23127,
        provider_server_socket=24128
    )
    crypt_provider.share_convert(size=1000)
    print('fd')

    # crypt_provider.conv2d((1, 3, 256, 256), (32, 3, 3, 3), stride=2)
    # crypt_provider.conv2d((1, 32, 128, 128), (32, 32, 3, 3))
    # crypt_provider.conv2d((1, 32, 128, 128), (64, 32, 3, 3))

