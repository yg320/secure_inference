import torch
from research.communication.utils import send_recv, send, recv
import time
import numpy as np

num_bit_to_dtype = {
    8: np.ubyte,
    16: np.ushort,
    32: np.uintc,
    64: np.ulonglong
}

class Server:
    def __init__(self, client_server_prf, server_provider_prf, client_server_socket, server_client_socket, server_provider_socket, provider_server_socket):
        self.client_server_prf = client_server_prf
        self.server_provider_prf = server_provider_prf
        self.client_server_socket = client_server_socket
        self.server_client_socket = server_client_socket
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


    def conv2d(self, X_share, Y_share, stride=1):
        assert Y_share.shape[2] == Y_share.shape[3]
        assert Y_share.shape[1] == X_share.shape[1]
        assert X_share.shape[2] == X_share.shape[3]

        b, i, m, _ = X_share.shape
        m = m // stride
        o, _, _, f = Y_share.shape
        output_shape = (b, o, m, m)
        padding = (f - 1) // 2

        A_share = torch.rand(size=X_share.shape, generator=self.server_provider_prf)
        B_share = torch.rand(size=Y_share.shape, generator=self.server_provider_prf)
        C_share = torch.rand(size=output_shape, generator=self.server_provider_prf)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        send(self.server_client_socket, E_share)
        E_share_client = recv(self.client_server_socket)
        send(self.server_client_socket, F_share)
        F_share_client = recv(self.client_server_socket)

        # E_share_client = send_recv(self.client_server_socket, self.server_client_socket, E_share)
        # F_share_client = send_recv(self.client_server_socket, self.server_client_socket, F_share)

        E = E_share_client + E_share
        F = F_share_client + F_share

        return -torch.conv2d(E, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
            torch.conv2d(X_share, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
            torch.conv2d(E, Y_share, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + C_share

    def share_convert(self, a_1):
        eta_pp = self.numpy_client_server_prf.integers(0, 2, size=a_0.shape, dtype=self.dtype)

        r = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_0 = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_1 = r - r_0

        alpha = (r < r_0).astype(self.dtype)

        a_tild_1 = a_1 + r_1
        beta_1 = (a_tild_1 < a_1).astype(self.dtype)
        send(self.server_provider_socket, a_tild_1)

        delta_1 = recv(self.provider_server_socket)

        # execute_secure_compare

        eta_p_1 = recv(self.provider_server_socket+1)

        t0 = eta_pp * eta_p_1
        t1 = self.add_mode_L_minus_one(t0, t0)
        eta_1 = self.sub_mode_L_minus_one(eta_p_1, t1)

        t0 = self.add_mode_L_minus_one(delta_1, eta_1)
        theta_1 = self.add_mode_L_minus_one(beta_1, t0)

        y_1 = self.sub_mode_L_minus_one(a_1, theta_1)

        return y_1


if __name__ == "__main__":

    global_prf = torch.Generator().manual_seed(0)
    image = torch.rand(size=(1, 3, 256, 256), generator=global_prf)
    weight_0 = torch.rand(size=(32, 3, 3, 3), generator=global_prf)
    weight_1 = torch.rand(size=(32, 32, 3, 3), generator=global_prf)
    weight_2 = torch.rand(size=(64, 32, 3, 3), generator=global_prf)

    server = Server(
        client_server_prf=torch.Generator().manual_seed(1),
        server_provider_prf=torch.Generator().manual_seed(3),
        client_server_socket=27123,
        server_client_socket=28124,
        server_provider_socket=23127,
        provider_server_socket=24128,
    )

    rng = np.random.default_rng(seed=0)
    a_0 = rng.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
    a_1 = rng.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
    a = a_0 + a_1
    y_1 = server.share_convert(a_1)
    y_0 = recv(server.client_server_socket)

    print(np.all(server.add_mode_L_minus_one(y_0, y_1) == a))
    print('fd')


    # image_share_server = torch.rand(size=(1, 3, 256, 256), generator=server.client_server_prf)
    # weight_0_share_client = torch.rand(size=(32, 3, 3, 3), generator=server.client_server_prf)
    # weight_1_share_client = torch.rand(size=(32, 32, 3, 3), generator=server.client_server_prf)
    # weight_2_share_client = torch.rand(size=(64, 32, 3, 3), generator=server.client_server_prf)
    #
    # weight_0_share_server = weight_0 - weight_0_share_client
    # weight_1_share_server = weight_1 - weight_1_share_client
    # weight_2_share_server = weight_2 - weight_2_share_client
    #
    # t0 = time.time()
    # activation_share_server = image_share_server
    # activation_share_server = server.conv2d(activation_share_server, weight_0_share_server, stride=2)
    # activation_share_server = server.conv2d(activation_share_server, weight_1_share_server)
    # activation_share_server = server.conv2d(activation_share_server, weight_2_share_server)
    # print(time.time() - t0)
    # activation_share_client = recv(server.client_server_socket)
    #
    # activation_recon = activation_share_client + activation_share_server
    t0 = time.time()
    # activation_non_secure = torch.conv2d(torch.conv2d(torch.conv2d(image, weight_0, padding=1, stride=2), weight_1, padding=1), weight_2, padding=1)
    # print(time.time() - t0)
    # print((activation_non_secure - (activation_share_server + activation_share_client)).abs().max())
    # print('fdslkj')
