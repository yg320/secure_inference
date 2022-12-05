import torch
import numpy as np

from research.communication.utils import send_recv, recv, send

num_bit_to_dtype = {
    8: np.ubyte,
    16: np.ushort,
    32: np.uintc,
    64: np.ulonglong
}



class Client:
    def __init__(self, client_server_prf, client_provider_prf, client_server_socket, server_client_socket, client_provider_socket, provider_client_socket):
        self.client_server_prf = client_server_prf
        self.client_provider_prf = client_provider_prf
        self.client_server_socket = client_server_socket
        self.server_client_socket = server_client_socket
        self.client_provider_socket = client_provider_socket
        self.provider_client_socket = provider_client_socket

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
        padding = (f - 1) // 2

        A_share = torch.rand(size=X_share.shape, generator=self.client_provider_prf)
        B_share = torch.rand(size=Y_share.shape, generator=self.client_provider_prf)
        C_share = recv(socket=self.provider_client_socket)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        E_share_server = recv(self.server_client_socket)
        send(self.client_server_socket, E_share)
        F_share_server = recv(self.server_client_socket)
        send(self.client_server_socket, F_share)

        # E_share_server = send_recv(self.server_client_socket, self.client_server_socket, E_share)
        # F_share_server = send_recv(self.server_client_socket, self.client_server_socket, F_share)

        E = E_share_server + E_share
        F = F_share_server + F_share

        return torch.conv2d(X_share, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
            torch.conv2d(E, Y_share, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + C_share

    def share_convert(self, a_0):
        eta_pp = self.numpy_client_server_prf.integers(0, 2, size=a_0.shape, dtype=self.dtype)

        r = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_0 = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_1 = r - r_0

        alpha = (r < r_0).astype(self.dtype)

        a_tild_0 = a_0 + r_0
        beta_0 = (a_tild_0 < a_0).astype(self.dtype)
        send(self.client_provider_socket, a_tild_0)

        delta_0 = recv(self.provider_client_socket)

        # execute_secure_compare

        eta_p_0 = recv(self.provider_client_socket+1)

        t0 = eta_pp * eta_p_0
        t1 = self.add_mode_L_minus_one(t0, t0)
        t2 = self.sub_mode_L_minus_one(eta_pp, t1)
        eta_0 = self.add_mode_L_minus_one(eta_p_0, t2)

        t0 = self.add_mode_L_minus_one(delta_0, eta_0)
        t1 = self.sub_mode_L_minus_one(t0, self.dtype(1))
        t2 = self.sub_mode_L_minus_one(t1, alpha)
        theta_0 = self.add_mode_L_minus_one(beta_0, t2)

        y_0 = self.sub_mode_L_minus_one(a_0, theta_0)
        assert False, "Add zero shit"
        return y_0


if __name__ == "__main__":
    # !/usr/bin/python3
    global_prf = torch.Generator().manual_seed(0)
    image = torch.rand(size=(1, 3, 256, 256), generator=global_prf)
    weight_0 = torch.rand(size=(32, 3, 3, 3), generator=global_prf)
    weight_1 = torch.rand(size=(32, 32, 3, 3), generator=global_prf)
    weight_2 = torch.rand(size=(64, 32, 3, 3), generator=global_prf)

    client = Client(
        client_server_prf=torch.Generator().manual_seed(1),
        client_provider_prf=torch.Generator().manual_seed(2),
        client_server_socket=27123,
        server_client_socket=28124,
        client_provider_socket=21125,
        provider_client_socket=22126,
    )

    rng = np.random.default_rng(seed=0)
    a_0 = rng.integers(client.min_val, client.max_val + 1, size=(1000,), dtype=client.dtype)
    a_1 = rng.integers(client.min_val, client.max_val + 1, size=(1000,), dtype=client.dtype)
    a = a_0 + a_1
    y_0 = client.share_convert(a_0)
    send(socket=client.client_server_socket, data=y_0)
    print('fd')
    # image_share_server = torch.rand(size=(1, 3, 256, 256), generator=client.client_server_prf)
    # weight_0_share_client = torch.rand(size=(32, 3, 3, 3), generator=client.client_server_prf)
    # weight_1_share_client = torch.rand(size=(32, 32, 3, 3), generator=client.client_server_prf)
    # weight_2_share_client = torch.rand(size=(64, 32, 3, 3), generator=client.client_server_prf)
    #
    # image_share_client = image - image_share_server
    #
    # activation_share_client = image_share_client
    # activation_share_client = client.conv2d(activation_share_client, weight_0_share_client, stride=2)
    # activation_share_client = client.conv2d(activation_share_client, weight_1_share_client)
    # activation_share_client = client.conv2d(activation_share_client, weight_2_share_client)
    #
    # # activation_share_client = client.conv2d(image_share_client, weight_share_client)
    # send(socket=client.client_server_socket, data=activation_share_client)
    #
    # print('fdslkj')

