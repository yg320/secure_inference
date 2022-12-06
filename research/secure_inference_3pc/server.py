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
        self.numpy_server_provider_prf = np.random.default_rng(seed=125)

        self.num_bits = 64
        self.dtype = num_bit_to_dtype[self.num_bits]
        self.torch_dtype = torch.int64
        self.min_val = np.iinfo(self.dtype).min
        self.max_val = np.iinfo(self.dtype).max

        self.L_minus_1 = 2 ** self.num_bits - 1

        self.send_time = 0
        self.recv_time = 0

    def add_mode_L_minus_one(self, a, b):
        ret = a + b
        ret[ret < a] += self.dtype(1)
        ret[ret == self.L_minus_1] = self.dtype(0)
        return ret

    def sub_mode_L_minus_one(self, a, b):
        ret = a - b
        ret[b > a] -= self.dtype(1)
        return ret

    def conv2d(self, X_share, Y_share, stride=1, bias=None):
        assert Y_share.shape[2] == Y_share.shape[3]
        assert Y_share.shape[1] == X_share.shape[1]
        assert X_share.shape[2] == X_share.shape[3]

        b, i, m, _ = X_share.shape
        m = m // stride
        o, _, _, f = Y_share.shape
        output_shape = (b, o, m, m)
        padding = (f - 1) // 2

        A_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=X_share.shape, dtype=self.torch_dtype, generator=self.server_provider_prf)
        B_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=Y_share.shape, dtype=self.torch_dtype, generator=self.server_provider_prf)
        C_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=output_shape,  dtype=self.torch_dtype, generator=self.server_provider_prf)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        t0 = time.time()
        send(self.server_client_socket, E_share)
        self.send_time += (time.time() - t0)

        t0 = time.time()
        x = recv(self.client_server_socket)
        self.recv_time += (time.time() - t0)

        E_share_client = torch.from_numpy(x)

        t0 = time.time()
        send(self.server_client_socket, F_share)
        self.send_time += (time.time() - t0)

        t0 = time.time()
        x = recv(self.client_server_socket)
        self.recv_time += (time.time() - t0)

        F_share_client = torch.from_numpy(x)


        E = E_share_client + E_share
        F = F_share_client + F_share

        out = (-torch.conv2d(E, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
                torch.conv2d(X_share, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
                torch.conv2d(E, Y_share, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + C_share) // 10000
        out = out + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return out

    def mult(self, X_share, Y_share):

        A_share = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        B_share = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        C_share = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        t0 = time.time()
        send(self.server_client_socket, E_share)
        self.send_time += (time.time() - t0)

        t0 = time.time()
        E_share_client = recv(self.client_server_socket)
        self.recv_time += (time.time() - t0)

        t0 = time.time()
        send(self.server_client_socket, F_share)
        self.send_time += (time.time() - t0)

        t0 = time.time()
        F_share_client = recv(self.client_server_socket)
        self.recv_time += (time.time() - t0)

        E = E_share_client + E_share
        F = F_share_client + F_share

        return - E * F + X_share * F + Y_share * E + C_share

    def share_convert(self, a_1):
        eta_pp = self.numpy_client_server_prf.integers(0, 2, size=a_1.shape, dtype=self.dtype)

        r = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
        r_0 = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
        r_1 = r - r_0

        alpha = (r < r_0).astype(self.dtype)

        a_tild_1 = a_1 + r_1
        beta_1 = (a_tild_1 < a_1).astype(self.dtype)


        t0 = time.time()
        send(self.server_provider_socket, a_tild_1)
        self.send_time += (time.time() - t0)


        t0 = time.time()
        delta_1 = recv(self.provider_server_socket)
        self.recv_time += (time.time() - t0)


        t0 = time.time()
        send(self.server_provider_socket, r)
        self.send_time += (time.time() - t0)

        t0 = time.time()
        send(self.server_provider_socket, eta_pp)
        self.send_time += (time.time() - t0)

        # execute_secure_compare
        t0 = time.time()
        eta_p_1 = recv(self.provider_server_socket)
        self.recv_time += (time.time() - t0)

        t0 = eta_pp * eta_p_1
        t1 = self.add_mode_L_minus_one(t0, t0)
        eta_1 = self.sub_mode_L_minus_one(eta_p_1, t1)

        t0 = self.add_mode_L_minus_one(delta_1, eta_1)
        theta_1 = self.add_mode_L_minus_one(beta_1, t0)

        y_1 = self.sub_mode_L_minus_one(a_1, theta_1)

        return y_1

    def msb(self, a_1):
        beta = self.numpy_client_server_prf.integers(0, 2, size=a_1.shape, dtype=self.dtype)
        t0 = time.time()
        x_1 = recv(self.provider_server_socket)
        self.recv_time += (time.time() - t0)

        t0 = time.time()
        x_bit_0_1 = recv(self.provider_server_socket)
        self.recv_time += (time.time() - t0)

        y_1 = self.add_mode_L_minus_one(a_1, a_1)
        r_1 = self.add_mode_L_minus_one(x_1, y_1)


        t0 = time.time()
        send(self.server_client_socket, r_1)
        self.send_time += (time.time() - t0)

        t0 = time.time()
        r_0 = recv(self.client_server_socket)
        self.recv_time += (time.time() - t0)

        r = self.add_mode_L_minus_one(r_0, r_1)

        t0 = time.time()
        send(self.server_provider_socket, r)
        self.send_time += (time.time() - t0)

        t0 = time.time()
        send(self.server_provider_socket, beta)
        self.send_time += (time.time() - t0)


        # execute_secure_compare
        t0 = time.time()
        beta_p_1 = recv(self.provider_server_socket)
        self.recv_time += (time.time() - t0)

        gamma_1 = beta_p_1 + (1 * beta) - (2 * beta * beta_p_1)
        delta_1 = x_bit_0_1 + (1 * (r % 2)) - (2 * (r % 2) * x_bit_0_1)

        theta_1 = self.mult(gamma_1, delta_1)
        alpha_1 = gamma_1 + delta_1 - 2 * theta_1

        return alpha_1

    def drelu(self, X1):
        X1_converted = self.share_convert(X1)
        MSB_1 = self.msb(X1_converted)
        return 1 - MSB_1

    def relu(self, X1):
        X1 = X1.astype(np.uint64)

        MSB_1 = self.drelu(X1)
        relu_1 = self.mult(X1, MSB_1)
        return torch.from_numpy(relu_1.astype(np.int64))

if __name__ == "__main__":


    share_convert_check = False
    conv_2d_check = False
    mult_check = False
    msb_check = False
    msb_share_check = False
    relu_check = False
    stem_check = True

    server = Server(
        client_server_prf=torch.Generator().manual_seed(1),
        server_provider_prf=torch.Generator().manual_seed(3),
        client_server_socket=27123,
        server_client_socket=28124,
        server_provider_socket=23127,
        provider_server_socket=24128,
    )

    if share_convert_check:
        rng = np.random.default_rng(seed=0)
        a_0 = rng.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
        a_1 = rng.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
        a = a_0 + a_1
        y_1 = server.share_convert(a_1)
        y_0 = recv(server.client_server_socket)

        print(np.all(server.add_mode_L_minus_one(y_0, y_1) == a))
        print('fd')

    if conv_2d_check:
        global_prf = torch.Generator().manual_seed(0)
        image = torch.rand(size=(1, 3, 256, 256), generator=global_prf)
        weight_0 = torch.rand(size=(32, 3, 3, 3), generator=global_prf)
        weight_1 = torch.rand(size=(32, 32, 3, 3), generator=global_prf)
        weight_2 = torch.rand(size=(64, 32, 3, 3), generator=global_prf)

        image_share_server = torch.rand(size=(1, 3, 256, 256), generator=server.client_server_prf)
        weight_0_share_client = torch.rand(size=(32, 3, 3, 3), generator=server.client_server_prf)
        weight_1_share_client = torch.rand(size=(32, 32, 3, 3), generator=server.client_server_prf)
        weight_2_share_client = torch.rand(size=(64, 32, 3, 3), generator=server.client_server_prf)

        weight_0_share_server = weight_0 - weight_0_share_client
        weight_1_share_server = weight_1 - weight_1_share_client
        weight_2_share_server = weight_2 - weight_2_share_client

        t0 = time.time()
        activation_share_server = image_share_server
        activation_share_server = server.conv2d(activation_share_server, weight_0_share_server, stride=2)
        activation_share_server = server.conv2d(activation_share_server, weight_1_share_server)
        activation_share_server = server.conv2d(activation_share_server, weight_2_share_server)
        print(time.time() - t0)
        activation_share_client = recv(server.client_server_socket)

        activation_recon = activation_share_client + activation_share_server
        t0 = time.time()
        activation_non_secure = torch.conv2d(torch.conv2d(torch.conv2d(image, weight_0, padding=1, stride=2), weight_1, padding=1), weight_2, padding=1)
        print(time.time() - t0)
        print((activation_non_secure - (activation_share_server + activation_share_client)).abs().max())
        print('fdslkj')

    if mult_check:
        X0 = server.numpy_client_server_prf.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
        Y0 = server.numpy_client_server_prf.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
        X1 = server.numpy_client_server_prf.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
        Y1 = server.numpy_client_server_prf.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)

        Z1 = server.mult(X1, Y1)
        Z0 = recv(server.client_server_socket)
        assert np.all((X0 + X1) * (Y0 + Y1) == (Z0 + Z1))
        print('fds')

    if msb_check:
        X0 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)
        X1 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)

        MSB_1 = server.msb(X1)
        MSB_0 = recv(socket=server.client_server_socket)
        MSB = (MSB_0 + MSB_1)
        X = server.add_mode_L_minus_one(X0, X1)
        np.all((X >> 31) == MSB)
        print(';lk')

    if msb_share_check:
        X0 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)
        X1 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)

        X1_converted = server.share_convert(X1)
        MSB_1 = server.msb(X1_converted)
        MSB_0 = recv(socket=server.client_server_socket)
        print('lkj')

    if relu_check:
        X0 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)
        X1 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)
        relu_1 = server.relu(X1)

        relu_0 = recv(socket=server.client_server_socket)

        float_X = (X0 + X1).astype(np.intc)
        float_X[float_X<0] = 0
        relu_real = float_X.astype(server.dtype)
        (relu_0 + relu_1) == relu_real
        print('fds;ljk')

    if stem_check:
        worker = Server(
            client_server_prf=torch.Generator().manual_seed(1),
            server_provider_prf=torch.Generator().manual_seed(3),
            client_server_socket=27123,
            server_client_socket=28124,
            server_provider_socket=23127,
            provider_server_socket=24128,
        )

        base = 10
        precision_fractional = 4

        Ws = [(torch.load(f"/home/yakir/tmp/W{i}.pt") * base ** precision_fractional).to(worker.torch_dtype) for i in range(5)]
        Bs = [(torch.load(f"/home/yakir/tmp/B{i}.pt") * base ** precision_fractional).to(worker.torch_dtype) for i in range(5)]
        I = (torch.load("/home/yakir/tmp/data.pt") * base ** precision_fractional).to(worker.torch_dtype)

        Ws_0 = [torch.randint(low=torch.iinfo(worker.torch_dtype).min // 2,
                              high=torch.iinfo(worker.torch_dtype).max // 2 + 1,
                              size=Ws[i].shape,
                              dtype=worker.torch_dtype,
                              generator=worker.client_server_prf
                              ) for i in range(5)]

        I1 = torch.randint(low=torch.iinfo(worker.torch_dtype).min // 2,
                           high=torch.iinfo(worker.torch_dtype).max // 2 + 1,
                           size=I.shape,
                           dtype=worker.torch_dtype,
                           generator=worker.client_server_prf
                           )

        Ws_1 = [Ws[i] - Ws_0[i] for i in range(5)]

        t0 = time.time()
        a_1 = I1
        a_1 = worker.conv2d(a_1, Ws_1[0], stride=2, bias=Bs[0])
        a_1 = worker.relu(a_1.numpy())

        # Block 0
        identity = a_1
        a_1 = worker.conv2d(a_1, Ws_1[1], stride=1, bias=Bs[1])
        a_1 = worker.relu(a_1.numpy())
        a_1 = worker.conv2d(a_1, Ws_1[2], stride=1, bias=Bs[2])
        a_1 = a_1 + identity
        a_1 = worker.relu(a_1.numpy())

        # Block 1
        identity = a_1
        a_1 = worker.conv2d(a_1, Ws_1[3], stride=1, bias=Bs[3])
        a_1 = worker.relu(a_1.numpy())
        a_1 = worker.conv2d(a_1, Ws_1[4], stride=1, bias=Bs[4])
        a_1 = a_1 + identity
        a_1 = worker.relu(a_1.numpy())

        print(time.time() - t0)
        print(worker.send_time + worker.recv_time)
        a_0 = recv(socket=server.client_server_socket)
        out = torch.from_numpy(a_0) + a_1
        out_float = out.to(torch.float32) / 10000
        out_float[0,:,0,0]
        print('fds')
        # a_1 = server.conv2d(a_1, W2_1, stride=1)
        # a_1 = server.relu(a_1.numpy())
        #
        # a_0 = recv(socket=server.client_server_socket)
        #
        # out = torch.from_numpy(a_0) + a_1
        # out_float = out.to(torch.float32)  / 10000
        #
        # out_real = torch.load("/home/yakir/tmp/data.pt")
        # out_real = torch.relu(torch.conv2d(out_real, torch.load("/home/yakir/tmp/W0.pt"), stride=2, padding=3))
        # out_real = torch.relu(torch.conv2d(out_real, torch.load("/home/yakir/tmp/W1.pt"), stride=1, padding=1))
        # out_real = torch.relu(torch.conv2d(out_real, torch.load("/home/yakir/tmp/W2.pt"), stride=1, padding=1))
        #
        # print('fds')
        # out_1 = server.conv2d(I1, W1, stride=2)
        # out_0 = recv(socket=server.client_server_socket)
        #
        # out_1 = server.relu(out_1.numpy())
        #
        # out_0 = recv(socket=server.client_server_socket)
        #
        # out = torch.from_numpy(out_0) + out_1
        # out_float = out.to(torch.float32)  / 10000
        # out_real = torch.relu(torch.conv2d(torch.load("/home/yakir/tmp/data.pt"), torch.load("/home/yakir/tmp/weight.pt"), stride=2, padding=3))
        # out_float - out_real