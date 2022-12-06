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


class Client:
    def __init__(self,
                 client_server_seed,
                 client_provider_seed,
                 client_server_port,
                 server_client_port,
                 client_provider_port,
                 provider_client_port):

        self.client_server_prf = torch.Generator().manual_seed(client_server_seed)
        self.numpy_client_server_prf = np.random.default_rng(seed=client_server_seed)

        self.client_provider_prf = torch.Generator().manual_seed(client_provider_seed)
        self.numpy_client_provider_prf = np.random.default_rng(seed=client_provider_seed)

        self.crypto_provider_numpy_queue = Receiver(provider_client_port)
        self.crypto_provider_numpy_queue.start()

        self.server_numpy_queue = Receiver(server_client_port)
        self.server_numpy_queue.start()

        self.client2server_queue = Sender(client_server_port)
        self.client2server_queue.start()

        self.client2provider_queue = Sender(client_provider_port)
        self.client2provider_queue.start()

        self.torch_dtype = torch.int64
        self.num_bits = 64
        self.dtype = num_bit_to_dtype[self.num_bits]
        self.min_val = np.iinfo(self.dtype).min
        self.max_val = np.iinfo(self.dtype).max
        self.L_minus_1 = 2 ** self.num_bits - 1

        assert time.time() <1670479523.3295212, "remove //2 and add zero shares"

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
        A_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=X_share.shape, dtype=self.torch_dtype, generator=self.client_provider_prf)
        B_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=Y_share.shape, dtype=self.torch_dtype, generator=self.client_provider_prf)
        C_share = torch.from_numpy(self.crypto_provider_numpy_queue.get())

        E_share = X_share - A_share
        F_share = Y_share - B_share

        E_share_server = torch.from_numpy(self.server_numpy_queue.get())
        self.client2server_queue.put(E_share)
        F_share_server = torch.from_numpy(self.server_numpy_queue.get())
        self.client2server_queue.put(F_share)

        # E_share_server = send_recv(self.server_client_socket, self.client_server_socket, E_share)
        # F_share_server = send_recv(self.server_client_socket, self.client_server_socket, F_share)

        E = E_share_server + E_share
        F = F_share_server + F_share

        # E = E.to("cuda:0")
        # F = F.to("cuda:0")
        # C_share = C_share.to("cuda:0")
        # X_share = X_share.to("cuda:0")
        # Y_share = Y_share.to("cuda:0")

        out = (torch.conv2d(X_share, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
               torch.conv2d(E, Y_share, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + C_share) // 10000

        # out = out.to("cpu")
        return out

    def mult(self, X_share, Y_share):

        A_share = self.numpy_client_provider_prf.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        B_share = self.numpy_client_provider_prf.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        C_share = self.crypto_provider_numpy_queue.get()

        E_share = X_share - A_share
        F_share = Y_share - B_share

        E_share_server = self.server_numpy_queue.get()
        self.client2server_queue.put(E_share)
        F_share_server = self.server_numpy_queue.get()
        self.client2server_queue.put(F_share)

        E = E_share_server + E_share
        F = F_share_server + F_share

        return X_share * F + Y_share * E + C_share

    def share_convert(self, a_0):
        eta_pp = self.numpy_client_server_prf.integers(0, 2, size=a_0.shape, dtype=self.dtype)

        r = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_0 = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_1 = r - r_0

        alpha = (r < r_0).astype(self.dtype)

        a_tild_0 = a_0 + r_0
        beta_0 = (a_tild_0 < a_0).astype(self.dtype)
        self.client2provider_queue.put(a_tild_0)

        delta_0 = self.crypto_provider_numpy_queue.get()

        # execute_secure_compare

        eta_p_0 = self.crypto_provider_numpy_queue.get()

        t0 = eta_pp * eta_p_0
        t1 = self.add_mode_L_minus_one(t0, t0)
        t2 = self.sub_mode_L_minus_one(eta_pp, t1)
        eta_0 = self.add_mode_L_minus_one(eta_p_0, t2)

        t0 = self.add_mode_L_minus_one(delta_0, eta_0)
        t1 = self.sub_mode_L_minus_one(t0, self.dtype(1))
        t2 = self.sub_mode_L_minus_one(t1, alpha)
        theta_0 = self.add_mode_L_minus_one(beta_0, t2)

        y_0 = self.sub_mode_L_minus_one(a_0, theta_0)

        return y_0

    def msb(self, a_0):
        beta = self.numpy_client_server_prf.integers(0, 2, size=a_0.shape, dtype=self.dtype)

        x_0 = self.crypto_provider_numpy_queue.get()
        x_bit_0_0 = self.crypto_provider_numpy_queue.get()

        y_0 = self.add_mode_L_minus_one(a_0, a_0)
        r_0 = self.add_mode_L_minus_one(x_0, y_0)
        r_1 = self.server_numpy_queue.get()
        self.client2server_queue.put(r_0)
        r = self.add_mode_L_minus_one(r_0, r_1)

        # execute_secure_compare

        beta_p_0 = self.crypto_provider_numpy_queue.get()

        gamma_0 = beta_p_0 + (0 * beta) - (2 * beta * beta_p_0)
        delta_0 = x_bit_0_0 + (0 * (r % 2)) - (2 * (r % 2) * x_bit_0_0)

        theta_0 = self.mult(gamma_0, delta_0)
        alpha_0 = gamma_0 + delta_0 - 2 * theta_0

        return alpha_0

    def drelu(self, X0):
        X0_converted = self.share_convert(X0)
        MSB_0 = self.msb(X0_converted)
        return -MSB_0

    def relu(self, X0):
        X0 = X0.astype(np.uint64)
        MSB_0 = self.drelu(X0)
        relu_0 = self.mult(X0, MSB_0)
        return torch.from_numpy(relu_0.astype(np.int64))


if __name__ == "__main__":
    # !/usr/bin/python3
    # global_prf = torch.Generator().manual_seed(0)
    # image = torch.rand(size=(1, 3, 256, 256), generator=global_prf)
    # weight_0 = torch.rand(size=(32, 3, 3, 3), generator=global_prf)
    # weight_1 = torch.rand(size=(32, 32, 3, 3), generator=global_prf)
    # weight_2 = torch.rand(size=(64, 32, 3, 3), generator=global_prf)
    #
    # client = Client(
    #     client_server_prf=torch.Generator().manual_seed(1),
    #     client_provider_prf=torch.Generator().manual_seed(2),
    #     client_server_socket=27123,
    #     server_client_socket=28124,
    #     client_provider_socket=45125,
    #     provider_client_socket=22126,
    # )
    #
    # share_convert_check = False
    # conv_2d_check = False
    # mult_check = False
    # msb_check = False
    # msb_share_check = False
    # relu_check = False
    # stem_check = True
    # if share_convert_check:
    #     rng = np.random.default_rng(seed=0)
    #     a_0 = rng.integers(client.min_val, client.max_val + 1, size=(1000,), dtype=client.dtype)
    #     a_1 = rng.integers(client.min_val, client.max_val + 1, size=(1000,), dtype=client.dtype)
    #     a = a_0 + a_1
    #     y_0 = client.share_convert(a_0)
    #     send(socket=client.client_server_socket, data=y_0)
    #
    # if conv_2d_check:
    #     image_share_server = torch.rand(size=(1, 3, 256, 256), generator=client.client_server_prf)
    #     weight_0_share_client = torch.rand(size=(32, 3, 3, 3), generator=client.client_server_prf)
    #     weight_1_share_client = torch.rand(size=(32, 32, 3, 3), generator=client.client_server_prf)
    #     weight_2_share_client = torch.rand(size=(64, 32, 3, 3), generator=client.client_server_prf)
    #
    #     image_share_client = image - image_share_server
    #
    #     activation_share_client = image_share_client
    #     activation_share_client = client.conv2d(activation_share_client, weight_0_share_client, stride=2)
    #     activation_share_client = client.conv2d(activation_share_client, weight_1_share_client)
    #     activation_share_client = client.conv2d(activation_share_client, weight_2_share_client)
    #
    #     send(socket=client.client_server_socket, data=activation_share_client)
    #
    # if mult_check:
    #     X0 = client.numpy_client_server_prf.integers(client.min_val, client.max_val + 1, size=(1000, ), dtype=client.dtype)
    #     Y0 = client.numpy_client_server_prf.integers(client.min_val, client.max_val + 1, size=(1000, ), dtype=client.dtype)
    #
    #     X1 = client.numpy_client_server_prf.integers(client.min_val, client.max_val + 1, size=(1000, ), dtype=client.dtype)
    #     Y1 = client.numpy_client_server_prf.integers(client.min_val, client.max_val + 1, size=(1000, ), dtype=client.dtype)
    #
    #     Z0 = client.mult(X0, Y0)
    #     import time
    #     # time.sleep(2)
    #     send(socket=client.client_server_socket, data=Z0)
    #
    # if msb_check:
    #     X0 = client.numpy_client_server_prf.integers(client.min_val, client.max_val, size=(1000,), dtype=client.dtype)
    #     X1 = client.numpy_client_server_prf.integers(client.min_val, client.max_val, size=(1000,), dtype=client.dtype)
    #     MSB_0 = client.msb(X0)
    #     send(socket=client.client_server_socket, data=MSB_0)
    #
    # if msb_share_check:
    #     X0 = client.numpy_client_server_prf.integers(client.min_val, client.max_val, size=(1000,), dtype=client.dtype)
    #     X1 = client.numpy_client_server_prf.integers(client.min_val, client.max_val, size=(1000,), dtype=client.dtype)
    #
    #     X0_converted = client.share_convert(X0)
    #     MSB_0 = client.msb(X0_converted)
    #     send(socket=client.client_server_socket, data=MSB_0)
    #
    # if relu_check:
    #     X0 = client.numpy_client_server_prf.integers(client.min_val, client.max_val, size=(1000,), dtype=client.dtype)
    #     X1 = client.numpy_client_server_prf.integers(client.min_val, client.max_val, size=(1000,), dtype=client.dtype)
    #     relu_0 = client.relu(X0)
    #
    #     send(socket=client.client_server_socket, data=relu_0)
    #

    stem_check = True
    if stem_check:
        worker = Client(
            client_server_seed=1,
            client_provider_seed=2,
            client_server_port=27423,
            server_client_port=28134,
            client_provider_port=45125,
            provider_client_port=22126,
        )
        time.sleep(5)

        base = 10
        precision_fractional = 4

        Ws = [(torch.load(f"/home/yakir/tmp/W{i}.pt") * base ** precision_fractional).to(worker.torch_dtype) for i in
              range(5)]
        # Bs = [(torch.load(f"/home/yakir/tmp/B{i}.pt") * base ** precision_fractional).to(worker.torch_dtype) for i in
        #       range(5)]
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

        # Ws_1 = [Ws[i] - Ws_0[i] for i in range(5)]

        I0 = I - I1
        a_1 = I0
        a_1 = worker.conv2d(a_1, Ws_0[0], stride=2)
        a_1 = worker.relu(a_1.numpy())

        # Block 0
        identity = a_1
        a_1 = worker.conv2d(a_1, Ws_0[1], stride=1)
        a_1 = worker.relu(a_1.numpy())
        a_1 = worker.conv2d(a_1, Ws_0[2], stride=1)
        a_1 = a_1 + identity
        a_1 = worker.relu(a_1.numpy())

        # Block 1
        identity = a_1
        a_1 = worker.conv2d(a_1, Ws_0[3], stride=1)
        a_1 = worker.relu(a_1.numpy())
        a_1 = worker.conv2d(a_1, Ws_0[4], stride=1)
        a_1 = a_1 + identity
        a_1 = worker.relu(a_1.numpy())

        worker.client2server_queue.put(a_1)
        # base = 10
        # precision_fractional = 4
        #
        # W0 = torch.load("/home/yakir/tmp/W0.pt")
        # W0 = (W0 * base ** precision_fractional).to(client.torch_dtype)
        #
        # W1 = torch.load("/home/yakir/tmp/W1.pt")
        # W1 = (W1 * base ** precision_fractional).to(client.torch_dtype)
        #
        # W2 = torch.load("/home/yakir/tmp/W2.pt")
        # W2 = (W2 * base ** precision_fractional).to(client.torch_dtype)
        #
        # I = torch.load("/home/yakir/tmp/data.pt")
        # I = (I * base ** precision_fractional).to(client.torch_dtype)
        #
        # W0_0 = torch.randint(low=torch.iinfo(client.torch_dtype).min // 2,
        #                      high=torch.iinfo(client.torch_dtype).max // 2 + 1,
        #                      size=W0.shape,
        #                      dtype=client.torch_dtype,
        #                      generator=client.client_server_prf
        #                      )
        # W1_0 = torch.randint(low=torch.iinfo(client.torch_dtype).min // 2,
        #                      high=torch.iinfo(client.torch_dtype).max // 2 + 1,
        #                      size=W1.shape,
        #                      dtype=client.torch_dtype,
        #                      generator=client.client_server_prf
        #                      )
        # W2_0 = torch.randint(low=torch.iinfo(client.torch_dtype).min // 2,
        #                      high=torch.iinfo(client.torch_dtype).max // 2 + 1,
        #                      size=W2.shape,
        #                      dtype=client.torch_dtype,
        #                      generator=client.client_server_prf
        #                      )
        # I1 = torch.randint(low=torch.iinfo(client.torch_dtype).min // 2,
        #                    high=torch.iinfo(client.torch_dtype).max // 2 + 1,
        #                    size=I.shape,
        #                    dtype=client.torch_dtype,
        #                    generator=client.client_server_prf
        #                    )
        #
        # W0_1 = W0 - W0_0
        # W1_1 = W1 - W1_0
        # W2_1 = W2 - W2_0
        # I0 = I - I1
        #
        # a_0 = I0
        # a_0 = client.conv2d(a_0, W0_0, stride=2)
        # a_0 = client.relu(a_0.numpy())
        #
        # a_0 = client.conv2d(a_0, W1_0, stride=1)
        # a_0 = client.relu(a_0.numpy())
        #
        # a_0 = client.conv2d(a_0, W2_0, stride=1)
        # a_0 = client.relu(a_0.numpy())
        #
        # send(socket=client.client_server_socket, data=a_0)
        #
        # # out_0 = client.relu(out_0.numpy())
        # # send(socket=client.client_server_socket, data=out_0)
        # # print('fds')
        #
        #
