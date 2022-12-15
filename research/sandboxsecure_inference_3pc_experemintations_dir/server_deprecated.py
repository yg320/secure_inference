import torch
from research.communication.utils import Sender, Receiver
import time
import numpy as np
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets, fuse_conv_bn


from research.communication.utils import Sender, Receiver
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets


class SecureConv2DServer(SecureModule):
    def __init__(self, W, stride, dilation, crypto_assets: CryptoAssets, network_assets: NetworkAssets):
        super(SecureConv2DServer, self).__init__(crypto_assets, network_assets)

        self.W_share = W
        self.stride = stride
        self.dilation = dilation

    def forward(self, X_share):

        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert self.W_share.shape[1] == X_share.shape[1]
        assert X_share.shape[2] == X_share.shape[3]
        assert self.stride[0] == self.stride[1]

        _, _, _, f = self.W_share.shape
        padding = (f - 1) // 2

        A_share = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_12_torch)
        B_share = self.crypto_assets.get_random_tensor_over_L(shape=self.W_share.shape, prf=self.crypto_assets.prf_12_torch)

        E_share = X_share - A_share
        F_share = self.W_share - B_share

        self.network_assets.sender_01.put(E_share)
        E_share_client = torch.from_numpy(self.network_assets.receiver_01.get())
        self.network_assets.sender_01.put(F_share)
        F_share_client = torch.from_numpy(self.network_assets.receiver_01.get())

        E = E_share_client + E_share
        F = F_share_client + F_share

        out = \
            torch.conv2d(E, self.W_share - F, bias=None, stride=self.stride, padding=padding, dilation=1, groups=1) + \
            torch.conv2d(X_share, F, bias=None, stride=self.stride, padding=padding, dilation=1, groups=1)

        C_share = self.crypto_assets.get_random_tensor_over_L(shape=self.out.shape, prf=self.crypto_assets.prf_12_torch)

        out = out + C_share
        out = out // self.trunc
        out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return out


class ShareConvertServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertServer, self).__init__(crypto_assets, network_assets)

    def forward(self, a_1):
        eta_pp = self.crypto_assets.prf_01_numpy.integers(0, 2, size=a_1.shape, dtype=self.dtype)

        r = self.crypto_assets.prf_01_numpy.integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
        r_0 = self.crypto_assets.prf_01_numpy.integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
        r_1 = r - r_0

        a_tild_1 = a_1 + r_1
        beta_1 = (a_tild_1 < a_1).astype(self.dtype)

        self.network_assets.sender_12.put(a_tild_1)

        delta_1 = self.network_assets.receiver_12.get()

        self.network_assets.sender_12.put(r)
        self.network_assets.sender_12.put(eta_pp)
        # execute_secure_compare

        eta_p_1 = self.network_assets.receiver_12.get()

        t0 = eta_pp * eta_p_1
        t1 = self.add_mode_L_minus_one(t0, t0)
        eta_1 = self.sub_mode_L_minus_one(eta_p_1, t1)

        t0 = self.add_mode_L_minus_one(delta_1, eta_1)
        theta_1 = self.add_mode_L_minus_one(beta_1, t0)

        y_1 = self.sub_mode_L_minus_one(a_1, theta_1)

        return y_1


class SecureMultiplicationServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationServer, self).__init__(crypto_assets, network_assets)

    def forward(self, X_share, Y_share):

        A_share = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        B_share = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        C_share = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        E = E_share_client + E_share
        F = F_share_client + F_share

        return - E * F + X_share * F + Y_share * E + C_share


class SecureMSBServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBServer, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMSBServer(crypto_assets, network_assets)

    def forward(self, a_1):
        beta = self.crypto_assets.prf_01_numpy.integers(0, 2, size=a_1.shape, dtype=self.dtype)

        x_1 = self.crypto_provider_numpy_queue.get()
        x_bit_0_1 = self.crypto_provider_numpy_queue.get()

        y_1 = self.add_mode_L_minus_one(a_1, a_1)
        r_1 = self.add_mode_L_minus_one(x_1, y_1)

        self.network_assets.sender_01.put(r_1)

        r_0 = self.network_assets.receiver_01.get()

        r = self.add_mode_L_minus_one(r_0, r_1)

        self.network_assets.sender_12.put(r)
        self.network_assets.sender_12.put(beta)

        # execute_secure_compare
        beta_p_1 = self.network_assets.receiver_12.get()

        gamma_1 = beta_p_1 + (1 * beta) - (2 * beta * beta_p_1)
        delta_1 = x_bit_0_1 + (1 * (r % 2)) - (2 * (r % 2) * x_bit_0_1)

        theta_1 = self.mult(gamma_1, delta_1)
        alpha_1 = gamma_1 + delta_1 - 2 * theta_1

        return alpha_1


class SecureDReLUServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUServer, self).__init__(crypto_assets, network_assets)

        self.share_convert = ShareConvertServer(crypto_assets, network_assets)
        self.msb = SecureMSBServer(crypto_assets, network_assets)

    def forward(self, X_share):
        X1_converted = self.share_convert(X_share)
        MSB_1 = self.msb(X1_converted)
        return 1 - MSB_1

class SecureReLUServer(torch.nn.Module):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUServer, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUServer(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)

    def forward(self, X_share):
        X_share = X_share.numpy().astype(np.uint64)
        MSB_0 = self.DReLU(X_share)
        relu_0 = self.mult(X_share, MSB_0)
        return torch.from_numpy(relu_0.astype(np.int64))











#
# class Server:
#     def __init__(self, client_server_seed,
#                  server_provider_seed,
#                  client_server_port,
#                  server_client_port,
#                  server_provider_port,
#                  provider_server_port):
#
#         self.client_server_prf = torch.Generator().manual_seed(client_server_seed)
#         self.numpy_client_server_prf = np.random.default_rng(seed=client_server_seed)
#
#         self.server_provider_prf = torch.Generator().manual_seed(server_provider_seed)
#         self.numpy_server_provider_prf = np.random.default_rng(seed=server_provider_seed)
#
#         self.crypto_provider_numpy_queue = Receiver(provider_server_port)
#         self.crypto_provider_numpy_queue.start()
#
#         self.client_numpy_queue = Receiver(client_server_port)
#         self.client_numpy_queue.start()
#
#         self.server2provider_queue = Sender(server_provider_port)
#         self.server2provider_queue.start()
#
#         self.server2client_queue = Sender(server_client_port)
#         self.server2client_queue.start()
#
#         self.torch_dtype = torch.int64
#         self.num_bits = 64
#         self.dtype = num_bit_to_dtype[self.num_bits]
#         self.min_val = np.iinfo(self.dtype).min
#         self.max_val = np.iinfo(self.dtype).max
#         self.L_minus_1 = 2 ** self.num_bits - 1
#
#         assert time.time() <1670479523.3295212, "remove //2 and add zero shares"
#
#         self.send_time = 0
#         self.recv_time = 0
#
#     def add_mode_L_minus_one(self, a, b):
#         ret = a + b
#         ret[ret < a] += self.dtype(1)
#         ret[ret == self.L_minus_1] = self.dtype(0)
#         return ret
#
#     def sub_mode_L_minus_one(self, a, b):
#         ret = a - b
#         ret[b > a] -= self.dtype(1)
#         return ret
#
#     def get_random_tensor(self, shape, prf):
#         return torch.randint(
#             low=torch.iinfo(worker.torch_dtype).min // 2,
#             high=torch.iinfo(worker.torch_dtype).max // 2 + 1,
#             size=shape,
#             dtype=worker.torch_dtype,
#             generator=prf
#         )
#
#     def conv2d(self, X_share, Y_share, stride=1, bias=None):
#         assert Y_share.shape[2] == Y_share.shape[3]
#         assert Y_share.shape[1] == X_share.shape[1]
#         assert X_share.shape[2] == X_share.shape[3]
#
#         b, i, m, _ = X_share.shape
#         m = m // stride
#         o, _, _, f = Y_share.shape
#         output_shape = (b, o, m, m)
#         padding = (f - 1) // 2
#
#         A_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=X_share.shape, dtype=self.torch_dtype, generator=self.server_provider_prf)
#         B_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=Y_share.shape, dtype=self.torch_dtype, generator=self.server_provider_prf)
#         C_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=output_shape,  dtype=self.torch_dtype, generator=self.server_provider_prf)
#
#         E_share = X_share - A_share
#         F_share = Y_share - B_share
#
#         t0 = time.time()
#         self.server2client_queue.put( E_share)
#         self.send_time += (time.time() - t0)
#
#         t0 = time.time()
#         x = self.client_numpy_queue.get()
#         self.recv_time += (time.time() - t0)
#
#         E_share_client = torch.from_numpy(x)
#
#         t0 = time.time()
#         self.server2client_queue.put( F_share)
#         self.send_time += (time.time() - t0)
#
#         t0 = time.time()
#         x = self.client_numpy_queue.get()
#         self.recv_time += (time.time() - t0)
#
#         F_share_client = torch.from_numpy(x)
#
#
#         E = E_share_client + E_share
#         F = F_share_client + F_share
#
#         # E = E.to("cuda:0")
#         # F = F.to("cuda:0")
#         # C_share = C_share.to("cuda:0")
#         # X_share = X_share.to("cuda:0")
#         # Y_share = Y_share.to("cuda:0")
#
#         # out = (-torch.conv2d(E, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
#         #         torch.conv2d(X_share, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
#         #         torch.conv2d(E, Y_share, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + C_share) // 10000
#
#         out = (torch.conv2d(E, Y_share - F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
#                torch.conv2d(X_share, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + C_share) // 10000
#
#         # out.to("cpu")
#         out = out + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         return out
#
#     def mult(self, X_share, Y_share):
#
#         A_share = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
#         B_share = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
#         C_share = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
#
#         E_share = X_share - A_share
#         F_share = Y_share - B_share
#
#         t0 = time.time()
#         self.server2client_queue.put( E_share)
#         self.send_time += (time.time() - t0)
#
#         t0 = time.time()
#         E_share_client = self.client_numpy_queue.get()
#         self.recv_time += (time.time() - t0)
#
#         t0 = time.time()
#         self.server2client_queue.put( F_share)
#         self.send_time += (time.time() - t0)
#
#         t0 = time.time()
#         F_share_client = self.client_numpy_queue.get()
#         self.recv_time += (time.time() - t0)
#
#         E = E_share_client + E_share
#         F = F_share_client + F_share
#
#         return - E * F + X_share * F + Y_share * E + C_share
#
#     def share_convert(self, a_1):
#         eta_pp = self.numpy_client_server_prf.integers(0, 2, size=a_1.shape, dtype=self.dtype)
#
#         r = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
#         r_0 = self.numpy_client_server_prf.integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
#         r_1 = r - r_0
#
#         alpha = (r < r_0).astype(self.dtype)
#
#         a_tild_1 = a_1 + r_1
#         beta_1 = (a_tild_1 < a_1).astype(self.dtype)
#
#
#         t0 = time.time()
#         self.server2provider_queue.put( a_tild_1)
#         self.send_time += (time.time() - t0)
#
#
#         t0 = time.time()
#         delta_1 = self.crypto_provider_numpy_queue.get()
#         self.recv_time += (time.time() - t0)
#
#
#         t0 = time.time()
#         self.server2provider_queue.put( r)
#         self.send_time += (time.time() - t0)
#
#         t0 = time.time()
#         self.server2provider_queue.put( eta_pp)
#         self.send_time += (time.time() - t0)
#
#         # execute_secure_compare
#         t0 = time.time()
#         eta_p_1 = self.crypto_provider_numpy_queue.get()
#         self.recv_time += (time.time() - t0)
#
#         t0 = eta_pp * eta_p_1
#         t1 = self.add_mode_L_minus_one(t0, t0)
#         eta_1 = self.sub_mode_L_minus_one(eta_p_1, t1)
#
#         t0 = self.add_mode_L_minus_one(delta_1, eta_1)
#         theta_1 = self.add_mode_L_minus_one(beta_1, t0)
#
#         y_1 = self.sub_mode_L_minus_one(a_1, theta_1)
#
#         return y_1
#
#     def msb(self, a_1):
#         beta = self.numpy_client_server_prf.integers(0, 2, size=a_1.shape, dtype=self.dtype)
#         t0 = time.time()
#         x_1 = self.crypto_provider_numpy_queue.get()
#         self.recv_time += (time.time() - t0)
#
#         t0 = time.time()
#         x_bit_0_1 = self.crypto_provider_numpy_queue.get()
#         self.recv_time += (time.time() - t0)
#
#         y_1 = self.add_mode_L_minus_one(a_1, a_1)
#         r_1 = self.add_mode_L_minus_one(x_1, y_1)
#
#
#         t0 = time.time()
#         self.server2client_queue.put( r_1)
#         self.send_time += (time.time() - t0)
#
#         t0 = time.time()
#         r_0 = self.client_numpy_queue.get()
#         self.recv_time += (time.time() - t0)
#
#         r = self.add_mode_L_minus_one(r_0, r_1)
#
#         t0 = time.time()
#         self.server2provider_queue.put( r)
#         self.send_time += (time.time() - t0)
#
#         t0 = time.time()
#         self.server2provider_queue.put( beta)
#         self.send_time += (time.time() - t0)
#
#
#         # execute_secure_compare
#         t0 = time.time()
#         beta_p_1 = self.crypto_provider_numpy_queue.get()
#         self.recv_time += (time.time() - t0)
#
#         gamma_1 = beta_p_1 + (1 * beta) - (2 * beta * beta_p_1)
#         delta_1 = x_bit_0_1 + (1 * (r % 2)) - (2 * (r % 2) * x_bit_0_1)
#
#         theta_1 = self.mult(gamma_1, delta_1)
#         alpha_1 = gamma_1 + delta_1 - 2 * theta_1
#
#         return alpha_1
#
#     def drelu(self, X1):
#         X1_converted = self.share_convert(X1)
#         MSB_1 = self.msb(X1_converted)
#         return 1 - MSB_1
#
#     def relu(self, X1):
#         X1 = X1.astype(np.uint64)
#
#         MSB_1 = self.drelu(X1)
#         relu_1 = self.mult(X1, MSB_1)
#         return torch.from_numpy(relu_1.astype(np.int64))
#
#     def fuse_and_encrypt(self, conv_module, batch_norm_module, trunc = 10000):
#         # TODO: this was copied from somewhere
#         fusedconv = torch.nn.Conv2d(
#             conv_module.in_channels,
#             conv_module.out_channels,
#             kernel_size=conv_module.kernel_size,
#             stride=conv_module.stride,
#             padding=conv_module.padding,
#             bias=True
#         )
#         fusedconv.weight.requires_grad = False
#         fusedconv.bias.requires_grad = False
#         w_conv = conv_module.weight.clone().view(conv_module.out_channels, -1)
#         w_bn = torch.diag(
#             batch_norm_module.weight.div(torch.sqrt(batch_norm_module.eps + batch_norm_module.running_var)))
#         fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
#         if conv_module.bias is not None:
#             b_conv = conv_module.bias
#         else:
#             b_conv = torch.zeros(conv_module.weight.size(0))
#         b_bn = batch_norm_module.bias - batch_norm_module.weight.mul(batch_norm_module.running_mean).div(
#             torch.sqrt(batch_norm_module.running_var + batch_norm_module.eps))
#         fusedconv.bias.copy_(torch.matmul(w_bn, b_conv) + b_bn)
#
#         W, B = fusedconv.weight, fusedconv.bias
#         W = (W * trunc).to(worker.torch_dtype)
#         B = (B * trunc).to(worker.torch_dtype)
#
#         W_shared = W - self.get_random_tensor(W.shape, prf=self.client_server_prf)
#
#         return W_shared, B


# class SecureConv2DServer(torch.nn.Module):
#     def __init__(self, W, bias, stride, dilation, prf_12_torch, client_numpy_queue, server2client_queue):
#         super(SecureConv2DServer, self).__init__()
#         self.W_share = W
#         self.bias = bias
#         self.stride = stride
#         self.dilation = dilation
#
#         self.trunc = 10000
#         self.prf_12_torch = prf_12_torch
#
#         self.client_numpy_queue = client_numpy_queue
#         self.server2client_queue = server2client_queue
#
#         self.torch_dtype = torch.int64
#         self.num_bits = 64
#         self.dtype = num_bit_to_dtype[self.num_bits]
#         self.min_val = np.iinfo(self.dtype).min
#         self.max_val = np.iinfo(self.dtype).max
#         self.L_minus_1 = 2 ** self.num_bits - 1
#
#
#     def forward(self, X_share):
#         assert self.W_share.shape[2] == self.W_share.shape[3]
#         assert self.W_share.shape[1] == X_share.shape[1]
#         assert X_share.shape[2] == X_share.shape[3]
#         assert self.stride[0] == self.stride[1]
#         b, i, m, _ = X_share.shape
#         m = m // self.stride[0]
#         o, _, _, f = self.W_share.shape
#         output_shape = (b, o, m, m)
#         padding = (f - 1) // 2
#
#         A_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2,
#                                 high=torch.iinfo(self.torch_dtype).max // 2 + 1,
#                                 size=X_share.shape,
#                                 dtype=self.torch_dtype,
#                                 generator=self.prf_12_torch)
#
#         B_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2,
#                                 high=torch.iinfo(self.torch_dtype).max // 2 + 1,
#                                 size=self.W_share.shape,
#                                 dtype=self.torch_dtype,
#                                 generator=self.prf_12_torch)
#
#         C_share = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2,
#                                 high=torch.iinfo(self.torch_dtype).max // 2 + 1,
#                                 size=output_shape,
#                                 dtype=self.torch_dtype,
#                                 generator=self.prf_12_torch)
#
#         E_share = X_share - A_share
#         F_share = self.W_share - B_share
#
#         self.server2client_queue.put(E_share)
#         E_share_client = torch.from_numpy(self.client_numpy_queue.get())
#         self.server2client_queue.put(F_share)
#         F_share_client = torch.from_numpy(self.client_numpy_queue.get())
#
#         E = E_share_client + E_share
#         F = F_share_client + F_share
#
#         out = \
#             torch.conv2d(E, self.W_share - F, bias=None, stride=self.stride, padding=padding, dilation=1, groups=1) + \
#             torch.conv2d(X_share, F, bias=None, stride=self.stride, padding=padding, dilation=1, groups=1) + C_share
#
#         out = out // self.trunc
#         out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         return out




if __name__ == "__main__":
    from research.distortion.utils import get_model
    from research.pipeline.backbones.secure_resnet import AvgPoolResNet

    image_shape = (1, 3, 64, 64)
    model = get_model(
        config="/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16.py",
        gpu_id=None,
        checkpoint_path="/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/iter_80000.pth"
    )

    sender_01 = Sender(12345)
    sender_02 = Sender(12346)
    sender_12 = Sender(12347)
    receiver_01 = Receiver(12348)
    receiver_02 = Receiver(12349)
    receiver_12 = Receiver(12350)

    crypto_assets = CryptoAssets(
        prf_01_numpy=np.random.default_rng(seed=0),
        prf_02_numpy=np.random.default_rng(seed=1),
        prf_12_numpy=np.random.default_rng(seed=2),
        prf_01_torch=torch.Generator().manual_seed(seed=0),
        prf_02_torch=torch.Generator().manual_seed(seed=1),
        prf_12_torch=torch.Generator().manual_seed(seed=2),
    )

    network_assets = NetworkAssets(
        sender_01=sender_01,
        sender_02=sender_02,
        sender_12=sender_12,
        receiver_01=receiver_01,
        receiver_02=receiver_02,
        receiver_12=receiver_12
    )

    W, B = fuse_conv_bn(conv_module=model.backbone.stem[0], batch_norm_module=model.backbone.stem[1])
    W = (W * crypto_assets.trunc).to(crypto_assets.torch_dtype)
    B = (B * crypto_assets.trunc).to(crypto_assets.torch_dtype)
    W = W - crypto_assets.get_random_tensor_over_L(W.shape, prf=crypto_assets.prf_01_torch)
    model.backbone.stem[0] = SecureConv2DServer(
        W=W,
        bias=B,
        stride=model.backbone.stem[0].stride,
        dilation=model.backbone.stem[0].dilation,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )

    model.backbone.stem[1] = torch.nn.Identity()
    model.backbone.stem[2] = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

    W, B = fuse_conv_bn(conv_module=model.backbone.stem[3], batch_norm_module=model.backbone.stem[4])
    W = (W * crypto_assets.trunc).to(crypto_assets.torch_dtype)
    B = (B * crypto_assets.trunc).to(crypto_assets.torch_dtype)
    W = W - crypto_assets.get_random_tensor_over_L(W.shape, prf=crypto_assets.prf_01_torch)
    model.backbone.stem[3] = SecureConv2DServer(
        W=W,
        bias=B,
        stride=model.backbone.stem[3].stride,
        dilation=model.backbone.stem[3].dilation,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )

    model.backbone.stem[4] = torch.nn.Identity()
    model.backbone.stem[5] = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

    W, B = fuse_conv_bn(conv_module=model.backbone.stem[6], batch_norm_module=model.backbone.stem[7])
    W = (W * crypto_assets.trunc).to(crypto_assets.torch_dtype)
    B = (B * crypto_assets.trunc).to(crypto_assets.torch_dtype)
    W = W - crypto_assets.get_random_tensor_over_L(W.shape, prf=crypto_assets.prf_01_torch)
    model.backbone.stem[6] = SecureConv2DServer(
        W=W,
        bias=B,
        stride=model.backbone.stem[6].stride,
        dilation=model.backbone.stem[6].dilation,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )

    model.backbone.stem[7] = torch.nn.Identity()
    model.backbone.stem[8] = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

    I1 = crypto_assets.get_random_tensor_over_L(image_shape, prf=crypto_assets.prf_01_torch)

    model.backbone.stem(I1)
    assert False

    # share_convert_check = False
    # conv_2d_check = False
    # mult_check = False
    # msb_check = False
    # msb_share_check = False
    # relu_check = False
    # stem_check = True
    #
    # server = Server(
    #     client_server_prf=torch.Generator().manual_seed(1),
    #     server_provider_prf=torch.Generator().manual_seed(3),
    #     client_server_socket=27123,
    #     server_client_socket=28124,
    #     server_provider_socket=23127,
    #     provider_server_socket=24128,
    # )
    #
    # if share_convert_check:
    #     rng = np.random.default_rng(seed=0)
    #     a_0 = rng.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
    #     a_1 = rng.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
    #     a = a_0 + a_1
    #     y_1 = server.share_convert(a_1)
    #     y_0 = recv(server.client_server_socket)
    #
    #     print(np.all(server.add_mode_L_minus_one(y_0, y_1) == a))
    #     print('fd')
    #
    # if conv_2d_check:
    #     global_prf = torch.Generator().manual_seed(0)
    #     image = torch.rand(size=(1, 3, 256, 256), generator=global_prf)
    #     weight_0 = torch.rand(size=(32, 3, 3, 3), generator=global_prf)
    #     weight_1 = torch.rand(size=(32, 32, 3, 3), generator=global_prf)
    #     weight_2 = torch.rand(size=(64, 32, 3, 3), generator=global_prf)
    #
    #     image_share_server = torch.rand(size=(1, 3, 256, 256), generator=server.client_server_prf)
    #     weight_0_share_client = torch.rand(size=(32, 3, 3, 3), generator=server.client_server_prf)
    #     weight_1_share_client = torch.rand(size=(32, 32, 3, 3), generator=server.client_server_prf)
    #     weight_2_share_client = torch.rand(size=(64, 32, 3, 3), generator=server.client_server_prf)
    #
    #     weight_0_share_server = weight_0 - weight_0_share_client
    #     weight_1_share_server = weight_1 - weight_1_share_client
    #     weight_2_share_server = weight_2 - weight_2_share_client
    #
    #     t0 = time.time()
    #     activation_share_server = image_share_server
    #     activation_share_server = server.conv2d(activation_share_server, weight_0_share_server, stride=2)
    #     activation_share_server = server.conv2d(activation_share_server, weight_1_share_server)
    #     activation_share_server = server.conv2d(activation_share_server, weight_2_share_server)
    #     print(time.time() - t0)
    #     activation_share_client = recv(server.client_server_socket)
    #
    #     activation_recon = activation_share_client + activation_share_server
    #     t0 = time.time()
    #     activation_non_secure = torch.conv2d(torch.conv2d(torch.conv2d(image, weight_0, padding=1, stride=2), weight_1, padding=1), weight_2, padding=1)
    #     print(time.time() - t0)
    #     print((activation_non_secure - (activation_share_server + activation_share_client)).abs().max())
    #     print('fdslkj')
    #
    # if mult_check:
    #     X0 = server.numpy_client_server_prf.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
    #     Y0 = server.numpy_client_server_prf.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
    #     X1 = server.numpy_client_server_prf.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
    #     Y1 = server.numpy_client_server_prf.integers(server.min_val, server.max_val + 1, size=(1000,), dtype=server.dtype)
    #
    #     Z1 = server.mult(X1, Y1)
    #     Z0 = recv(server.client_server_socket)
    #     assert np.all((X0 + X1) * (Y0 + Y1) == (Z0 + Z1))
    #     print('fds')
    #
    # if msb_check:
    #     X0 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)
    #     X1 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)
    #
    #     MSB_1 = server.msb(X1)
    #     MSB_0 = recv(socket=server.client_server_socket)
    #     MSB = (MSB_0 + MSB_1)
    #     X = server.add_mode_L_minus_one(X0, X1)
    #     np.all((X >> 31) == MSB)
    #     print(';lk')
    #
    # if msb_share_check:
    #     X0 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)
    #     X1 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)
    #
    #     X1_converted = server.share_convert(X1)
    #     MSB_1 = server.msb(X1_converted)
    #     MSB_0 = recv(socket=server.client_server_socket)
    #     print('lkj')
    #
    # if relu_check:
    #     X0 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)
    #     X1 = server.numpy_client_server_prf.integers(server.min_val, server.max_val, size=(1000,), dtype=server.dtype)
    #     relu_1 = server.relu(X1)
    #
    #     relu_0 = recv(socket=server.client_server_socket)
    #
    #     float_X = (X0 + X1).astype(np.intc)
    #     float_X[float_X<0] = 0
    #     relu_real = float_X.astype(server.dtype)
    #     (relu_0 + relu_1) == relu_real
    #     print('fds;ljk')
    stem_check = True
    if stem_check:
        worker = Server(
            client_server_seed=1,
            server_provider_seed=3,
            client_server_port=27423,
            server_client_port=28134,
            server_provider_port=23129,
            provider_server_port=24328,
        )
        time.sleep(5)

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

        a_1 = I1
        cur_time = time.time()
        a_1 = worker.conv2d(a_1, Ws_1[0], stride=2, bias=Bs[0])
        print(f"conv took - {time.time() - cur_time}")

        cur_time = time.time()
        a_1 = worker.relu(a_1.numpy())
        print(f"relu took - {time.time() - cur_time}")
        # Block 0
        identity = a_1

        cur_time = time.time()
        a_1 = worker.conv2d(a_1, Ws_1[1], stride=1, bias=Bs[1])
        print(f"conv took - {time.time() - cur_time}")

        a_1 = worker.relu(a_1.numpy())

        cur_time = time.time()
        a_1 = worker.conv2d(a_1, Ws_1[2], stride=1, bias=Bs[2])
        print(f"conv took - {time.time() - cur_time}")
        a_1 = a_1 + identity

        cur_time = time.time()
        a_1 = worker.relu(a_1.numpy())
        print(f"relu took - {time.time() - cur_time}")

        # Block 1
        identity = a_1
        cur_time = time.time()

        a_1 = worker.conv2d(a_1, Ws_1[3], stride=1, bias=Bs[3])
        print(f"conv took - {time.time() - cur_time}")
        cur_time = time.time()
        a_1 = worker.relu(a_1.numpy())
        print(f"relu took - {time.time() - cur_time}")
        cur_time = time.time()

        a_1 = worker.conv2d(a_1, Ws_1[4], stride=1, bias=Bs[4])
        print(f"conv took - {time.time() - cur_time}")
        a_1 = a_1 + identity
        cur_time = time.time()
        a_1 = worker.relu(a_1.numpy())
        print(f"relu took - {time.time() - cur_time}")

        print(worker.send_time + worker.recv_time)
        a_0 = worker.client_numpy_queue.get()
        out = torch.from_numpy(a_0) + a_1
        out_float = out.to(torch.float32) / 10000
        print(out_float[0,:,0,0])
        print('fds')

        worker.crypto_provider_numpy_queue.make_stop()
        worker.client_numpy_queue.make_stop()
        worker.server2client_queue.make_stop()
        worker.server2provider_queue.make_stop()
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