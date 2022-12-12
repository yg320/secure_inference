import torch
from research.communication.utils import Sender, Receiver
import numpy as np
import time

from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets, fuse_conv_bn


from research.communication.utils import Sender, Receiver
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets


class SecureConv2DCryptoProvider(SecureModule):
    def __init__(self, W_shape, stride, dilation, crypto_assets: CryptoAssets, network_assets: NetworkAssets):
        super(SecureConv2DCryptoProvider, self).__init__(crypto_assets, network_assets)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation

    def forward(self, X_share):
        _, _, _, f = self.W_shape
        padding = (f - 1) // 2

        A_share_1 = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_12_torch)
        B_share_1 = self.crypto_assets.get_random_tensor_over_L(shape=self.W_shape, prf=self.crypto_assets.prf_12_torch)
        A_share_0 = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_02_torch)
        B_share_0 = self.crypto_assets.get_random_tensor_over_L(shape=self.W_shape, prf=self.crypto_assets.prf_02_torch)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1

        C = torch.conv2d(A, B, bias=None, stride=self.stride, padding=padding, dilation=1, groups=1)

        C_share_1 = self.crypto_assets.get_random_tensor_over_L(shape=C.shape, prf=self.crypto_assets.prf_12_torch)
        C_share_0 = C - C_share_1

        self.network_assets.sender_02.put(C_share_0)

        return C_share_0


class ShareConvertCryptoProvider(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertCryptoProvider, self).__init__(crypto_assets, network_assets)

    def share_convert(self, size):

        a_tild_0 = self.network_assets.receiver_02.get()
        a_tild_1 = self.network_assets.receiver_12.get()

        x = (a_tild_0 + a_tild_1)
        delta = (x < a_tild_0).astype(self.dtype)

        delta_0 = self.crypto_assets.private_prf_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        delta_1 = self.sub_mode_L_minus_one(delta, delta_0)

        self.network_assets.sender_02.put(delta_0)
        self.network_assets.sender_12.put(delta_1)

        r = self.network_assets.receiver_12.get()
        eta_pp = self.network_assets.receiver_12.get()
        eta_p = eta_pp ^ (x > (r - 1))

        eta_p_0 = self.crypto_assets.private_prf_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        eta_p_1 = self.sub_mode_L_minus_one(eta_p, eta_p_0)

        self.network_assets.sender_02.put(eta_p_0)
        self.network_assets.sender_12.put(eta_p_1)

        return


class SecureMultiplicationCryptoProvider(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationCryptoProvider, self).__init__(crypto_assets, network_assets)

    def forward(self, shape):

        A_share_1 = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
        B_share_1 = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
        C_share_1 = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
        A_share_0 = self.crypto_assets.prf_02_numpy.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
        B_share_0 = self.crypto_assets.prf_02_numpy.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1

        C_share_0 = A * B - C_share_1

        self.network_assets.sender_02.put(C_share_0)


class SecureMSBCryptoProvider(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBCryptoProvider, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMSBCryptoProvider(crypto_assets, network_assets)

    def forward(self, size):
        x = self.crypto_assets.private_prf_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        x_0 = self.crypto_assets.private_prf_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        x_1 = self.sub_mode_L_minus_one(x, x_0)

        x_bit0 = x % 2
        x_bit_0_0 = self.crypto_assets.private_prf_numpy.integers(self.min_val, self.max_val + 1, size=size, dtype=self.dtype)
        x_bit_0_1 = x_bit0 - x_bit_0_0

        self.network_assets.sender_02.put(x_0)
        self.network_assets.sender_02.put(x_bit_0_0)

        self.network_assets.sender_12.put(x_1)
        self.network_assets.sender_12.put(x_bit_0_1)

        r =    self.network_assets.receiver_12.get()
        beta = self.network_assets.receiver_12.get()

        beta_p = beta ^ (x > r)
        beta_p_0 = self.crypto_assets.private_prf_numpy.integers(self.min_val, self.max_val + 1, size=size, dtype=self.dtype)
        beta_p_1 = beta_p - beta_p_0

        self.network_assets.sender_02.put(beta_p_0)
        self.network_assets.sender_12.put(beta_p_1)

        self.mult(size)
        return


class SecureDReLUCryptoProvider(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUCryptoProvider, self).__init__(crypto_assets, network_assets)

        self.share_convert = SecureDReLUCryptoProvider(crypto_assets, network_assets)
        self.msb = SecureDReLUCryptoProvider(crypto_assets, network_assets)

    def forward(self, X_share):
        self.share_convert(X_share.shape)
        self.msb(X_share.shape)
        return X_share


class SecureReLUCryptoProvider(torch.nn.Module):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUCryptoProvider, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureReLUCryptoProvider(crypto_assets, network_assets)
        self.mult = SecureReLUCryptoProvider(crypto_assets, network_assets)

    def forward(self, X_share):
        self.drelu(X_share.shape)
        self.mult(X_share.shape)
        return X_share





# num_bit_to_dtype = {
#     8: np.ubyte,
#     16: np.ushort,
#     32: np.uintc,
#     64: np.ulonglong
# }
#
# class CryptoProvider:
#     def __init__(self, server_provider_seed,
#                  client_provider_seed,
#                  client_provider_port,
#                  provider_client_port,
#                  server_provider_port,
#                  provider_server_port):
#
#         self.client_provider_prf = torch.Generator().manual_seed(client_provider_seed)
#         self.numpy_client_provider_prf = np.random.default_rng(seed=client_provider_seed)
#
#         self.server_provider_prf = torch.Generator().manual_seed(server_provider_seed)
#         self.numpy_server_provider_prf = np.random.default_rng(seed=server_provider_seed)
#
#         self.private_prf_numpy = np.random.default_rng(seed=1234)
#         self.private_prf = torch.Generator().manual_seed(1234)
#
#         self.client_numpy_queue = Receiver(client_provider_port)
#         self.client_numpy_queue.start()
#
#         self.server_numpy_queue = Receiver(server_provider_port)
#         self.server_numpy_queue.start()
#
#         self.provider2server_queue = Sender(provider_server_port)
#         self.provider2server_queue.start()
#
#         self.provider2client_queue = Sender(provider_client_port)
#         self.provider2client_queue.start()
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
#     def get_random_tensor(self, shape, prf):
#         return torch.randint(
#             low=torch.iinfo(worker.torch_dtype).min // 2,
#             high=torch.iinfo(worker.torch_dtype).max // 2 + 1,
#             size=shape,
#             dtype=worker.torch_dtype,
#             generator=prf
#         )
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
#
#     def conv2d(self, X_share_shape, Y_share_shape, stride=1):
#
#         b, i, m, _ = X_share_shape
#         m = m // stride
#         o, _, _, f = Y_share_shape
#         output_shape = (b, o, m, m)
#         padding = (f - 1) // 2
#
#         # TODO: remove //2
#         A_share_server = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=X_share_shape, dtype=self.torch_dtype, generator=self.server_provider_prf)
#         B_share_server = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=Y_share_shape, dtype=self.torch_dtype, generator=self.server_provider_prf)
#         C_share_server = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=output_shape,  dtype=self.torch_dtype, generator=self.server_provider_prf)
#         A_share_client = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=X_share_shape, dtype=self.torch_dtype, generator=self.client_provider_prf)
#         B_share_client = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2, high=torch.iinfo(self.torch_dtype).max // 2 + 1, size=Y_share_shape, dtype=self.torch_dtype, generator=self.client_provider_prf)
#
#         A = A_share_client + A_share_server
#         B = B_share_client + B_share_server
#
#         C_share_client = torch.conv2d(A, B, bias=None, stride=stride, padding=padding, dilation=1, groups=1) - C_share_server
#         self.provider2client_queue.put(C_share_client)
#
#     def mult(self, shape):
#
#         A_share_server = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
#         B_share_server = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
#         C_share_server = self.numpy_server_provider_prf.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
#
#         A_share_client = self.numpy_client_provider_prf.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
#         B_share_client = self.numpy_client_provider_prf.integers(self.min_val, self.max_val + 1, size=shape, dtype=self.dtype)
#
#
#         A = A_share_client + A_share_server
#         B = B_share_client + B_share_server
#
#         C_share_client = A * B - C_share_server
#         self.provider2client_queue.put(C_share_client)
#
#     def share_convert(self, size):
#
#         a_tild_0 = self.client_numpy_queue.get()
#         a_tild_1 = self.server_numpy_queue.get()
#
#         x = (a_tild_0 + a_tild_1)
#         delta = (x < a_tild_0).astype(self.dtype)
#
#         delta_0 = self.private_prf_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
#         delta_1 = self.sub_mode_L_minus_one(delta, delta_0)
#
#         self.provider2client_queue.put(delta_0)
#         self.provider2server_queue.put(delta_1)
#
#         r = self.server_numpy_queue.get()
#         eta_pp = self.server_numpy_queue.get()
#         eta_p = eta_pp ^ (x > (r - 1))
#
#         eta_p_0 = self.private_prf_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
#         eta_p_1 = self.sub_mode_L_minus_one(eta_p, eta_p_0)
#
#         self.provider2client_queue.put(eta_p_0)
#         self.provider2server_queue.put(eta_p_1)
#
#         return
#
#     def msb(self, size):
#         x = self.private_prf_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
#         x_0 = self.private_prf_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
#         x_1 = self.sub_mode_L_minus_one(x, x_0)
#
#         x_bit0 = x % 2
#         x_bit_0_0 = self.private_prf_numpy.integers(self.min_val, self.max_val + 1, size=size, dtype=self.dtype)
#         x_bit_0_1 = x_bit0 - x_bit_0_0
#
#         self.provider2client_queue.put(x_0)
#         self.provider2client_queue.put(x_bit_0_0)
#
#         self.provider2server_queue.put(x_1)
#         self.provider2server_queue.put(x_bit_0_1)
#
#         r = self.server_numpy_queue.get()
#         beta = self.server_numpy_queue.get()
#
#         beta_p = beta ^ (x > r)
#         beta_p_0 = self.private_prf_numpy.integers(self.min_val, self.max_val + 1, size=size, dtype=self.dtype)
#         beta_p_1 = beta_p - beta_p_0
#
#         self.provider2client_queue.put(beta_p_0)
#         self.provider2server_queue.put(beta_p_1)
#
#         self.mult(size)
#         return
#
#     def drelu(self, size):
#         crypt_provider.share_convert(size)
#         crypt_provider.msb(size)
#
#     def relu(self, size):
#         self.drelu(size)
#         self.mult(size)
#
#
#
# class SecureConv2DCryptoProvider(torch.nn.Module):
#     def __init__(self, W_shape, stride, dilation, prf_02_torch, prf_12_torch, provider2client_queue):
#         super(SecureConv2DCryptoProvider, self).__init__()
#         self.W_shape = W_shape
#         self.stride = stride
#         self.dilation = dilation
#
#         self.prf_12_torch = prf_12_torch
#         self.prf_02_torch = prf_02_torch
#
#         self.provider2client_queue = provider2client_queue
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
#         o, _, _, f = self.W_shape
#         padding = (f - 1) // 2
#
#         A_share_server = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2,
#                                        high=torch.iinfo(self.torch_dtype).max // 2 + 1,
#                                        size=X_share.shape,
#                                        dtype=self.torch_dtype,
#                                        generator=self.prf_12_torch)
#
#         B_share_server = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2,
#                                        high=torch.iinfo(self.torch_dtype).max // 2 + 1,
#                                        size=self.W_shape,
#                                        dtype=self.torch_dtype,
#                                        generator=self.prf_12_torch)
#
#         A_share_client = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2,
#                                        high=torch.iinfo(self.torch_dtype).max // 2 + 1,
#                                        size=X_share.shape,
#                                        dtype=self.torch_dtype,
#                                        generator=self.prf_02_torch)
#
#         B_share_client = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2,
#                                        high=torch.iinfo(self.torch_dtype).max // 2 + 1,
#                                        size=self.W_shape,
#                                        dtype=self.torch_dtype,
#                                        generator=self.prf_02_torch)
#
#         A = A_share_client + A_share_server
#         B = B_share_client + B_share_server
#
#         C = torch.conv2d(A, B, bias=None, stride=self.stride, padding=padding, dilation=1, groups=1)
#
#         C_share_server = torch.randint(low=torch.iinfo(self.torch_dtype).min // 2,
#                                        high=torch.iinfo(self.torch_dtype).max // 2 + 1,
#                                        size=C.shape,
#                                        dtype=self.torch_dtype,
#                                        generator=self.prf_12_torch)
#
#         C_share_client = C - C_share_server
#
#         self.provider2client_queue.put(C_share_client)
#
#         return C_share_client

if __name__ == "__main__":


    from research.distortion.utils import get_model
    from research.pipeline.backbones.secure_resnet import AvgPoolResNet
    image_shape = (1, 3, 64, 64)

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

    model = get_model(
        config="/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16.py",
        gpu_id=None,
        checkpoint_path=None
    )


    model.backbone.stem[0] = SecureConv2DCryptoProvider(
        W_shape=model.backbone.stem[0].weight.shape,
        stride=model.backbone.stem[0].stride,
        dilation=model.backbone.stem[0].dilation,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )
    model.backbone.stem[1] = torch.nn.Identity()
    model.backbone.stem[2] = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[3] = SecureConv2DCryptoProvider(
        W_shape=model.backbone.stem[3].weight.shape,
        stride=model.backbone.stem[3].stride,
        dilation=model.backbone.stem[3].dilation,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )
    model.backbone.stem[4] = torch.nn.Identity()
    model.backbone.stem[5] = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[6] = SecureConv2DCryptoProvider(
        W_shape=model.backbone.stem[6].weight.shape,
        stride=model.backbone.stem[6].stride,
        dilation=model.backbone.stem[6].dilation,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )
    model.backbone.stem[7] = torch.nn.Identity()
    model.backbone.stem[8] = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

    dummy_I = crypto_assets.get_random_tensor_over_L(
        shape=image_shape,
        prf=crypto_assets.private_prf
    )
    model.backbone.stem(dummy_I)


    assert False








    #
    #
    # from research.distortion.utils import get_model
    # from research.pipeline.backbones.secure_resnet import AvgPoolResNet
    # image_shape = (1, 3, 64, 64)
    #
    # model = get_model(
    #     config="/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16.py",
    #     gpu_id=None,
    #     checkpoint_path=None
    # )
    #
    # worker = CryptoProvider(
    #     server_provider_seed=3,
    #     client_provider_seed=2,
    #     client_provider_port=45125,
    #     provider_client_port=22126,
    #     server_provider_port=23129,
    #     provider_server_port=24328
    # )
    #
    # model.backbone.stem[0] = SecureConv2DCryptoProvider(
    #     W_shape=model.backbone.stem[0].weight.shape,
    #     stride=model.backbone.stem[0].stride,
    #     dilation=model.backbone.stem[0].dilation,
    #     prf_02_torch=worker.client_provider_prf,
    #     prf_12_torch=worker.server_provider_prf,
    #     provider2client_queue=worker.provider2client_queue
    # )
    # model.backbone.stem[1] = torch.nn.Identity()
    # model.backbone.stem[2] = torch.nn.Identity()
    #
    # model.backbone.stem[3] = SecureConv2DCryptoProvider(
    #     W_shape=model.backbone.stem[3].weight.shape,
    #     stride=model.backbone.stem[3].stride,
    #     dilation=model.backbone.stem[3].dilation,
    #     prf_02_torch=worker.client_provider_prf,
    #     prf_12_torch=worker.server_provider_prf,
    #     provider2client_queue=worker.provider2client_queue
    # )
    # model.backbone.stem[4] = torch.nn.Identity()
    # model.backbone.stem[5] = torch.nn.Identity()
    #
    # model.backbone.stem[6] = SecureConv2DCryptoProvider(
    #     W_shape=model.backbone.stem[6].weight.shape,
    #     stride=model.backbone.stem[6].stride,
    #     dilation=model.backbone.stem[6].dilation,
    #     prf_02_torch=worker.client_provider_prf,
    #     prf_12_torch=worker.server_provider_prf,
    #     provider2client_queue=worker.provider2client_queue
    # )
    # model.backbone.stem[7] = torch.nn.Identity()
    # model.backbone.stem[8] = torch.nn.Identity()
    #
    # dummy_I = worker.get_random_tensor(
    #     shape=image_shape,
    #     prf=worker.private_prf
    # )
    # model.backbone.stem(dummy_I)
    #
    # assert False
    #
    #
    # # share_convert_check = False
    # # conv_2d_check = False
    # # mult_check = False
    # # msb_check = False
    # # msb_share_check = False
    # # relu_check = False
    #
    # # if share_convert_check:
    # #     crypt_provider.share_convert(size=1000)
    # #
    # # if conv_2d_check:
    # #     crypt_provider.conv2d((1, 3, 256, 256), (32, 3, 3, 3), stride=2)
    # #     crypt_provider.conv2d((1, 32, 128, 128), (32, 32, 3, 3))
    # #     crypt_provider.conv2d((1, 32, 128, 128), (64, 32, 3, 3))
    # #
    # # if mult_check:
    # #     crypt_provider.mult(1000)
    # #
    # # if msb_check:
    # #     crypt_provider.msb(1000)
    # #
    # # if msb_share_check:
    # #
    # #     crypt_provider.share_convert(1000)
    # #     crypt_provider.msb(1000)
    # #
    # # if relu_check:
    # #     crypt_provider.relu(1000)
    # time.sleep(5)
    # stem_check = True
    # if stem_check:
    #     crypt_provider.conv2d((1, 3, 64, 64), (64, 3, 7, 7), stride=2)
    #     crypt_provider.relu(size=(1, 64, 32, 32))
    #
    #     crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
    #     crypt_provider.relu(size=(1, 64, 32, 32))
    #
    #     crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
    #     crypt_provider.relu(size=(1, 64, 32, 32))
    #
    #     crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
    #     crypt_provider.relu(size=(1, 64, 32, 32))
    #
    #     crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
    #     crypt_provider.relu(size=(1, 64, 32, 32))
    #
    #     crypt_provider.server_numpy_queue.make_stop()
    #     crypt_provider.client_numpy_queue.make_stop()
    #     crypt_provider.provider2server_queue.make_stop()
    #     crypt_provider.provider2client_queue.make_stop()
    #     # crypt_provider.conv2d((1, 3, 64, 64), (64, 3, 7, 7), stride=2)
    #     # crypt_provider.relu(size=(1, 64, 32, 32))
    #     #
    #     # crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
    #     # crypt_provider.relu(size=(1, 64, 32, 32))
    #     #
    #     # crypt_provider.conv2d((1, 64, 32, 32), (64, 64, 3, 3), stride=1)
    #     # crypt_provider.relu(size=(1, 64, 32, 32))