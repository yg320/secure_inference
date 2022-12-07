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

    def forward(self, size):

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
        self.mult = SecureMultiplicationCryptoProvider(crypto_assets, network_assets)

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

        self.share_convert = ShareConvertCryptoProvider(crypto_assets, network_assets)
        self.msb = SecureMSBCryptoProvider(crypto_assets, network_assets)

    def forward(self, X_share):
        self.share_convert(X_share.shape)
        self.msb(X_share.shape)
        return X_share


class SecureReLUCryptoProvider(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUCryptoProvider, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUCryptoProvider(crypto_assets, network_assets)
        self.mult = SecureMultiplicationCryptoProvider(crypto_assets, network_assets)

    def forward(self, X_share):
        X_share = X_share.numpy().astype(np.uint64)
        X_share = self.DReLU(X_share)
        self.mult(X_share.shape)
        return X_share

if __name__ == "__main__":


    from research.distortion.utils import get_model
    from research.pipeline.backbones.secure_resnet import AvgPoolResNet
    image_shape = (1, 3, 64, 64)

    port_01 = 12354
    port_10 = 12355
    port_02 = 12356
    port_20 = 12357
    port_12 = 12358
    port_21 = 12359

    prf_01_seed = 0
    prf_02_seed = 1
    prf_12_seed = 2

    sender_01 = None
    sender_02 = Sender(port_20)
    sender_12 = Sender(port_21)
    receiver_01 = None
    receiver_02 = Receiver(port_02)
    receiver_12 = Receiver(port_12)

    prf_01_numpy = None
    prf_02_numpy = np.random.default_rng(seed=prf_02_seed)
    prf_12_numpy = np.random.default_rng(seed=prf_12_seed)
    prf_01_torch = None
    prf_02_torch = torch.Generator().manual_seed(prf_02_seed)
    prf_12_torch = torch.Generator().manual_seed(prf_12_seed)

    crypto_assets = CryptoAssets(
        prf_01_numpy=prf_01_numpy,
        prf_02_numpy=prf_02_numpy,
        prf_12_numpy=prf_12_numpy,
        prf_01_torch=prf_01_torch,
        prf_02_torch=prf_02_torch,
        prf_12_torch=prf_12_torch,
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
        prf=crypto_assets.private_prf_torch
    )

    import time
    time.sleep(5)
    print("Start")
    model.backbone.stem(dummy_I)


    assert False


