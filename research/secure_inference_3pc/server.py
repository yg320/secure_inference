import torch
from research.communication.utils import Sender, Receiver
import time
import numpy as np
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets, fuse_conv_bn


from research.communication.utils import Sender, Receiver
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets


class SecureConv2DServer(SecureModule):
    def __init__(self, W, bias, stride, dilation, crypto_assets: CryptoAssets, network_assets: NetworkAssets):
        super(SecureConv2DServer, self).__init__(crypto_assets, network_assets)

        self.W_share = W
        self.bias = bias
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

        C_share = self.crypto_assets.get_random_tensor_over_L(shape=out.shape, prf=self.crypto_assets.prf_12_torch)

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
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)

    def forward(self, a_1):
        beta = self.crypto_assets.prf_01_numpy.integers(0, 2, size=a_1.shape, dtype=self.dtype)

        x_1 =       self.network_assets.receiver_12.get()
        x_bit_0_1 = self.network_assets.receiver_12.get()

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

class SecureReLUServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUServer, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUServer(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)

    def forward(self, X_share):
        X_share = X_share.numpy().astype(np.uint64)
        MSB_0 = self.DReLU(X_share)
        relu_0 = self.mult(X_share, MSB_0)
        return torch.from_numpy(relu_0.astype(np.int64))


if __name__ == "__main__":
    from research.distortion.utils import get_model
    from research.pipeline.backbones.secure_resnet import AvgPoolResNet

    image_shape = (1, 3, 64, 64)
    model = get_model(
        config="/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16.py",
        gpu_id=None,
        checkpoint_path="/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/iter_80000.pth"
    )
    desired_out = model.backbone.stem(torch.load("/home/yakir/tmp/data.pt") )
    #
    port_01 = 12354
    port_10 = 12355
    port_02 = 12356
    port_20 = 12357
    port_12 = 12358
    port_21 = 12359

    prf_01_seed = 0
    prf_02_seed = 1
    prf_12_seed = 2

    sender_01 = Sender(port_10)
    sender_02 = None
    sender_12 = Sender(port_12)
    receiver_01 = Receiver(port_01)
    receiver_02 = None
    receiver_12 = Receiver(port_21)

    prf_01_numpy = np.random.default_rng(seed=prf_01_seed)
    prf_02_numpy = None
    prf_12_numpy = np.random.default_rng(seed=prf_12_seed)
    prf_01_torch = torch.Generator().manual_seed(prf_01_seed)
    prf_02_torch = None
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
    import time
    time.sleep(5)
    print("Start")
    out_1 = model.backbone.stem(I1)
    out_0 = network_assets.receiver_01.get()

    out = (torch.from_numpy(out_0) + out_1)
    out = out.to(torch.float32) / 10000
    assert False
