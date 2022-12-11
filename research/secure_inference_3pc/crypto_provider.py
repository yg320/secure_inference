import torch
from research.communication.utils import Sender, Receiver
import numpy as np
import time

from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets, fuse_conv_bn, pre_conv, post_conv, mat_mult_single, Addresses, P, sub_mode_p, decompose


from research.communication.utils import Sender, Receiver
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets


class SecureConv2DCryptoProvider(SecureModule):
    def __init__(self, W_shape, stride, dilation, padding, crypto_assets: CryptoAssets, network_assets: NetworkAssets):
        super(SecureConv2DCryptoProvider, self).__init__(crypto_assets, network_assets)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):

        A_share_1 = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_12_torch)
        B_share_1 = self.crypto_assets.get_random_tensor_over_L(shape=self.W_shape, prf=self.crypto_assets.prf_12_torch)
        A_share_0 = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_02_torch)
        B_share_0 = self.crypto_assets.get_random_tensor_over_L(shape=self.W_shape, prf=self.crypto_assets.prf_02_torch)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1

        A = A.numpy()
        B = B.numpy()

        A, B, batch_size, nb_channels_out, nb_rows_out, nb_cols_out = pre_conv(A, B, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

        A = A.copy()
        B = B.copy()

        out_numpy = mat_mult_single(A[0], B)
        out_numpy = out_numpy[np.newaxis]
        out_numpy = post_conv(None, out_numpy, batch_size, nb_channels_out, nb_rows_out, nb_cols_out)
        C = torch.from_numpy(out_numpy)

        # C = torch.conv2d(A, B, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

        C_share_1 = self.crypto_assets.get_random_tensor_over_L(shape=C.shape, prf=self.crypto_assets.prf_12_torch)
        C_share_0 = C - C_share_1

        self.network_assets.sender_02.put(C_share_0)

        return C_share_0

class SecureConv2DCryptoProvider_V2(SecureModule):
    def __init__(self, W_shape, stride, dilation, padding, crypto_assets: CryptoAssets, network_assets: NetworkAssets):
        super(SecureConv2DCryptoProvider, self).__init__(crypto_assets, network_assets)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):

        A_share_1 = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_12_torch)
        B_share_1 = self.crypto_assets.get_random_tensor_over_L(shape=self.W_shape, prf=self.crypto_assets.prf_12_torch)
        A_share_0 = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_02_torch)
        B_share_0 = self.crypto_assets.get_random_tensor_over_L(shape=self.W_shape, prf=self.crypto_assets.prf_02_torch)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1

        C = torch.conv2d(A, B, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

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
        X_share_np = X_share.numpy().astype(self.dtype)
        X_share_np = self.DReLU(X_share_np)
        self.mult(X_share_np.shape)
        return X_share

def build_secure_conv(crypto_assets, network_assets, conv_module):
    return SecureConv2DCryptoProvider(
        W_shape=conv_module.weight.shape,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )
if __name__ == "__main__":


    from research.distortion.utils import get_model
    from research.pipeline.backbones.secure_resnet import AvgPoolResNet
    image_shape = (1, 3, 192, 256)

    addresses = Addresses()
    port_01 = addresses.port_01
    port_10 = addresses.port_10
    port_02 = addresses.port_02
    port_20 = addresses.port_20
    port_12 = addresses.port_12
    port_21 = addresses.port_21



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

    model.backbone.stem[0] = build_secure_conv(crypto_assets=crypto_assets, network_assets=network_assets, conv_module=model.backbone.stem[0])
    model.backbone.stem[1] = torch.nn.Identity()
    model.backbone.stem[2] = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[3] = build_secure_conv(crypto_assets=crypto_assets, network_assets=network_assets, conv_module=model.backbone.stem[3])
    model.backbone.stem[4] = torch.nn.Identity()
    model.backbone.stem[5] = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[6] = build_secure_conv(crypto_assets=crypto_assets, network_assets=network_assets, conv_module=model.backbone.stem[6])
    model.backbone.stem[7] = torch.nn.Identity()
    model.backbone.stem[8] = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

    for layer in [1, 2, 3, 4]:
        for block in [0, 1]:
            cur_res_layer = getattr(model.backbone, f"layer{layer}")
            cur_res_layer[block].conv1 = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].conv1)
            cur_res_layer[block].bn1 = torch.nn.Identity()
            cur_res_layer[block].relu_1 = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

            cur_res_layer[block].conv2 = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].conv2)
            cur_res_layer[block].bn2 = torch.nn.Identity()
            cur_res_layer[block].relu_2 = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

            if cur_res_layer[block].downsample:
                cur_res_layer[block].downsample = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].downsample[0])

    model.decode_head.image_pool[1].conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.image_pool[1].conv)
    model.decode_head.image_pool[1].bn = torch.nn.Identity()
    model.decode_head.image_pool[1].activate = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

    for i in range(4):
        model.decode_head.aspp_modules[i].conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.aspp_modules[i].conv)
        model.decode_head.aspp_modules[i].bn = torch.nn.Identity()
        model.decode_head.aspp_modules[i].activate = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

    model.decode_head.bottleneck.conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.bottleneck.conv)
    model.decode_head.bottleneck.bn = torch.nn.Identity()
    model.decode_head.bottleneck.activate = SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)

    model.decode_head.conv_seg = build_secure_conv(crypto_assets, network_assets, model.decode_head.conv_seg)
    model.decode_head.image_pool[0].forward = lambda x: x.sum(dim=[2, 3], keepdims=True) // (x.shape[2] * x.shape[3])

    dummy_I = crypto_assets.get_random_tensor_over_L(
        shape=image_shape,
        prf=crypto_assets.private_prf_torch
    )

    import time
    time.sleep(5)
    print("Start")
    image = dummy_I
    out = model.backbone.stem(image)


    assert False


