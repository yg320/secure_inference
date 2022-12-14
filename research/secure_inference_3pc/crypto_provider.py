import torch
import numpy as np

from research.secure_inference_3pc.base import P, sub_mode_p, decompose, SpaceToDepth, get_assets
from research.secure_inference_3pc.conv2d import conv_2d, compile_numba_funcs
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets

from research.distortion.utils import get_model
from research.pipeline.backbones.secure_resnet import AvgPoolResNet
from research.pipeline.backbones.secure_aspphead import SecureASPPHead
from research.secure_inference_3pc.resnet_converter import securify_model

class SecureConv2DCryptoProvider(SecureModule):
    def __init__(self, W_shape, stride, dilation, padding, crypto_assets: CryptoAssets, network_assets: NetworkAssets):
        super(SecureConv2DCryptoProvider, self).__init__(crypto_assets, network_assets)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):
        X_share = X_share.numpy()
        assert X_share.dtype == self.signed_type

        A_share_0 = self.crypto_assets.prf_02_numpy.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=X_share.shape, dtype=np.int64)
        B_share_0 = self.crypto_assets.prf_02_numpy.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=self.W_shape, dtype=np.int64)
        A_share_1 = self.crypto_assets.prf_12_numpy.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=X_share.shape, dtype=np.int64)
        B_share_1 = self.crypto_assets.prf_12_numpy.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=self.W_shape, dtype=np.int64)


        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1
        C = conv_2d(A, B, None, None, self.padding, self.stride, self.dilation)

        # C = torch.conv2d(A, B, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
        C_share_1 = self.crypto_assets.prf_12_numpy.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=C.shape, dtype=np.int64)
        C_share_0 = C - C_share_1

        self.network_assets.sender_02.put(C_share_0)

        return torch.from_numpy(C_share_0)


class PrivateCompareCryptoProvider(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PrivateCompareCryptoProvider, self).__init__(crypto_assets, network_assets)

    def forward(self):
        d_bits_0 = self.network_assets.receiver_02.get()
        d_bits_1 = self.network_assets.receiver_12.get()

        d = (d_bits_0 + d_bits_1) % P
        beta_p = (d == 0).any(axis=-1).astype(self.crypto_assets.numpy_dtype)

        return beta_p


class ShareConvertCryptoProvider(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertCryptoProvider, self).__init__(crypto_assets, network_assets)
        self.private_compare = PrivateCompareCryptoProvider(crypto_assets, network_assets)

    def forward(self, size):
        a_tild_0 = self.network_assets.receiver_02.get()
        a_tild_1 = self.network_assets.receiver_12.get()

        x = (a_tild_0 + a_tild_1)

        x_bits = decompose(x)

        x_bits_0 = self.crypto_assets.prf_02_numpy.integers(0, P, size=x_bits.shape, dtype=np.int8)
        x_bits_1 = sub_mode_p(x_bits, x_bits_0)

        delta = (x < a_tild_0).astype(self.dtype)

        delta_1 = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        delta_0 = self.sub_mode_L_minus_one(delta, delta_1)

        self.network_assets.sender_02.put(delta_0)
        self.network_assets.sender_12.put(x_bits_1.astype(np.int8))

        # r = self.network_assets.receiver_12.get()
        # eta_p = self.network_assets.receiver_12.get()
        # eta_p = eta_p ^ (x > r)

        eta_p = self.private_compare()

        eta_p_0 = self.crypto_assets.prf_02_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        eta_p_1 = self.sub_mode_L_minus_one(eta_p, eta_p_0)

        self.network_assets.sender_12.put(eta_p_1)

        return


class SecureMultiplicationCryptoProvider(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationCryptoProvider, self).__init__(crypto_assets, network_assets)

    def forward(self, shape):
        A_share_1 = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=shape,
                                                             dtype=self.dtype)
        B_share_1 = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=shape,
                                                             dtype=self.dtype)
        C_share_1 = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=shape,
                                                             dtype=self.dtype)
        A_share_0 = self.crypto_assets.prf_02_numpy.integers(self.min_val, self.max_val + 1, size=shape,
                                                             dtype=self.dtype)
        B_share_0 = self.crypto_assets.prf_02_numpy.integers(self.min_val, self.max_val + 1, size=shape,
                                                             dtype=self.dtype)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1

        C_share_0 = A * B - C_share_1

        self.network_assets.sender_02.put(C_share_0)


class SecureMSBCryptoProvider(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBCryptoProvider, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMultiplicationCryptoProvider(crypto_assets, network_assets)
        self.private_compare = PrivateCompareCryptoProvider(crypto_assets, network_assets)

    def forward(self, size):
        x = self.crypto_assets.private_prf_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)

        x_bits = decompose(x)

        # x_bits_0 = self.crypto_assets.prf_02_numpy.integers(0, P, size=x_bits.shape, dtype=np.int32)
        x_bits_0 = self.crypto_assets.prf_02_numpy.integers(0, P, size=x_bits.shape, dtype=np.int8)
        x_bits_1 = sub_mode_p(x_bits, x_bits_0)

        x_1 = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val, size=size, dtype=self.dtype)
        x_0 = self.sub_mode_L_minus_one(x, x_1)

        x_bit0 = x % 2
        x_bit_0_0 = self.crypto_assets.private_prf_numpy.integers(self.min_val, self.max_val + 1, size=size,
                                                                  dtype=self.dtype)
        x_bit_0_1 = x_bit0 - x_bit_0_0

        self.network_assets.sender_02.put(x_0)
        self.network_assets.sender_02.put(x_bit_0_0)

        self.network_assets.sender_12.put(x_bits_1.astype(np.int8))
        self.network_assets.sender_12.put(x_bit_0_1)

        # r = self.network_assets.receiver_12.get()
        # beta = self.network_assets.receiver_12.get()
        # beta_p = beta ^ (x > r)
        beta_p = self.private_compare()

        beta_p_0 = self.crypto_assets.private_prf_numpy.integers(self.min_val, self.max_val + 1, size=size,
                                                                 dtype=self.dtype)
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
        assert X_share.dtype == self.dtype
        self.share_convert(X_share.shape)
        self.msb(X_share.shape)
        return X_share


class SecureReLUCryptoProvider(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUCryptoProvider, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUCryptoProvider(crypto_assets, network_assets)
        self.mult = SecureMultiplicationCryptoProvider(crypto_assets, network_assets)

    def forward(self, X_share):
        shape = X_share.shape
        X_share = X_share.numpy()
        X_share_np = X_share.astype(self.dtype).flatten()
        X_share_np = self.DReLU(X_share_np)
        self.mult(X_share_np.shape)
        return torch.from_numpy(X_share)


class SecureBlockReLUCryptoProvider(SecureModule):

    def __init__(self, crypto_assets, network_assets, block_sizes):
        super(SecureBlockReLUCryptoProvider, self).__init__(crypto_assets, network_assets)
        self.block_sizes = np.array(block_sizes)
        self.DReLU = SecureDReLUCryptoProvider(crypto_assets, network_assets)
        self.mult = SecureMultiplicationCryptoProvider(crypto_assets, network_assets)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if 0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def forward(self, activation):

        activation = activation.numpy()
        assert activation.dtype == self.signed_type

        mean_tensors = []

        for block_size in self.active_block_sizes:

            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = activation[:, cur_channels]

            reshaped_input = SpaceToDepth(block_size)(cur_input)
            assert reshaped_input.dtype == self.signed_type

            mean_tensor = np.sum(reshaped_input, axis=-1, keepdims=True)

            mean_tensors.append(mean_tensor.flatten())

        mean_tensors = np.concatenate(mean_tensors)
        assert mean_tensors.dtype == self.signed_type
        self.DReLU(mean_tensors.astype(self.dtype))
        self.mult(activation[:, ~self.is_identity_channels].shape)
        activation = activation.astype(self.signed_type)
        return torch.from_numpy(activation)


def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module):
    return SecureConv2DCryptoProvider(
        W_shape=conv_module.weight.shape,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def build_secure_relu(crypto_assets, network_assets):
    return SecureReLUCryptoProvider(crypto_assets=crypto_assets, network_assets=network_assets)


def run_inference(model, image_shape, crypto_assets, network_assets):
    dummy_I = crypto_assets.get_random_tensor_over_L(
        shape=image_shape,
        prf=crypto_assets.private_prf_torch
    )

    print("Start")
    image = dummy_I
    _ = model.decode_head(model.backbone(image))
    network_assets.sender_02.put(None)
    network_assets.sender_12.put(None)


if __name__ == "__main__":

    compile_numba_funcs()

    config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16_secure.py"
    secure_config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16_secure.py"
    image_path = "/home/yakir/tmp/image_0.pt"
    model_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/knapsack_0.15_192x192_2x16_finetune_80k_v2/iter_80000.pth"
    relu_spec_file = "/home/yakir/Data2/assets_v4/distortions/ade_20k_192x192/ResNet18/block_size_spec_0.15.pickle"
    image_shape = (1, 3, 192, 192)

    model = get_model(
        config=config_path,
        gpu_id=None,
        checkpoint_path=None
    )

    compile_numba_funcs()
    crypto_assets, network_assets = get_assets(2)
    securify_model(model, build_secure_conv, build_secure_relu, crypto_assets, network_assets, block_relu=SecureBlockReLUCryptoProvider, relu_spec_file=relu_spec_file)
    run_inference(model, image_shape, crypto_assets, network_assets)
