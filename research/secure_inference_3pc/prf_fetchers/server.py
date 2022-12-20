import torch
import numpy as np
from research.secure_inference_3pc.base import fuse_conv_bn, decompose, get_c, module_67, DepthToSpace, SpaceToDepth, get_assets, TypeConverter
from research.secure_inference_3pc.conv2d import conv_2d, compile_numba_funcs, get_output_shape

from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets

from research.distortion.utils import get_model
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER
from research.pipeline.backbones.secure_resnet import AvgPoolResNet
from research.pipeline.backbones.secure_aspphead import SecureASPPHead
from research.secure_inference_3pc.resnet_converter import securify_mobilenetv2_model
from functools import partial
from research.secure_inference_3pc.timer import Timer
class SecureConv2DServer(SecureModule):
    def __init__(self, W, bias, stride, dilation, padding, groups, crypto_assets: CryptoAssets, network_assets: NetworkAssets):
        super(SecureConv2DServer, self).__init__(crypto_assets, network_assets)

        self.W_share = W.numpy()
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):
        X_share = X_share.numpy()
        out_shape = get_output_shape(X_share, self.W_share, self.padding, self.dilation, self.stride)

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=X_share.shape, dtype=np.int64)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=self.W_share.shape, dtype=np.int64)
        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=out_shape, dtype=np.int64)

        return torch.from_numpy(np.zeros(shape=out_shape, dtype=X_share.dtype))


class PrivateCompareServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PrivateCompareServer, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_1, r, beta):

        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_1.shape, dtype=np.int32)
        print("PrivateCompareServer")

class ShareConvertServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertServer, self).__init__(crypto_assets, network_assets)
        self.private_compare = PrivateCompareServer(crypto_assets, network_assets)

    def forward(self, a_1):

        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=np.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(self.min_val, self.max_val, size=a_1.shape, dtype=self.dtype)
        print("ShareConvertServer")

        return a_1


class SecureMultiplicationServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationServer, self).__init__(crypto_assets, network_assets)

    def forward(self, X_share, Y_share):
        assert X_share.dtype == self.dtype
        assert Y_share.dtype == self.dtype

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        print("SecureMultiplicationServer")

        return X_share


class SecureMSBServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBServer, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)
        self.private_compare = PrivateCompareServer(crypto_assets, network_assets)

    def forward(self, a_1):
        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=np.int8)
        x_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(self.min_val, self.max_val, size=a_1.shape, dtype=self.dtype)
        print("SecureMSBServer")

        return a_1


class SecureDReLUServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUServer, self).__init__(crypto_assets, network_assets)

        self.share_convert = ShareConvertServer(crypto_assets, network_assets)
        self.msb = SecureMSBServer(crypto_assets, network_assets)

    def forward(self, X_share):
        X1_converted = self.share_convert(self.dtype(2) * X_share)
        MSB_1 = self.msb(X1_converted)
        print("SecureDReLUServer")

        return MSB_1


class SecureReLUServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUServer, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUServer(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)

    def forward(self, X_share):

        aaa = X_share.numpy()
        aaa = aaa.astype(self.dtype).flatten()
        MSB_0 = self.DReLU(aaa)
        relu_0 = self.mult(aaa, MSB_0)
        print("SecureReLUServer")

        return X_share


class SecureBlockReLUServer(SecureModule):

    def __init__(self, crypto_assets, network_assets, block_sizes):
        super(SecureBlockReLUServer, self).__init__(crypto_assets, network_assets)
        self.block_sizes = np.array(block_sizes)
        self.DReLU = SecureDReLUServer(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if 0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def forward(self, activation):
        activation = activation.numpy()
        assert activation.dtype == self.signed_type

        reshaped_inputs = []
        mean_tensors = []
        channels = []
        orig_shapes = []

        for block_size in self.active_block_sizes:

            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = activation[:, cur_channels]

            reshaped_input = SpaceToDepth(block_size)(cur_input)
            assert reshaped_input.dtype == self.signed_type
            mean_tensor = np.sum(reshaped_input, axis=-1, keepdims=True)

            channels.append(cur_channels)
            reshaped_inputs.append(reshaped_input)
            orig_shapes.append(mean_tensor.shape)
            mean_tensors.append(mean_tensor.flatten())

        cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
        mean_tensors = np.concatenate(mean_tensors)
        assert mean_tensors.dtype == self.signed_type

        activation = activation.astype(self.dtype)
        sign_tensors = self.DReLU(mean_tensors.astype(self.dtype))

        relu_map = np.ones_like(activation)
        for i in range(len(self.active_block_sizes)):
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i+1])].reshape(orig_shapes[i])
            relu_map[:, channels[i]] = DepthToSpace(self.active_block_sizes[i])(sign_tensor.repeat(reshaped_inputs[i].shape[-1], axis=-1))

        activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels], activation[:, ~self.is_identity_channels])
        activation = activation.astype(self.signed_type)

        return torch.from_numpy(activation)


def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module):
    if bn_module:
        W, B = fuse_conv_bn(conv_module=conv_module, batch_norm_module=bn_module)
        W = TypeConverter.f2i(W)
        B = TypeConverter.f2i(B)
        W = W - crypto_assets[CLIENT, SERVER].get_random_tensor_over_L(shape=W.shape)

    else:
        W = conv_module.weight
        W = TypeConverter.f2i(W)
        W = W - crypto_assets[CLIENT, SERVER].get_random_tensor_over_L(shape=W.shape)
        B = None

    return SecureConv2DServer(
        W=W,
        bias=B,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        groups=conv_module.groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def build_secure_relu(crypto_assets, network_assets):
    return SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)


def run_inference(model, image_shape, crypto_assets, network_assets):
    image = crypto_assets[CLIENT, SERVER].get_random_tensor_over_L(shape=image_shape)
    out = model.decode_head(model.backbone(image))
    network_assets.sender_01.put(out)



if __name__ == "__main__":
    config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py"
    secure_config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline_secure.py"
    model_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth"
    relu_spec_file = None #"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/test/block_size_spec_0.15.pickle"
    image_shape = (1, 3, 256, 256)
    num_images = 100

    model = get_model(
        config=secure_config_path,
        gpu_id=None,
        checkpoint_path=model_path
    )

    # compile_numba_funcs()

    crypto_assets, network_assets = get_assets(1, repeat=num_images)

    build_secure_conv = partial(build_secure_conv, crypto_assets=crypto_assets, network_assets=network_assets)
    build_secure_relu = partial(build_secure_relu, crypto_assets=crypto_assets, network_assets=network_assets)

    securify_mobilenetv2_model(model,
                               build_secure_conv,
                               build_secure_relu,
                               block_relu=partial(SecureBlockReLUServer, crypto_assets=crypto_assets, network_assets=network_assets),
                               relu_spec_file=relu_spec_file)

    for _ in range(num_images):
        with Timer("PRF"):
            out = run_inference(model, image_shape, crypto_assets, network_assets)


    crypto_assets.done()
    network_assets.done()
