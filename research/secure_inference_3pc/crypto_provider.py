import torch
import numpy as np

from research.secure_inference_3pc.base import P, sub_mode_p, decompose, get_assets
from research.secure_inference_3pc.conv2d import conv_2d
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.base import NetworkAssets
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE
from research.secure_inference_3pc.modules.conv2d import get_output_shape
from research.secure_inference_3pc.conv2d_torch import Conv2DHandler
from research.bReLU import NumpySecureOptimizedBlockReLU
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool

from research.distortion.utils import get_model
from research.pipeline.backbones.secure_resnet import AvgPoolResNet
from research.pipeline.backbones.secure_aspphead import SecureASPPHead
from research.secure_inference_3pc.resnet_converter import get_secure_model, init_prf_fetcher
from research.secure_inference_3pc.params import Params
from research.secure_inference_3pc.modules.crypto_provider import PRFFetcherConv2D, PRFFetcherReLU, PRFFetcherMaxPool, PRFFetcherSecureModelSegmentation, PRFFetcherSecureModelClassification, PRFFetcherBlockReLU

from functools import partial
import mmcv
from research.mmlab_extension.resnet_cifar_v2 import ResNet_CIFAR_V2
from research.mmlab_extension.classification.resnet import AvgPoolResNet, MyResNet

class SecureConv2DCryptoProvider(SecureModule):
    def __init__(self, W_shape, stride, dilation, padding, groups, device="cpu", **kwargs):
        super(SecureConv2DCryptoProvider, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

        self.conv2d_handler = Conv2DHandler("cuda:0")
        self.device = device

    def forward(self, X_share):
        # return np.zeros(get_output_shape(X_share, self.W_shape, self.padding, self.dilation, self.stride), dtype=X_share.dtype)

        assert X_share.dtype == SIGNED_DTYPE
        # TODO: intergers should be called without all of these arguments
        A_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        A_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1
        if self.device == "cpu":
            C = conv_2d(A, B, None, None, self.padding, self.stride, self.dilation, self.groups)
        else:
            C = self.conv2d_handler.conv2d(A, B, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
        C_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=C.shape, dtype=SIGNED_DTYPE)
        C_share_0 = C - C_share_1

        self.network_assets.sender_02.put(C_share_0)

        return C_share_0


class PrivateCompareCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareCryptoProvider, self).__init__(**kwargs)

    def forward(self):
        d_bits_0 = self.network_assets.receiver_02.get()
        d_bits_1 = self.network_assets.receiver_12.get()

        d = (d_bits_0 + d_bits_1) % P
        beta_p = (d == 0).any(axis=-1).astype(SIGNED_DTYPE)

        return beta_p


class ShareConvertCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertCryptoProvider, self).__init__(**kwargs)
        self.private_compare = PrivateCompareCryptoProvider(**kwargs)

    def forward(self, size):
        a_tild_0 = self.network_assets.receiver_02.get()
        a_tild_1 = self.network_assets.receiver_12.get()

        x = (a_tild_0 + a_tild_1)

        x_bits = decompose(x)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=x_bits.shape, dtype=np.int8)
        x_bits_1 = sub_mode_p(x_bits, x_bits_0)

        delta = (0 < a_tild_0 - x).astype(SIGNED_DTYPE)

        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        delta_0 = self.sub_mode_L_minus_one(delta, delta_1)

        self.network_assets.sender_02.put(delta_0)
        self.network_assets.sender_12.put(x_bits_1.astype(np.int8))

        # r = self.network_assets.receiver_12.get()
        # eta_p = self.network_assets.receiver_12.get()
        # eta_p = eta_p ^ (x > r)

        eta_p = self.private_compare()

        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        eta_p_1 = self.sub_mode_L_minus_one(eta_p, eta_p_0)

        self.network_assets.sender_12.put(eta_p_1)

        return


class SecureMultiplicationCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationCryptoProvider, self).__init__(**kwargs)

    def forward(self, shape):
        A_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        B_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        C_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        A_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        B_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1

        C_share_0 = A * B - C_share_1

        self.network_assets.sender_02.put(C_share_0)


class SecureMSBCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBCryptoProvider, self).__init__(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)
        self.private_compare = PrivateCompareCryptoProvider(**kwargs)

    def forward(self, size):
        x = self.prf_handler[CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)

        x_bits = decompose(x)

        # x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=x_bits.shape, dtype=np.int32)
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=x_bits.shape, dtype=np.int8)
        x_bits_1 = sub_mode_p(x_bits, x_bits_0)

        x_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        x_0 = self.sub_mode_L_minus_one(x, x_1)

        x_bit0 = x % 2
        x_bit_0_0 = self.prf_handler[CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=size, dtype=SIGNED_DTYPE)
        x_bit_0_1 = x_bit0 - x_bit_0_0

        self.network_assets.sender_02.put(x_0)
        self.network_assets.sender_02.put(x_bit_0_0)

        self.network_assets.sender_12.put(x_bits_1.astype(np.int8))
        self.network_assets.sender_12.put(x_bit_0_1)

        # r = self.network_assets.receiver_12.get()
        # beta = self.network_assets.receiver_12.get()
        # beta_p = beta ^ (x > r)
        beta_p = self.private_compare()

        beta_p_0 = self.prf_handler[CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=size, dtype=SIGNED_DTYPE)
        beta_p_1 = beta_p - beta_p_0

        self.network_assets.sender_02.put(beta_p_0)
        self.network_assets.sender_12.put(beta_p_1)

        self.mult(size)
        return


class SecureDReLUCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureDReLUCryptoProvider, self).__init__(**kwargs)

        self.share_convert = ShareConvertCryptoProvider(**kwargs)
        self.msb = SecureMSBCryptoProvider(**kwargs)

    def forward(self, X_share):

        self.share_convert(X_share.shape)
        self.msb(X_share.shape)
        return X_share


class SecureReLUCryptoProvider(SecureModule):
    def __init__(self, dummy_relu=False, **kwargs):
        super(SecureReLUCryptoProvider, self).__init__(**kwargs)

        self.DReLU = SecureDReLUCryptoProvider(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        if self.dummy_relu:
            return X_share
        else:
            X_share_np = X_share.flatten()
            X_share_np = self.DReLU(X_share_np)
            self.mult(X_share_np.shape)
            return X_share

class SecureBlockReLUCryptoProvider(SecureModule, NumpySecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        NumpySecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUCryptoProvider(**kwargs)
        self.secure_mult = SecureMultiplicationCryptoProvider(**kwargs)

    def mult(self, x, y):
        self.secure_mult(x.shape)
        return x



class SecureSelectShareCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareCryptoProvider, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationCryptoProvider(**kwargs)

    def forward(self, share, dummy0=None, dummy1=None):

        self.secure_multiplication(share.shape)
        return share

class SecureMaxPoolCryptoProvider(SecureMaxPool):
    def __init__(self, kernel_size, stride, padding, dummy_max_pool, **kwargs):
        super(SecureMaxPoolCryptoProvider, self).__init__(kernel_size, stride, padding, dummy_max_pool, **kwargs)
        self.select_share = SecureSelectShareCryptoProvider(**kwargs)
        self.dReLU = SecureDReLUCryptoProvider(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)

    def forward(self, x):
        if self.dummy_max_pool:
            return x[:, :, ::2, ::2]
        return super(SecureMaxPoolCryptoProvider, self).forward(x)


def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False, device="cpu"):
    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DCryptoProvider

    return conv_class(
        W_shape=conv_module.weight.shape,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        groups=conv_module.groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        device=device
    )

def build_secure_fully_connected(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False):
    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DCryptoProvider

    return conv_class(
        W_shape=tuple(conv_module.weight.shape) + (1, 1),
        stride=(1, 1),
        dilation=(1, 1),
        padding=(0, 0),
        groups=1,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def build_secure_relu(is_prf_fetcher=False, dummy_relu=False, **kwargs):
    relu_class = PRFFetcherReLU if is_prf_fetcher else SecureReLUCryptoProvider
    return relu_class(dummy_relu=dummy_relu, **kwargs)



class SecureModelSegmentation(SecureModule):
    def __init__(self, model,  **kwargs):
        super(SecureModelSegmentation, self).__init__( **kwargs)
        self.model = model

    def forward(self, image_shape):

        dummy_image = self.prf_handler[CRYPTO_PROVIDER].integers(low=MIN_VAL,
                                                                 high=MAX_VAL,
                                                                 size=image_shape,
                                                                 dtype=SIGNED_DTYPE)
        _ = self.model.decode_head(self.model.backbone(dummy_image))


class SecureModelClassification(SecureModule):
    def __init__(self, model,  **kwargs):
        super(SecureModelClassification, self).__init__( **kwargs)
        self.model = model

    def forward(self, image_shape):

        dummy_image = self.prf_handler[CRYPTO_PROVIDER].integers(low=MIN_VAL,
                                                                 high=MAX_VAL,
                                                                 size=image_shape,
                                                                 dtype=SIGNED_DTYPE)
        out = self.model.backbone(dummy_image)[0]
        out = self.model.neck(out)
        out = self.model.head.fc(out)

if __name__ == "__main__":
    party = 2
    cfg = mmcv.Config.fromfile(Params.SECURE_CONFIG_PATH)

    crypto_assets, network_assets = get_assets(party, repeat=Params.NUM_IMAGES, simulated_bandwidth=Params.SIMULATED_BANDWIDTH)

    if Params.PRF_PREFETCH:
        prf_fetcher = init_prf_fetcher(
            cfg=cfg,
            Params=Params,
            max_pool=PRFFetcherMaxPool,
            build_secure_conv=build_secure_conv,
            build_secure_relu=build_secure_relu,
            build_secure_fully_connected=build_secure_fully_connected,
            prf_fetcher_secure_model=PRFFetcherSecureModelSegmentation if cfg.model.type == "EncoderDecoder" else PRFFetcherSecureModelClassification,
            secure_block_relu=PRFFetcherBlockReLU,
            relu_spec_file=Params.RELU_SPEC_FILE,
            crypto_assets=crypto_assets,
            network_assets=network_assets,
            dummy_relu=Params.DUMMY_RELU,
            dummy_max_pool=Params.DUMMY_MAX_POOL)
    else:
        prf_fetcher = None

    model = get_secure_model(
        cfg,
        checkpoint_path=None,
        build_secure_conv=build_secure_conv,
        build_secure_relu=build_secure_relu,
        build_secure_fully_connected=build_secure_fully_connected,
        max_pool=SecureMaxPoolCryptoProvider,
        secure_model_class=SecureModelSegmentation if cfg.model.type == "EncoderDecoder" else SecureModelClassification,
        block_relu=SecureBlockReLUCryptoProvider,
        relu_spec_file=Params.RELU_SPEC_FILE,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        dummy_relu=Params.DUMMY_RELU,
        dummy_max_pool=Params.DUMMY_MAX_POOL,
        prf_fetcher=prf_fetcher,
        device=Params.DEVICE

    )
    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.fetch(repeat=Params.NUM_IMAGES, model=model.prf_fetcher,
                                            image=np.zeros(shape=Params.IMAGE_SHAPE, dtype=SIGNED_DTYPE))

    for _ in range(Params.NUM_IMAGES):
        out = model(Params.IMAGE_SHAPE)

    network_assets.done()

    print("Num of bytes sent 2 -> 0", network_assets.sender_02.num_of_bytes_sent)
    print("Num of bytes sent 2 -> 1", network_assets.sender_12.num_of_bytes_sent)