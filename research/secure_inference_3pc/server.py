import torch
import mmcv
from research.secure_inference_3pc.backend import backend


from research.secure_inference_3pc.base import fuse_conv_bn, decompose, get_c_party_1, module_67,  get_assets, TypeConverter, NetworkAssets, get_c_party_1_torch, decompose_torch_1
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.conv2d import conv_2d
from research.secure_inference_3pc.conv2d_torch import Conv2DHandler
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE
from research.secure_inference_3pc.resnet_converter import get_secure_model, init_prf_fetcher
from research.secure_inference_3pc.params import Params
from research.secure_inference_3pc.modules.server import PRFFetcherConv2D, PRFFetcherReLU, PRFFetcherMaxPool, PRFFetcherSecureModelSegmentation, PRFFetcherSecureModelClassification, PRFFetcherBlockReLU

from research.bReLU import SecureOptimizedBlockReLU

from research.mmlab_extension.resnet_cifar_v2 import ResNet_CIFAR_V2
from research.mmlab_extension.classification.resnet import AvgPoolResNet, MyResNet
from research.secure_inference_3pc.modules.server import SecureConv2DServer, SecureReLUServer, SecureMaxPoolServer, SecureBlockReLUServer



def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False, device="cpu"):
    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DServer

    if bn_module:
        W, B = fuse_conv_bn(conv_module=conv_module, batch_norm_module=bn_module)
        W = TypeConverter.f2i(W)
        B = TypeConverter.f2i(B)

    else:
        W = conv_module.weight
        assert conv_module.bias is None
        W = TypeConverter.f2i(W)
        B = None

    return conv_class(
        W=W,
        bias=B,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        groups=conv_module.groups,
        device=device,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        is_prf_fetcher=is_prf_fetcher
    )

def build_secure_fully_connected(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False, device="cpu"):
    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DServer

    assert bn_module is None
    stride = (1, 1)
    dilation = (1, 1)
    padding = (0, 0)
    groups = 1

    W = TypeConverter.f2i(conv_module.weight.unsqueeze(2).unsqueeze(3))
    B = TypeConverter.f2i(conv_module.bias)

    return conv_class(
        W=W,
        bias=B,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
        device=device,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def build_secure_relu(crypto_assets, network_assets, is_prf_fetcher=False, dummy_relu=False):
    relu_class = PRFFetcherReLU if is_prf_fetcher else SecureReLUServer
    return relu_class(crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu, is_prf_fetcher=is_prf_fetcher)



class SecureModelSegmentation(SecureModule):
    def __init__(self, model, **kwargs):
        super(SecureModelSegmentation, self).__init__(**kwargs)
        self.model = model

    def forward(self, image_shape):

        image = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL,
                                                          high=MAX_VAL,
                                                          size=image_shape,
                                                          dtype=SIGNED_DTYPE)
        out = self.model.decode_head(self.model.backbone(image))
        self.network_assets.sender_01.put(out)

class SecureModelClassification(SecureModule):
    def __init__(self, model, **kwargs):
        super(SecureModelClassification, self).__init__(**kwargs)
        self.model = model

    def forward(self, image_shape):

        image = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL,
                                                          high=MAX_VAL,
                                                          size=image_shape,
                                                          dtype=SIGNED_DTYPE)

        out = self.model.backbone(image)[0]
        out = self.model.neck(out)
        out = self.model.head.fc(out)
        self.network_assets.sender_01.put(out)


if __name__ == "__main__":
    party = 1
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
        checkpoint_path=Params.MODEL_PATH,
        build_secure_conv=build_secure_conv,
        build_secure_relu=build_secure_relu,
        build_secure_fully_connected=build_secure_fully_connected,
        max_pool=SecureMaxPoolServer,
        secure_model_class=SecureModelSegmentation if cfg.model.type == "EncoderDecoder" else SecureModelClassification,
        block_relu=SecureBlockReLUServer,
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
                                            image=backend.zeros(shape=Params.IMAGE_SHAPE, dtype=SIGNED_DTYPE))

    for _ in range(Params.NUM_IMAGES):

        out = model(Params.IMAGE_SHAPE)

    network_assets.done()

    print("Num of bytes sent 1 -> 0", network_assets.sender_01.num_of_bytes_sent)
    print("Num of bytes sent 1 -> 2", network_assets.sender_12.num_of_bytes_sent)