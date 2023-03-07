import argparse
import mmcv
from research.secure_inference_3pc.backend import backend

from research.secure_inference_3pc.base import fuse_conv_bn, get_assets, TypeConverter
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, DUMMY_RELU, PRF_PREFETCH
from research.secure_inference_3pc.model_securifier import get_secure_model, init_prf_fetcher
from research.secure_inference_3pc.parties.server.prf_modules import PRFFetcherConv2D, PRFFetcherReLU, \
    PRFFetcherMaxPool, \
    PRFFetcherSecureModelSegmentation, PRFFetcherSecureModelClassification, PRFFetcherBlockReLU

from research.mmlab_extension.segmentation.secure_aspphead import SecureASPPHead
from research.mmlab_extension.classification.resnet_cifar_v2 import ResNet_CIFAR_V2  # TODO: why is this needed?
from research.mmlab_extension.classification.resnet import MyResNet  # TODO: why is this needed?
from research.mmlab_extension.segmentation.resnet_seg import AvgPoolResNetSeg

from research.secure_inference_3pc.parties.server.secure_modules import SecureConv2DServer, SecureReLUServer, \
    SecureMaxPoolServer, \
    SecureBlockReLUServer


def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False, device="cpu"):
    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DServer

    if bn_module:
        W, B = fuse_conv_bn(conv_module=conv_module, batch_norm_module=bn_module)
        W = TypeConverter.f2i(W)
        B = TypeConverter.f2i(B)

    else:
        W = conv_module.weight
        # assert conv_module.bias is None
        W = TypeConverter.f2i(W)
        B = TypeConverter.f2i(conv_module.bias)

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


def build_secure_fully_connected(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False,
                                 device="cpu"):
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


def build_secure_relu(crypto_assets, network_assets, is_prf_fetcher=False, dummy_relu=False, **kwargs):
    relu_class = PRFFetcherReLU if is_prf_fetcher else SecureReLUServer
    return relu_class(crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu,
                      is_prf_fetcher=is_prf_fetcher, **kwargs)


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

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--secure_config_path', type=str, default=None)
    parser.add_argument('--relu_spec_file', type=str, default=None)
    args = parser.parse_args()


    cfg = mmcv.Config.fromfile(args.secure_config_path)

    crypto_assets, network_assets = get_assets(party, device=args.device)

    if PRF_PREFETCH:
        prf_fetcher = init_prf_fetcher(
            cfg=cfg,
            checkpoint_path=None,
            max_pool=PRFFetcherMaxPool,
            build_secure_conv=build_secure_conv,
            build_secure_relu=build_secure_relu,
            build_secure_fully_connected=build_secure_fully_connected,
            prf_fetcher_secure_model=PRFFetcherSecureModelSegmentation if cfg.model.type == "EncoderDecoder" else PRFFetcherSecureModelClassification,
            secure_block_relu=PRFFetcherBlockReLU,
            relu_spec_file=args.relu_spec_file,
            crypto_assets=crypto_assets,
            network_assets=network_assets,
            dummy_relu=DUMMY_RELU,
            device=args.device,
        )
    else:
        prf_fetcher = None

    model = get_secure_model(
        cfg,
        checkpoint_path=args.model_path,
        build_secure_conv=build_secure_conv,
        build_secure_relu=build_secure_relu,
        build_secure_fully_connected=build_secure_fully_connected,
        max_pool=SecureMaxPoolServer,
        secure_model_class=SecureModelSegmentation if cfg.model.type == "EncoderDecoder" else SecureModelClassification,
        block_relu=SecureBlockReLUServer,
        relu_spec_file=args.relu_spec_file,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        dummy_relu=DUMMY_RELU,
        prf_fetcher=prf_fetcher,
        device=args.device

    )

    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.fetch(model=model.prf_fetcher)

    while True:

        image_size = network_assets.receiver_01.get()

        if image_size.shape == (1,):
            break
        network_assets.sender_01.put(image_size)

        if model.prf_fetcher:
            model.prf_fetcher.prf_handler.fetch_image(image_shape=image_size)

        out = model(image_size)

    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.done()
    network_assets.done()

    # print("Num of bytes sent 1 -> 0", network_assets.sender_01.num_of_bytes_sent)
    # print("Num of bytes sent 1 -> 0", network_assets.sender_01.num_of_bytes_sent)
    print("Num of bytes sent 1 ", network_assets.sender_12.num_of_bytes_sent + network_assets.sender_01.num_of_bytes_sent)
