import torch
from research.secure_inference_3pc.const import SIGNED_DTYPE
import pickle
from research.distortion.arch_utils.factory import arch_utils_factory
from functools import partial
from research.distortion.utils import get_model
from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.modules.base import DummyShapeTensor
from research.secure_inference_3pc.timer import timer
import numpy as np
import torch.nn as nn


def convert_conv_module(module, build_secure_conv, build_secure_relu):
    module.conv = build_secure_conv(conv_module=module.conv, bn_module=module.bn)
    module.bn = torch.nn.Identity()
    if hasattr(module, "activate"):
        module.activate = build_secure_relu()


def convert_decoder(decoder, build_secure_conv, build_secure_relu, prf_prefetch):
    convert_conv_module(decoder.image_pool[1], build_secure_conv, build_secure_relu)

    for i in range(4):
        convert_conv_module(decoder.aspp_modules[i], build_secure_conv, build_secure_relu)
    convert_conv_module(decoder.bottleneck, build_secure_conv, build_secure_relu)

    decoder.conv_seg = build_secure_conv(conv_module=decoder.conv_seg, bn_module=None)

    def foo(x):
        return (x.sum(axis=(2, 3), keepdims=True) // (x.shape[2] * x.shape[3])).astype(
            x.dtype)  # TODO: is this the best way to do this?)

    # decoder.image_pool[0] = SecureGlobalAveragePooling2dWithResize()
    # if prf_prefetch:
    #     decoder.image_pool[0] = torch.nn.Identity()
    # else:
    #     decoder.image_pool[0] = SecureGlobalAveragePooling2dWithResize()

    # TODO: replace with SecureGlobalAveragePooling2d
    if prf_prefetch:
        decoder.image_pool[0].forward = lambda x: DummyShapeTensor((x[0], x[1], 1, 1))
    else:
        decoder.image_pool[0].forward = foo


def securify_deeplabv3_mobilenetv2(model, build_secure_conv, build_secure_relu, prf_prefetch=False):
    convert_conv_module(model.backbone.conv1, build_secure_conv, build_secure_relu)

    for layer in [1, 2, 3, 4, 5, 6, 7]:
        cur_layer = getattr(model.backbone, f"layer{layer}")
        for block in cur_layer:
            for conv_module in block.conv:
                convert_conv_module(conv_module, build_secure_conv, build_secure_relu)

    convert_decoder(model.decode_head, build_secure_conv, build_secure_relu, prf_prefetch)

    return model


# TODO: code duplication
def securify_resnet_deeplab(model, max_pool, build_secure_conv, build_secure_relu, prf_prefetch=False, switch_pool_relu=False):
    model.backbone.stem[0] = build_secure_conv(conv_module=model.backbone.stem[0], bn_module=model.backbone.stem[1])
    model.backbone.stem[1] = torch.nn.Identity()
    model.backbone.stem[2] = build_secure_relu()

    model.backbone.stem[3] = build_secure_conv(conv_module=model.backbone.stem[3], bn_module=model.backbone.stem[4])
    model.backbone.stem[4] = torch.nn.Identity()
    model.backbone.stem[5] = build_secure_relu()

    model.backbone.stem[6] = build_secure_conv(conv_module=model.backbone.stem[6], bn_module=model.backbone.stem[7])
    model.backbone.stem[7] = torch.nn.Identity()
    model.backbone.stem[8] = build_secure_relu()

    if hasattr(model.backbone, "maxpool"):
        model.backbone.maxpool = max_pool()
        if switch_pool_relu:
            model.backbone.stem[8], model.backbone.maxpool = model.backbone.maxpool, model.backbone.stem[8]

    for layer in [1, 2, 3, 4]:
        cur_res_layer = getattr(model.backbone, f"layer{layer}")
        for block in cur_res_layer:

            block.conv1 = build_secure_conv(conv_module=block.conv1, bn_module=block.bn1)
            block.bn1 = torch.nn.Identity()
            block.relu_1 = build_secure_relu()

            block.conv2 = build_secure_conv(conv_module=block.conv2, bn_module=block.bn2)
            block.bn2 = torch.nn.Identity()
            block.relu_2 = build_secure_relu()

            if hasattr(block, "conv3"):
                block.conv3 = build_secure_conv(conv_module=block.conv3, bn_module=block.bn3)
                block.bn3 = torch.nn.Identity()
                block.relu_3 = build_secure_relu()

            if block.downsample:
                block.downsample = build_secure_conv(conv_module=block.downsample[0], bn_module=block.downsample[1])

    convert_decoder(model.decode_head, build_secure_conv, build_secure_relu, prf_prefetch)
    return model


def securify_resnet(model, max_pool, build_secure_conv, build_secure_relu, build_secure_fully_connected, prf_prefetch=False, switch_pool_relu=False):
    model.backbone.conv1 = build_secure_conv(conv_module=model.backbone.conv1, bn_module=model.backbone.bn1)
    model.backbone.bn1 = torch.nn.Identity()
    model.backbone.relu = build_secure_relu()

    if hasattr(model.backbone, "maxpool"):
        model.backbone.maxpool = max_pool()
        if switch_pool_relu:
            model.backbone.relu, model.backbone.maxpool = model.backbone.maxpool, model.backbone.relu

    for layer in [1, 2, 3, 4]:
        cur_res_layer = getattr(model.backbone, f"layer{layer}")
        for block in cur_res_layer:

            block.conv1 = build_secure_conv(conv_module=block.conv1, bn_module=block.bn1)
            block.bn1 = torch.nn.Identity()
            block.relu_1 = build_secure_relu()

            block.conv2 = build_secure_conv(conv_module=block.conv2, bn_module=block.bn2)
            block.bn2 = torch.nn.Identity()
            block.relu_2 = build_secure_relu()

            if hasattr(block, "conv3"):
                block.conv3 = build_secure_conv(conv_module=block.conv3, bn_module=block.bn3)
                block.bn3 = torch.nn.Identity()
                block.relu_3 = build_secure_relu()

            if block.downsample:
                block.downsample = build_secure_conv(conv_module=block.downsample[0], bn_module=block.downsample[1])

    model.head.fc = build_secure_fully_connected(conv_module=model.head.fc, bn_module=None)

    if prf_prefetch:
        model.neck = PRFPrefetchSecureGlobalAveragePooling2d()
    else:
        model.neck = SecureGlobalAveragePooling2d()


class SecureGlobalAveragePooling2d(nn.Module):
    def __init__(self):
        super(SecureGlobalAveragePooling2d, self).__init__()

    # TODO: is this the best way to do this?)
    def forward(self, x):
        return backend.mean(x, axis=(2, 3), keepdims=True, dtype=x.dtype)


class SecureGlobalAveragePooling2dWithResize(nn.Module):
    def __init__(self):
        super(SecureGlobalAveragePooling2dWithResize, self).__init__()

    # TODO: is this the best way to do this?)
    def forward(self, x):
        t = backend.mean(x, axis=(2, 3), keepdims=True, dtype=x.dtype)

        return t.repeat(x.shape[2], axis=2).repeat(x.shape[3], axis=3)


class PRFPrefetchSecureGlobalAveragePooling2d(nn.Module):
    def __init__(self):
        super(PRFPrefetchSecureGlobalAveragePooling2d, self).__init__()

    # TODO: is this the best way to do this?)
    def forward(self, x):
        return DummyShapeTensor((x[0], x[1], 1, 1))


class MyAvgPoolFetcher(nn.Module):
    def __init__(self):
        super(MyAvgPoolFetcher, self).__init__()

    # TODO: is this the best way to do this?)
    def forward(self, x):
        return DummyShapeTensor((x[0], x[1], (x[2] + 1) // 2, (x[3] + 1) // 2))


class MyAvgPool(nn.Module):
    def __init__(self):
        super(MyAvgPool, self).__init__()
        self.r = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    # TODO: is this the best way to do this?)

    def forward(self, x):
        out = self.r(torch.from_numpy(x.astype(np.int64))).numpy().astype(x.dtype)

        return out


def get_secure_model(cfg, checkpoint_path, build_secure_conv, build_secure_relu, build_secure_fully_connected, max_pool,
                     secure_model_class, crypto_assets, network_assets, dummy_relu, block_relu=None,
                     relu_spec_file=None, prf_fetcher=None, device="cpu"):
    build_secure_conv = partial(build_secure_conv, crypto_assets=crypto_assets, network_assets=network_assets,
                                device=device)
    build_secure_fully_connected = partial(build_secure_fully_connected, crypto_assets=crypto_assets,
                                           network_assets=network_assets, device=device)
    build_secure_relu = partial(build_secure_relu, crypto_assets=crypto_assets, network_assets=network_assets,
                                dummy_relu=dummy_relu, device=device)
    max_pool = partial(max_pool, crypto_assets=crypto_assets, network_assets=network_assets, device=device)
    block_relu = partial(block_relu, crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu,
                         device=device)

    secure_model_class = partial(secure_model_class, crypto_assets=crypto_assets, network_assets=network_assets,
                                 device=device)

    model = get_model(
        config=cfg,
        gpu_id=None,
        checkpoint_path=checkpoint_path
    )

    if cfg.model.type == "EncoderDecoder" and cfg.model.backbone.type == "MobileNetV2":
        securify_deeplabv3_mobilenetv2(model,
                                       build_secure_conv,
                                       build_secure_relu)
    if cfg.model.type == "EncoderDecoder" and cfg.model.backbone.type in ["AvgPoolResNetSeg", "MyResNetSeg"]:
        max_pool_layer = MyAvgPool if cfg.model.backbone.type == 'AvgPoolResNetSeg' else max_pool

        securify_resnet_deeplab(model,
                                max_pool_layer,
                                build_secure_conv,
                                build_secure_relu,
                                switch_pool_relu=cfg.model.backbone.type == "MyResNetSeg")

    elif cfg.model.type == "ImageClassifier" and cfg.model.backbone.type in ['AvgPoolResNet', "MyResNet", 'ResNet_CIFAR_V2']:
        max_pool_layer = MyAvgPool if cfg.model.backbone.type == 'AvgPoolResNet' else max_pool
        securify_resnet(model,
                        max_pool_layer,
                        build_secure_conv,
                        build_secure_relu,
                        build_secure_fully_connected,
                        switch_pool_relu=cfg.model.backbone.type == 'MyResNet')

    else:
        raise NotImplementedError(f"{cfg.model.type} {cfg.model.backbone.type}")
    if relu_spec_file:
        block_relu = partial(block_relu, crypto_assets=crypto_assets, network_assets=network_assets,
                             dummy_relu=dummy_relu)

        layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))
        arch_utils = arch_utils_factory(cfg)
        arch_utils.set_bReLU_layers(model, layer_name_to_block_sizes, block_relu_class=block_relu)

    ret = secure_model_class(model)
    ret.prf_fetcher = prf_fetcher
    return ret


def init_prf_fetcher(cfg, checkpoint_path, max_pool, build_secure_conv, build_secure_relu, build_secure_fully_connected,
                     prf_fetcher_secure_model, secure_block_relu, relu_spec_file, crypto_assets, network_assets,
                     dummy_relu, device):

    build_secure_conv = partial(build_secure_conv, crypto_assets=crypto_assets, network_assets=network_assets,
                                is_prf_fetcher=True, device=device)
    build_secure_fully_connected = partial(build_secure_fully_connected, crypto_assets=crypto_assets,
                                           network_assets=network_assets, is_prf_fetcher=True, device=device)
    build_secure_relu = partial(build_secure_relu, crypto_assets=crypto_assets, network_assets=network_assets,
                                dummy_relu=dummy_relu, is_prf_fetcher=True, device=device)

    max_pool = partial(max_pool, crypto_assets=crypto_assets, network_assets=network_assets, is_prf_fetcher=True,
                       device=device)
    secure_block_relu = partial(secure_block_relu, crypto_assets=crypto_assets, network_assets=network_assets,
                                dummy_relu=dummy_relu, is_prf_fetcher=True, device=device)
    prf_fetcher_secure_model = partial(prf_fetcher_secure_model, crypto_assets=crypto_assets,
                                       network_assets=network_assets, is_prf_fetcher=True, device=device)

    prf_fetcher_model = get_model(
        config=cfg,
        gpu_id=None,
        checkpoint_path=None
    )

    if cfg.model.type == "EncoderDecoder" and cfg.model.backbone.type == "MobileNetV2":
        securify_deeplabv3_mobilenetv2(
            model=prf_fetcher_model,
            build_secure_conv=build_secure_conv,
            build_secure_relu=build_secure_relu,
            prf_prefetch=True
        )
    elif cfg.model.type == "EncoderDecoder" and cfg.model.backbone.type == "AvgPoolResNetSeg":
        max_pool_layer = MyAvgPoolFetcher if cfg.model.backbone.type == 'AvgPoolResNetSeg' else max_pool

        securify_resnet_deeplab(prf_fetcher_model,
                                max_pool_layer,
                                build_secure_conv,
                                build_secure_relu,
                                prf_prefetch=True,

                                switch_pool_relu=cfg.model.backbone.type == "MyResNetV1cSeg")

    elif cfg.model.type == "ImageClassifier" and cfg.model.backbone.type in ['AvgPoolResNet', "MyResNet", "ResNet_CIFAR_V2"]:
        max_pool_layer = MyAvgPoolFetcher if cfg.model.backbone.type == 'AvgPoolResNet' else max_pool

        securify_resnet(
            model=prf_fetcher_model,
            max_pool=max_pool_layer,
            build_secure_conv=build_secure_conv,
            build_secure_relu=build_secure_relu,
            build_secure_fully_connected=build_secure_fully_connected,
            prf_prefetch=True,
            switch_pool_relu=cfg.model.backbone.type == 'MyResNet'
        )
    else:
        raise NotImplementedError(f"{cfg.model.type} {cfg.model.backbone.type}")

    if relu_spec_file:
        secure_block_relu = partial(secure_block_relu, crypto_assets=crypto_assets, network_assets=network_assets,
                                    dummy_relu=dummy_relu)

        layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))
        arch_utils = arch_utils_factory(cfg)
        arch_utils.set_bReLU_layers(prf_fetcher_model, layer_name_to_block_sizes, block_relu_class=secure_block_relu)
    prf_fetcher_model = prf_fetcher_secure_model(prf_fetcher_model)
    return prf_fetcher_model
