import torch
from research.secure_inference_3pc.const import SIGNED_DTYPE
import pickle
from research.distortion.arch_utils.factory import arch_utils_factory
from functools import partial
from research.distortion.utils import get_model
from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.modules.base import DummyShapeTensor
from research.secure_inference_3pc.timer import timer
def securify_resnet18_model(model, build_secure_conv, build_secure_relu, crypto_assets, network_assets, block_relu=None, relu_spec_file=None):
    model.backbone.stem[0] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[0], model.backbone.stem[1])
    model.backbone.stem[1] = torch.nn.Identity()
    model.backbone.stem[2] = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[3] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[3], model.backbone.stem[4])
    model.backbone.stem[4] = torch.nn.Identity()
    model.backbone.stem[5] = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[6] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[6], model.backbone.stem[7])
    model.backbone.stem[7] = torch.nn.Identity()
    model.backbone.stem[8] = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    for layer in [1, 2, 3, 4]:
        for block in [0, 1]:
            cur_res_layer = getattr(model.backbone, f"layer{layer}")
            cur_res_layer[block].conv1 = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].conv1, cur_res_layer[block].bn1)
            cur_res_layer[block].bn1 = torch.nn.Identity()
            cur_res_layer[block].relu_1 = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

            cur_res_layer[block].conv2 = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].conv2, cur_res_layer[block].bn2)
            cur_res_layer[block].bn2 = torch.nn.Identity()
            cur_res_layer[block].relu_2 = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

            if cur_res_layer[block].downsample:
                cur_res_layer[block].downsample = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].downsample[0], cur_res_layer[block].downsample[1])

    model.decode_head.image_pool[1].conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.image_pool[1].conv, model.decode_head.image_pool[1].bn)
    model.decode_head.image_pool[1].bn = torch.nn.Identity()
    model.decode_head.image_pool[1].activate = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    for i in range(4):
        model.decode_head.aspp_modules[i].conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.aspp_modules[i].conv, model.decode_head.aspp_modules[i].bn)
        model.decode_head.aspp_modules[i].bn = torch.nn.Identity()
        model.decode_head.aspp_modules[i].activate = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    model.decode_head.bottleneck.conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.bottleneck.conv, model.decode_head.bottleneck.bn)
    model.decode_head.bottleneck.bn = torch.nn.Identity()
    model.decode_head.bottleneck.activate = build_secure_relu(crypto_assets=crypto_assets, network_assets=network_assets)

    model.decode_head.conv_seg = build_secure_conv(crypto_assets, network_assets, model.decode_head.conv_seg, None)
    model.decode_head.image_pool[0].forward = lambda x: x.sum(dim=[2, 3], keepdims=True) // (x.shape[2] * x.shape[3])

    if relu_spec_file:
        SecureBlockReLUClient_partial = partial(block_relu, crypto_assets=crypto_assets, network_assets=network_assets)
        layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))
        arch_utils = arch_utils_factory('AvgPoolResNet')
        arch_utils.set_bReLU_layers(model, layer_name_to_block_sizes, block_relu_class=SecureBlockReLUClient_partial)

def convert_conv_module(module, build_secure_conv, build_secure_relu):
    module.conv = build_secure_conv(conv_module=module.conv, bn_module=module.bn)
    module.bn = torch.nn.Identity()
    if hasattr(module, "activate"):
        module.activate = build_secure_relu()

def convert_decoder(decoder, build_secure_conv, build_secure_relu):
    convert_conv_module(decoder.image_pool[1], build_secure_conv, build_secure_relu)

    for i in range(4):
        convert_conv_module(decoder.aspp_modules[i], build_secure_conv, build_secure_relu)
    convert_conv_module(decoder.bottleneck, build_secure_conv, build_secure_relu)

    decoder.conv_seg = build_secure_conv(conv_module=decoder.conv_seg, bn_module=None)
    def foo(x):
        return (x.sum(axis=(2, 3), keepdims=True) // (x.shape[2] * x.shape[3])).astype(x.dtype)  # TODO: is this the best way to do this?)

    # TODO: replace with SecureGlobalAveragePooling2d
    decoder.image_pool[0].forward = foo


def securify_deeplabv3_mobilenetv2(model, build_secure_conv, build_secure_relu, secure_model_class, crypto_assets, network_assets, dummy_relu, block_relu=None, relu_spec_file=None):

    convert_conv_module(model.backbone.conv1, build_secure_conv, build_secure_relu)

    for layer in [1, 2, 3, 4, 5, 6, 7]:
        cur_layer = getattr(model.backbone, f"layer{layer}")
        for block in cur_layer:
            for conv_module in block.conv:
                convert_conv_module(conv_module, build_secure_conv, build_secure_relu)

    convert_decoder(model.decode_head, build_secure_conv, build_secure_relu)


    return model
import torch.nn as nn

class SecureGlobalAveragePooling2d(nn.Module):
    def __init__(self):
        super(SecureGlobalAveragePooling2d, self).__init__()

    # TODO: is this the best way to do this?)
    def forward(self, x):
        return backend.mean(x, axis=(2, 3), keepdims=True, dtype=x.dtype)

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
        return DummyShapeTensor((x[0], x[1], x[2]//2, x[3]//2))

class MyAvgPool(nn.Module):
    def __init__(self):
        super(MyAvgPool, self).__init__()
        self.r = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    # TODO: is this the best way to do this?)

    def forward(self, x):
        return self.r(torch.from_numpy(x)).numpy()
def securify_resnet_cifar(model, max_pool, build_secure_conv, build_secure_relu, build_secure_fully_connected, secure_model_class, crypto_assets, network_assets, dummy_relu, block_relu=None, relu_spec_file=None, prf_prefetch=False):
    model.backbone.conv1 = build_secure_conv(conv_module=model.backbone.conv1, bn_module=model.backbone.bn1)
    model.backbone.bn1 = torch.nn.Identity()
    model.backbone.relu = build_secure_relu()

    model.backbone.maxpool = max_pool()

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

def get_secure_model(cfg, checkpoint_path, build_secure_conv, build_secure_relu, build_secure_fully_connected, max_pool, secure_model_class, crypto_assets, network_assets, dummy_relu, block_relu=None, relu_spec_file=None, dummy_max_pool=False, prf_fetcher=None, device="cpu"):

    build_secure_conv = partial(build_secure_conv, crypto_assets=crypto_assets, network_assets=network_assets, device=device)
    build_secure_fully_connected = partial(build_secure_fully_connected, crypto_assets=crypto_assets, network_assets=network_assets, device=device)
    build_secure_relu = partial(build_secure_relu, crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu, device=device)
    max_pool = partial(max_pool, crypto_assets=crypto_assets, network_assets=network_assets, dummy_max_pool=dummy_max_pool, device=device)
    block_relu = partial(block_relu, crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu, device=device)

    secure_model_class = partial(secure_model_class, crypto_assets=crypto_assets, network_assets=network_assets, device=device)

    model = get_model(
        config=cfg,
        gpu_id=None,
        checkpoint_path=checkpoint_path
    )
    # arr = model.backbone(torch.load("/home/yakir/tmp.pt"))[0]
    # print(model.head.fc(model.neck(arr)).argmax())
    if cfg.model.type == "EncoderDecoder" and cfg.model.backbone.type == "MobileNetV2":
        securify_deeplabv3_mobilenetv2(model, build_secure_conv, build_secure_relu, secure_model_class, crypto_assets, network_assets, dummy_relu, block_relu, relu_spec_file)
    elif cfg.model.type == "ImageClassifier" and cfg.model.backbone.type == "ResNet_CIFAR_V2":
        securify_resnet_cifar(model, build_secure_conv, build_secure_relu, build_secure_fully_connected, secure_model_class, crypto_assets, network_assets, dummy_relu, block_relu, relu_spec_file)
    elif cfg.model.type == "ImageClassifier" and cfg.model.backbone.type in ['AvgPoolResNet', "MyResNet"]:
        securify_resnet_cifar(model, MyAvgPool, build_secure_conv, build_secure_relu, build_secure_fully_connected, secure_model_class, crypto_assets, network_assets, dummy_relu, block_relu, relu_spec_file)
    else:
        raise NotImplementedError(f"{cfg.model.type} {cfg.model.backbone.type}")
    if relu_spec_file:
        block_relu = partial(block_relu, crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu)

        layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))
        arch_utils = arch_utils_factory(cfg)
        arch_utils.set_bReLU_layers(model, layer_name_to_block_sizes, block_relu_class=block_relu)

    ret = secure_model_class(model)
    ret.prf_fetcher = prf_fetcher
    return ret


def init_prf_fetcher(cfg, Params, max_pool, build_secure_conv, build_secure_relu, build_secure_fully_connected, prf_fetcher_secure_model, secure_block_relu, relu_spec_file, crypto_assets, network_assets, dummy_relu, dummy_max_pool, device):

    build_secure_conv = partial(build_secure_conv, crypto_assets=crypto_assets, network_assets=network_assets, is_prf_fetcher=True, device=device)
    build_secure_fully_connected = partial(build_secure_fully_connected, crypto_assets=crypto_assets, network_assets=network_assets, is_prf_fetcher=True, device=device)
    build_secure_relu = partial(build_secure_relu, crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu, is_prf_fetcher=True, device=device)

    max_pool = partial(max_pool, crypto_assets=crypto_assets, network_assets=network_assets, dummy_max_pool=dummy_max_pool, is_prf_fetcher=True, device=device)
    secure_block_relu = partial(secure_block_relu, crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu, is_prf_fetcher=True, device=device)
    prf_fetcher_secure_model = partial(prf_fetcher_secure_model, crypto_assets=crypto_assets, network_assets=network_assets, is_prf_fetcher=True, device=device)

    prf_fetcher_model = get_model(
        config=Params.SECURE_CONFIG_PATH,
        gpu_id=None,
        checkpoint_path=None
    )

    securify_resnet_cifar(
        model=prf_fetcher_model,
        max_pool=MyAvgPoolFetcher,
        build_secure_conv=build_secure_conv,
        build_secure_relu=build_secure_relu,
        build_secure_fully_connected=build_secure_fully_connected,
        secure_model_class=prf_fetcher_secure_model,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        dummy_relu=dummy_relu,
        block_relu=secure_block_relu,
        relu_spec_file=relu_spec_file,
        prf_prefetch=True)

    if relu_spec_file:
        secure_block_relu = partial(secure_block_relu, crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu)

        layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))
        arch_utils = arch_utils_factory(cfg)
        arch_utils.set_bReLU_layers(prf_fetcher_model, layer_name_to_block_sizes, block_relu_class=secure_block_relu)
    prf_fetcher_model = prf_fetcher_secure_model(prf_fetcher_model)
    return prf_fetcher_model
