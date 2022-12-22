import torch
import pickle
from research.distortion.utils import ArchUtilsFactory
from functools import partial
from research.distortion.utils import get_model

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
        arch_utils = ArchUtilsFactory()('AvgPoolResNet')
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
    decoder.image_pool[0].forward = lambda x: x.sum(dim=[2, 3], keepdims=True) // (x.shape[2] * x.shape[3])


def securify_mobilenetv2_model(model, build_secure_conv, build_secure_relu, secure_model_class, crypto_assets, network_assets, dummy_relu, block_relu=None, relu_spec_file=None):

    build_secure_conv = partial(build_secure_conv, crypto_assets=crypto_assets, network_assets=network_assets)
    build_secure_relu = partial(build_secure_relu, crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu)
    secure_model_class = partial(secure_model_class, crypto_assets=crypto_assets, network_assets=network_assets)

    convert_conv_module(model.backbone.conv1, build_secure_conv, build_secure_relu)

    for layer in [1, 2, 3, 4, 5, 6, 7]:
        cur_layer = getattr(model.backbone, f"layer{layer}")
        for block in cur_layer:
            for conv_module in block.conv:
                convert_conv_module(conv_module, build_secure_conv, build_secure_relu)

    convert_decoder(model.decode_head, build_secure_conv, build_secure_relu)

    if relu_spec_file:
        block_relu = partial(block_relu, crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu)

        layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))
        arch_utils = ArchUtilsFactory()("MobileNetV2")
        arch_utils.set_bReLU_layers(model, layer_name_to_block_sizes, block_relu_class=block_relu)

    return secure_model_class(model)


def init_prf_fetcher(Params, build_secure_conv, build_secure_relu, prf_fetcher_secure_model, secure_block_relu, crypto_assets, network_assets):
    prf_fetcher_model = get_model(
        config=Params.SECURE_CONFIG_PATH,
        gpu_id=None,
        checkpoint_path=None
    )

    prf_fetcher_model = securify_mobilenetv2_model(
        prf_fetcher_model,
        build_secure_conv=partial(build_secure_conv, is_prf_fetcher=True),
        build_secure_relu=partial(build_secure_relu, is_prf_fetcher=True),
        secure_model_class=prf_fetcher_secure_model,
        block_relu=secure_block_relu,
        relu_spec_file=Params.RELU_SPEC_FILE,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        dummy_relu=Params.DUMMY_RELU
    )

    prf_fetcher_model.prf_handler.fetch(repeat=Params.NUM_IMAGES, model=prf_fetcher_model, image=torch.zeros(size=Params.IMAGE_SHAPE, dtype=torch.int64))
