from torch.nn import Module
import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.utils import get_device, setup_multi_processes
import torch
import torch.nn.functional as F
import numpy as np
from research.block_relu.consts import BLOCK_SIZES
from mmseg.datasets import build_dataset


class BlockRelu(Module):

    def __init__(self, block_size_indices):
        super(BlockRelu, self).__init__()
        self.block_size_indices = block_size_indices
        self.active_block_indices = np.unique(self.block_size_indices)

    def forward(self, activation):

        with torch.no_grad():
            relu_map = torch.zeros_like(activation)
            relu_map[:, self.block_size_indices == 0] = \
                activation[:, self.block_size_indices == 0].sign().add_(1).div_(2)

            for block_size_index in self.active_block_indices:
                if block_size_index == 0:
                    continue
                if block_size_index in self.active_block_indices:
                    channels = self.block_size_indices == block_size_index
                    cur_input = activation[:, channels]

                    avg_pool = torch.nn.AvgPool2d(
                        kernel_size=BLOCK_SIZES[block_size_index],
                        stride=BLOCK_SIZES[block_size_index], ceil_mode=True)

                    cur_relu_map = avg_pool(cur_input).sign_().add_(1).div_(2)
                    o = F.interpolate(input=cur_relu_map, scale_factor=BLOCK_SIZES[block_size_index])
                    relu_map[:, channels] = o[:, :, :activation.shape[2], :activation.shape[3]]

                    torch.cuda.empty_cache()
        return relu_map.mul_(activation)


def get_model(config, gpu_id, checkpoint_path):
    cfg = mmcv.Config.fromfile(config)

    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if gpu_id is not None:
        cfg.gpu_ids = [gpu_id]

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    if checkpoint_path is not None:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    cfg.device = get_device()

    model = revert_sync_batchnorm(model)
    if gpu_id is not None:
        model = model.to(f"cuda:{gpu_id}")
    model = model.eval()

    return model


def get_data(config):
    cfg = mmcv.Config.fromfile(config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    return dataset


def run_model_block(model, activation, block_name):
    if block_name == "stem":
        activation = model.backbone.stem(activation)
        activation = model.backbone.maxpool(activation)
    elif block_name == "decode":
        activation = model.decode_head([None, None, None, activation])
    else:
        res_layer_name, block_name = block_name.split("_")
        layer = getattr(model.backbone, res_layer_name)
        res_block = layer._modules[block_name]
        activation = res_block(activation)

    return activation


def get_layer(model, layer_name):
    if "decode" in layer_name:
        if layer_name == "decode_0":
            return model.decode_head.image_pool[1].activate
        elif layer_name == "decode_5":
            return model.decode_head.bottleneck.activate
        else:
            _, relu_name = layer_name.split("_")
            relu_index = int(relu_name)
            assert relu_index in [1, 2, 3, 4]
            return model.decode_head.aspp_modules[relu_index - 1].activate
    elif "stem" in layer_name:
        _, relu_name = layer_name.split("_")
        return model.backbone.stem._modules[relu_name]
    else:
        res_layer_name, block_name, relu_name = layer_name.split("_")

        layer = getattr(model.backbone, res_layer_name)
        res_block = layer._modules[block_name]
        return getattr(res_block, f"secure_relu_{relu_name}")

    # stem_2
    # stem_5
    # stem_8
    # layer1_0_1
    # layer1_0_2
    # layer1_0_3
    # layer1_1_1
    # layer1_1_2
    # layer1_1_3
    # layer1_2_1
    # layer1_2_2
    # layer1_2_3
    # layer2_0_1
    # layer2_0_2
    # layer2_0_3
    # layer2_1_1
    # layer2_1_2
    # layer2_1_3
    # layer2_2_1
    # layer2_2_2
    # layer2_2_3
    # layer2_3_1
    # layer2_3_2
    # layer2_3_3
    # layer3_0_1
    # layer3_0_2
    # layer3_0_3
    # layer3_1_1
    # layer3_1_2
    # layer3_1_3
    # layer3_2_1
    # layer3_2_2
    # layer3_2_3
    # layer3_3_1
    # layer3_3_2
    # layer3_3_3
    # layer3_4_1
    # layer3_4_2
    # layer3_4_3
    # layer3_5_1
    # layer3_5_2
    # layer3_5_3
    # layer4_0_1
    # layer4_0_2
    # layer4_0_3
    # layer4_1_1
    # layer4_1_2
    # layer4_1_3
    # layer4_2_1
    # layer4_2_2
    # layer4_2_3
    # decode_0
    # decode_1
    # decode_2
    # decode_3
    # decode_4
    # decode_5

    # model.backbone.stem._modules['2']
    # model.backbone.stem._modules['5']
    # model.backbone.stem._modules['8']
    #
    # model.backbone.layer1._modules['0'].secure_relu_1
    # model.backbone.layer1._modules['0'].secure_relu_2
    # model.backbone.layer1._modules['0'].secure_relu_3
    # model.backbone.layer1._modules['1'].secure_relu_1
    # model.backbone.layer1._modules['1'].secure_relu_2
    # model.backbone.layer1._modules['1'].secure_relu_3
    # model.backbone.layer1._modules['2'].secure_relu_1
    # model.backbone.layer1._modules['2'].secure_relu_2
    # model.backbone.layer1._modules['2'].secure_relu_3
    #
    # model.backbone.layer2._modules['0'].secure_relu_1
    # model.backbone.layer2._modules['0'].secure_relu_2
    # model.backbone.layer2._modules['0'].secure_relu_3
    # model.backbone.layer2._modules['1'].secure_relu_1
    # model.backbone.layer2._modules['1'].secure_relu_2
    # model.backbone.layer2._modules['1'].secure_relu_3
    # model.backbone.layer2._modules['2'].secure_relu_1
    # model.backbone.layer2._modules['2'].secure_relu_2
    # model.backbone.layer2._modules['2'].secure_relu_3
    # model.backbone.layer2._modules['3'].secure_relu_1
    # model.backbone.layer2._modules['3'].secure_relu_2
    # model.backbone.layer2._modules['3'].secure_relu_3
    #
    # model.backbone.layer3._modules['0'].secure_relu_1
    # model.backbone.layer3._modules['0'].secure_relu_2
    # model.backbone.layer3._modules['0'].secure_relu_3
    # model.backbone.layer3._modules['1'].secure_relu_1
    # model.backbone.layer3._modules['1'].secure_relu_2
    # model.backbone.layer3._modules['1'].secure_relu_3
    # model.backbone.layer3._modules['2'].secure_relu_1
    # model.backbone.layer3._modules['2'].secure_relu_2
    # model.backbone.layer3._modules['2'].secure_relu_3
    # model.backbone.layer3._modules['3'].secure_relu_1
    # model.backbone.layer3._modules['3'].secure_relu_2
    # model.backbone.layer3._modules['3'].secure_relu_3
    # model.backbone.layer3._modules['4'].secure_relu_1
    # model.backbone.layer3._modules['4'].secure_relu_2
    # model.backbone.layer3._modules['4'].secure_relu_3
    # model.backbone.layer3._modules['5'].secure_relu_1
    # model.backbone.layer3._modules['5'].secure_relu_2
    # model.backbone.layer3._modules['5'].secure_relu_3
    #
    # model.backbone.layer4._modules['0'].secure_relu_1
    # model.backbone.layer4._modules['0'].secure_relu_2
    # model.backbone.layer4._modules['0'].secure_relu_3
    # model.backbone.layer4._modules['1'].secure_relu_1
    # model.backbone.layer4._modules['1'].secure_relu_2
    # model.backbone.layer4._modules['1'].secure_relu_3
    # model.backbone.layer4._modules['2'].secure_relu_1
    # model.backbone.layer4._modules['2'].secure_relu_2
    # model.backbone.layer4._modules['2'].secure_relu_3
    #
    # model.decode_head.image_pool[1].activate
    # model.decode_head.aspp_modules[0]
    # model.decode_head.aspp_modules[1]
    # model.decode_head.aspp_modules[2]
    # model.decode_head.aspp_modules[3]
    # model.decode_head.bottleneck.activate


def set_layer(model, layer_name, block_relu):
    if "decode" in layer_name:
        if layer_name == "decode_0":
            model.decode_head.image_pool[1].activate = block_relu
        elif layer_name == "decode_5":
            model.decode_head.bottleneck.activate = block_relu
        else:
            _, relu_name = layer_name.split("_")
            relu_index = int(relu_name)
            assert relu_index in [1, 2, 3, 4]
            model.decode_head.aspp_modules[relu_index - 1].activate = block_relu
    elif "stem" in layer_name:
        _, relu_name = layer_name.split("_")
        model.backbone.stem._modules[relu_name] = block_relu
    else:
        res_layer_name, block_name, relu_name = layer_name.split("_")

        layer = getattr(model.backbone, res_layer_name)
        res_block = layer._modules[block_name]
        setattr(res_block, f"secure_relu_{relu_name}", block_relu)


def set_bReLU_layers(model, layer_name_to_block_size_indices):
    layer_name_to_layers = {layer_name: BlockRelu(block_size_indices=block_size_indices) for
                            layer_name, block_size_indices in layer_name_to_block_size_indices.items()}
    set_layers(model, layer_name_to_layers)


def set_layers(model, layer_names_to_layers):
    for layer_name, layer in layer_names_to_layers.items():
        set_layer(model, layer_name, layer)
