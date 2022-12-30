from torch.nn import Module
import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmseg.utils import get_device, setup_multi_processes
import torch
import torch.nn.functional as F
import numpy as np
from mmseg.datasets import build_dataset
from research.bReLU import BlockRelu

from mmcls.models import build_classifier
from mmseg.models import build_segmentor
from mmdet.models import build_detector

# TODO: most of this garbage is not important. Get rid of it
def get_model(config, gpu_id=None, checkpoint_path=None):
    if type(config) == str:
        cfg = mmcv.Config.fromfile(config)
    else:
        cfg = config

    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.type != 'ImageClassifier':
        cfg.data.test.test_mode = True
    if gpu_id is not None:
        cfg.gpu_ids = [gpu_id]

    cfg.model.train_cfg = None
    if cfg.model.type == 'ImageClassifier':
        model = build_classifier(cfg.model)
    elif cfg.model.type == 'EncoderDecoder':
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    elif cfg.model.type == 'SingleStageDetector':
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
    else:
        raise NotImplementedError
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

        model.CLASSES = checkpoint['meta']['CLASSES']
        if 'PALETTE' in checkpoint['meta']:
            model.PALETTE = checkpoint['meta']['PALETTE']

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    cfg.device = get_device()
    if cfg.model.type != 'ImageClassifier':
        model = revert_sync_batchnorm(model)

    if gpu_id is not None:
        model = model.to(f"cuda:{gpu_id}")
    model = model.eval()

    return model


def get_data(dataset):

    if dataset == "coco_stuff164k":
        cfg = {
            'type': 'COCOStuffDataset',
            'data_root': 'data/coco_stuff164k',
            'img_dir': 'images/train2017',
            'ann_dir': 'annotations/train2017',
            'pipeline': [
                {'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations'},
                {'type': 'Resize', 'img_scale': (2048, 512), 'keep_ratio': True},
                {'type': 'RandomFlip', 'prob': 0.0},
                {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375],
                 'to_rgb': True},
                {'type': 'DefaultFormatBundle'},
                {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}]
        }
        crop_size = 512

    # elif dataset == "coco_stuff10k":
    #     assert False
    #     cfg = {'type': 'COCOStuffDataset',
    #            'data_root': 'data/coco_stuff10k',
    #            'reduce_zero_label': True,
    #            'img_dir': 'images/test2014',
    #            'ann_dir': 'annotations/test2014',
    #            'pipeline': [
    #                {'type': 'LoadImageFromFile'},
    #                {'type': 'LoadAnnotations', 'reduce_zero_label': True},
    #                {'type': 'Resize', 'img_scale': (2048, 512), 'keep_ratio': True},
    #                {'type': 'RandomFlip', 'prob': 0.0},
    #                {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375],
    #                 'to_rgb': True},
    #                {'type': 'DefaultFormatBundle'},
    #                {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}]
    #            }
    elif dataset == "ade_20k":
        cfg = {'type': 'ADE20KDataset',
               'data_root': 'data/ade/ADEChallengeData2016',
               'img_dir': 'images/training',
               'ann_dir': 'annotations/training',
               'pipeline': [
                   {'type': 'LoadImageFromFile'},
                   {'type': 'LoadAnnotations', 'reduce_zero_label': True},
                   {'type': 'Resize', 'img_scale': (2048, 512), 'keep_ratio': True},
                   {'type': 'RandomFlip', 'prob': 0.0},
                   {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375],
                    'to_rgb': True},
                   {'type': 'Pad', 'size': (512, 512), 'pad_val': 0, 'seg_pad_val': 255},
                   {'type': 'DefaultFormatBundle'},
                   {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}]
               }
        crop_size = 512

    elif dataset == "ade_20k_256x256":
        cfg = {'type': 'ADE20KDataset',
               'data_root': 'data/ade/ADEChallengeData2016',
               'img_dir': 'images/training',
               'ann_dir': 'annotations/training',
               'pipeline': [
                   {'type': 'LoadImageFromFile'},
                   {'type': 'LoadAnnotations', 'reduce_zero_label': True},
                   {'type': 'Resize', 'img_scale': (1024, 256), 'keep_ratio': True},
                   {'type': 'RandomFlip', 'prob': 0.0},
                   {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True},
                   {'type': 'Pad', 'size': (256, 256), 'pad_val': 0, 'seg_pad_val': 255},
                   {'type': 'DefaultFormatBundle'},
                   {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}]
               }

        crop_size = 256

    elif dataset == "ade_20k_96x96":
        cfg = {'type': 'ADE20KDataset',
               'data_root': 'data/ade/ADEChallengeData2016',
               'img_dir': 'images/training',
               'ann_dir': 'annotations/training',
               'pipeline': [
                   {'type': 'LoadImageFromFile'},
                   {'type': 'LoadAnnotations', 'reduce_zero_label': True},
                   {'type': 'Resize', 'img_scale': (384, 96), 'keep_ratio': True},
                   {'type': 'RandomFlip', 'prob': 0.0},
                   {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True},
                   {'type': 'Pad', 'size': (96, 96), 'pad_val': 0, 'seg_pad_val': 255},
                   {'type': 'DefaultFormatBundle'},
                   {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}]
               }

        crop_size = 96

    elif dataset == "ade_20k_192x192":
        cfg = {'type': 'ADE20KDataset',
               'data_root': 'data/ade/ADEChallengeData2016',
               'img_dir': 'images/training',
               'ann_dir': 'annotations/training',
               'pipeline': [
                   {'type': 'LoadImageFromFile'},
                   {'type': 'LoadAnnotations', 'reduce_zero_label': True},
                   {'type': 'Resize', 'img_scale': (768, 192), 'keep_ratio': True},
                   {'type': 'RandomFlip', 'prob': 0.0},
                   {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True},
                   {'type': 'Pad', 'size': (192, 192), 'pad_val': 0, 'seg_pad_val': 255},
                   {'type': 'DefaultFormatBundle'},
                   {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}]
               }

        crop_size = 192


    else:
        cfg = None

    dataset = build_dataset(cfg)
    dataset.crop_size = crop_size
    return dataset


def get_channel_order_statistics(params):
    layer_names = params.LAYER_NAMES
    l_2_d = params.LAYER_NAME_TO_DIMS
    channel_order_to_channel = np.hstack([np.arange(l_2_d[layer_name][0]) for layer_name in layer_names])
    channel_order_to_layer = np.hstack([[layer_name] * l_2_d[layer_name][0] for layer_name in layer_names])
    channel_order_to_dim = np.hstack([[l_2_d[layer_name][1]] * l_2_d[layer_name][0] for layer_name in layer_names])

    return channel_order_to_layer, channel_order_to_channel, channel_order_to_dim


def get_num_of_channels(params):
    return sum(params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in params.LAYER_NAMES)


def get_channels_component(params, group=None, group_size=None, seed=123, shuffle=True, channel_ord_range=None):

    if channel_ord_range is None:
        num_of_channels = get_num_of_channels(params)
        channels = np.arange(num_of_channels)
    else:
        channels = np.arange(channel_ord_range[0], channel_ord_range[1])

    if shuffle:
        assert seed is not None
        np.random.seed(seed)
        np.random.shuffle(channels)

    if group_size is not None:
        assert group is not None
        return channels[group_size * group: group_size * (group + 1)]
    else:
        return channels

def center_crop(tensor, size):
    if tensor.shape[1] < size or tensor.shape[2] < size:
        raise ValueError
    h = (tensor.shape[1] - size) // 2
    w = (tensor.shape[2] - size) // 2
    return tensor[:, h:h + size, w:w + size]

class ArchUtils:
    def __init__(self):
        pass

    def set_layers(self, model, layer_names_to_layers):
        for layer_name, layer in layer_names_to_layers.items():
            self.set_layer(model, layer_name, layer)

    def set_bReLU_layers(self, model, layer_name_to_block_sizes, block_relu_class=BlockRelu):
        layer_name_to_layers = {layer_name: block_relu_class(block_sizes=block_sizes)
                                for layer_name, block_sizes in layer_name_to_block_sizes.items()}
        self.set_layers(model, layer_name_to_layers)

class ResNetUtils(ArchUtils):
    def __init__(self):
        pass

    def run_model_block(self, model, activation, block_name):
        if block_name == "stem":
            activation = model.backbone.stem(activation)
            activation = model.backbone.maxpool(activation)
        elif block_name == "decode":
            activation = model.decode_head([None, None, None, activation])

        #     activation = model.decode_head._forward_feature([None, None, None, activation])
        #
        # elif block_name == "cls_decode":
        #     activation = model.decode_head.cls_seg(activation)

        else:
            res_layer_name, block_name = block_name.split("_")
            layer = getattr(model.backbone, res_layer_name)
            res_block = layer._modules[block_name]
            activation = res_block(activation)

        return activation


    def get_layer(self, model, layer_name):
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
            return getattr(res_block, f"relu_{relu_name}")

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


    def set_layer(self, model, layer_name, block_relu):
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
            setattr(res_block, f"relu_{relu_name}", block_relu)


from research.distortion.arch_utils.classification.ResNet_CIFAR import ResNet_CIFAR_Utils as ResNet_CIFAR_Classification_Utils
from research.distortion.arch_utils.detection.ssd.ssdlite_mobilenetv2_scratch_600e_coco import MobileNetV2_Utils as MobileNetV2_Detection_Utils
from research.distortion.arch_utils.segmentation.MobileNetV2 import MobileNetV2_Utils as MobileNetV2_Segmentation_Utils

class ArchUtilsFactory:
    def __call__(self, model_cfg):
        arch_name = model_cfg.backbone.type
        model_type = model_cfg.type

        if model_type == "SingleStageDetector" and arch_name == "MobileNetV2":
            return MobileNetV2_Detection_Utils()
        if model_type == 'ImageClassifier' and arch_name == 'ResNet_CIFAR':
            return ResNet_CIFAR_Classification_Utils()
        if model_type == "EncoderDecoder" and arch_name == "MobileNetV2":
            return MobileNetV2_Segmentation_Utils()

        assert False
        # if arch_name in ["ResNetV1c", "SecureResNet", "MyResNet", "AvgPoolResNet"]:
        #     return ResNetUtils()
        # elif arch_name == "MobileNetV2":
        #     return MobileNetUtils()
        # elif arch_name == "ResNet_CIFAR_V2":
        #     return ResNet_CIFAR_Utils()
        # else:
        #     assert False


if __name__ == "__main__":
    cfg = {'type': 'ADE20KDataset',
           'data_root': 'data/ade/ADEChallengeData2016',
           'img_dir': 'images/validation',
           'ann_dir': 'annotations/validation',
           'pipeline': [
               {'type': 'LoadImageFromFile'},
               {'type': 'MultiScaleFlipAug',
                'img_scale': (256, 256),
                'flip': False,
                'transforms': [
                    {'type': 'Resize', 'keep_ratio': False},
                    {'type': 'RandomFlip'},
                    {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True},
                    {'type': 'ImageToTensor', 'keys': ['img']},
                    {'type': 'Collect', 'keys': ['img']}]}],
           'test_mode': True}
    dataset = build_dataset(cfg)
    print(dataset[0])
