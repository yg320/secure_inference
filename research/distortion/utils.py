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
# from mmdet.models import build_detector

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

#
# def center_crop(tensor, size):
#     if tensor.shape[1] < size or tensor.shape[2] < size:
#         raise ValueError
#     h = (tensor.shape[1] - size) // 2
#     w = (tensor.shape[2] - size) // 2
#     return tensor[:, h:h + size, w:w + size]
#
