import numpy as np
import torch

import mmcv
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmseg.utils import get_device, setup_multi_processes
from mmcls.models import build_classifier
from mmseg.models import build_segmentor
from functools import lru_cache


@lru_cache(maxsize=None)
def get_num_relus(block_size, activation_dim):
    if block_size in [(0, 1), (1, 0)]:
        return 0

    if block_size == (1, 1):
        return activation_dim ** 2

    avg_pool = torch.nn.AvgPool2d(
        kernel_size=block_size,
        stride=block_size, ceil_mode=True)

    cur_input = torch.zeros(size=(1, 1, activation_dim, activation_dim))
    cur_relu_map = avg_pool(cur_input)
    num_relus = cur_relu_map.shape[2] * cur_relu_map.shape[3]

    return num_relus


@lru_cache(maxsize=None)
def get_brelu_bandwidth(block_size, activation_dim, l=8, log_p=8, protocol="Porthos", scalar_vector_optimization=True,
                        with_prf=True):
    assert with_prf
    if block_size in [(0, 1), (1, 0)]:
        return 0
    num_relus = get_num_relus(block_size, activation_dim)
    if protocol == "Porthos":
        dReLU_bandwidth = (6 * log_p + 19 - 5) * num_relus
    elif protocol == "SecureNN":
        dReLU_bandwidth = (8 * log_p + 24 - 5) * num_relus
    else:
        assert False
    if scalar_vector_optimization:
        assert with_prf
        mult_bandwidth = 3 * activation_dim ** 2 + 2 * num_relus
    else:
        assert with_prf
        mult_bandwidth = 5 * activation_dim ** 2
    bandwidth = l * (dReLU_bandwidth + mult_bandwidth)
    return bandwidth  # Bandwidth is in Bytes (therefore l=8 and not 64)


def get_block_spec_num_relus(block_spec, params):
    num_relus = 0
    for layer_name, block_sizes in block_spec.items():
        activation_dim = params.LAYER_NAME_TO_DIMS[layer_name][1]
        for block_size in block_sizes:
            num_relus += get_num_relus(tuple(block_size), activation_dim)
    return num_relus


def get_channel_order_statistics(params):
    layer_names = params.LAYER_NAMES
    l_2_d = params.LAYER_NAME_TO_DIMS
    channel_order_to_channel = np.hstack([np.arange(l_2_d[layer_name][0]) for layer_name in layer_names])
    channel_order_to_layer = np.hstack([[layer_name] * l_2_d[layer_name][0] for layer_name in layer_names])
    channel_order_to_dim = np.hstack([[l_2_d[layer_name][1]] * l_2_d[layer_name][0] for layer_name in layer_names])

    return channel_order_to_layer, channel_order_to_channel, channel_order_to_dim


def get_num_of_channels(params):
    return sum(params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in params.LAYER_NAMES)


def get_channels_subset(params, seed=None, cur_iter=None, num_iters=None):
    if seed is not None:
        assert cur_iter is not None
        assert num_iters is not None

    layer_names = params.LAYER_NAMES
    total_num_channels = get_num_of_channels(params)
    channel_order_to_layer, channel_order_to_channel, _ = get_channel_order_statistics(params)

    all_channels = np.arange(total_num_channels)

    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(all_channels)
        channels_to_use = np.array_split(all_channels, num_iters)[cur_iter]
    else:
        channels_to_use = all_channels

    channels_to_run = {layer_name: [] for layer_name in params.LAYER_NAMES}
    for channel_order in channels_to_use:
        layer_name = channel_order_to_layer[channel_order]
        channel_in_layer_index = channel_order_to_channel[channel_order]
        channels_to_run[layer_name].append(channel_in_layer_index)

    for layer_name in layer_names:
        channels_to_run[layer_name].sort()

    return channels_to_run, channels_to_use


def get_model(config, gpu_id=None, checkpoint_path=None):
    # TODO: We can clean this up a bit, most of the code here is unnecessary
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
