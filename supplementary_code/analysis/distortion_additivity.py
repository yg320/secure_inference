import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import argparse
import copy
import os
from tqdm import tqdm
from typing import Dict
import pickle
import mmcv
import torch
import numpy as np
import contextlib
from functools import lru_cache
import ctypes

from research.distortion.parameters.factory import param_factory
from research.distortion.utils import get_channels_subset
from research.distortion.utils import get_model
from research.utils import build_data
from research.distortion.arch_utils.factory import arch_utils_factory
from research.distortion.utils import get_channel_order_statistics, get_num_of_channels
from research.distortion.distortion_extractor import DistortionUtils


np.random.seed(1234)

config = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py"
checkpoint = "/home/yakir/epoch_14_avg_pool.pth"
mode = "distortion_extraction"

gpu_id = 0
batch_size = 64
cfg = mmcv.Config.fromfile(config)
params = param_factory(cfg)

layer_name = params.LAYER_NAMES[0]
input_block_name = params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
output_block_name = params.BLOCK_NAMES[-1]

distortion_utils = DistortionUtils(gpu_id=gpu_id, params=params, checkpoint=checkpoint, cfg=cfg, mode=mode)

channel_order_to_layer, channel_order_to_channel, channel_order_to_dim = get_channel_order_statistics(params)
num_channels = get_num_of_channels(params)
channels_to_use = range(num_channels) #np.random.choice(num_channels, 2000, replace=False)

block_size_spec = {layer_name:np.ones((params.LAYER_NAME_TO_DIMS[layer_name][0], 2), dtype=np.int32) for layer_name in params.LAYER_NAMES}
distortions = {layer_name:np.load(os.path.join("/home/yakir/distortion_epoch_14_avg_pool/", f"{layer_name}.npy")) for layer_name in params.LAYER_NAMES}

additive_noises = []
noises = []
additive_losses = []
losses = []

for i in tqdm(range(10000)):

    additive_noise = 0

    for index, channel_order in enumerate(channels_to_use):
        layer_name = channel_order_to_layer[channel_order]
        channel = channel_order_to_channel[channel_order]
        block_size_index = np.random.choice(len(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]) - 1)
        block_size_spec[layer_name][channel] = params.LAYER_NAME_TO_BLOCK_SIZES[layer_name][block_size_index]
        additive_noise -= distortions[layer_name][channel, block_size_index]

    batch_noise = []
    batch_loss = []
    for batch_index in range(8):
        cur_assets = distortion_utils.get_batch_distortion(
            clean_block_size_spec=dict(),
            baseline_block_size_spec=dict(),
            block_size_spec=block_size_spec,
            batch_index=batch_index,
            batch_size=batch_size,
            input_block_name=input_block_name,
            output_block_name=output_block_name)

        batch_noise.append(cur_assets["Noise"])
        batch_loss.append(cur_assets["Distorted Loss"])
    loss = np.mean(batch_loss)
    noise = np.mean(batch_noise)

    additive_noises.append(additive_noise)
    losses.append(loss)
    noises.append(noise)

    pickle.dump({"additive_noises":additive_noises, "noises":noises, "losses":losses},
                open("/home/yakir/distortion_additivity.pickle", "wb"))