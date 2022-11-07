import glob
import os.path

import numpy as np
import torch
from collections import defaultdict
from research.block_relu.params import ResNetParams
from tqdm import tqdm
import argparse

BLOCKS_LAYERS = \
    [
        ["stem_2",
         "stem_5",
         "stem_8"],
        ["layer1_0_1",
         "layer1_0_2",
         "layer1_0_3"],
        ["layer1_1_1",
         "layer1_1_2",
         "layer1_1_3"],
        ["layer1_2_1",
         "layer1_2_2",
         "layer1_2_3"],
        ["layer2_0_1",
         "layer2_0_2",
         "layer2_0_3"],
        ["layer2_1_1",
         "layer2_1_2",
         "layer2_1_3"],
        ["layer2_2_1",
         "layer2_2_2",
         "layer2_2_3"],
        ["layer2_3_1",
         "layer2_3_2",
         "layer2_3_3"],
        ["layer3_0_1",
         "layer3_0_2",
         "layer3_0_3"],
        ["layer3_1_1",
         "layer3_1_2",
         "layer3_1_3"],
        ["layer3_2_1",
         "layer3_2_2",
         "layer3_2_3"],
        ["layer3_3_1",
         "layer3_3_2",
         "layer3_3_3"],
        ["layer3_4_1",
         "layer3_4_2",
         "layer3_4_3"],
        ["layer3_5_1",
         "layer3_5_2",
         "layer3_5_3"],
        ["layer4_0_1",
         "layer4_0_2",
         "layer4_0_3"],
        ["layer4_1_1",
         "layer4_1_2",
         "layer4_1_3"],
        ["layer4_2_1",
         "layer4_2_2",
         "layer4_2_3"],
        ["decode_0",
         "decode_1",
         "decode_2",
         "decode_3",
         "decode_4",
         "decode_5"]
    ]


params = ResNetParams(HIERARCHY_NAME=None,
                      LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS=None,
                      LAYER_HIERARCHY_SPEC=None,
                      DATASET="coco_stuff164k",
                      CONFIG="/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/my_resnet_coco-stuff_164k.py",
                      CHECKPOINT="/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k/iter_80000.pth")
group_to_relevant_ratio = [0.207, 0.126, 0.048, 0.062, 0.263, 0.059, 0.147, 0.237, 0.248, 0.039, 0.045, 0.041, 0.043, 0.108, 0.021, 0.039, 0.05, 0.001]
# group_to_relevant_ratio = [0.207, 0.126, 0.048, 0.062, 0.01, 0.059, 0.147, 0.237, 0.248, 0.039, 0.045, 0.041, 0.043, 0.108, 0.021, 0.039, 0.05, 0.001]
for group_index in np.arange(2,18,3):
    layer_names = BLOCKS_LAYERS[group_index]

    Ps = []
    Ws = []
    layer_index_to_index_to_block_sizes = []
    for layer_name in layer_names:
        block_sizes = np.array(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])
        assert np.max(block_sizes) <= 255
        layer_dim = int(np.sqrt(params.LAYER_NAME_TO_RELU_COUNT[layer_name] // params.LAYER_NAME_TO_CHANNELS[layer_name]))

        block_index_to_num_relus = []
        for block_size_index, block_size in enumerate(block_sizes):
            avg_pool = torch.nn.AvgPool2d(
                kernel_size=tuple(block_size),
                stride=tuple(block_size), ceil_mode=True)

            cur_input = torch.zeros(size=(1, 1, layer_dim, layer_dim))
            cur_relu_map = avg_pool(cur_input)
            num_relus = cur_relu_map.shape[2] * cur_relu_map.shape[3]
            block_index_to_num_relus.append(num_relus)
        W = np.array(block_index_to_num_relus)

        files = glob.glob(
            f"/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/block/noise_{layer_name}_batch_*_8.npy")
        noise = np.stack([np.load(f) for f in files]).mean(axis=0)


        P = -np.array(noise)

        block_size_groups = defaultdict(list)
        for block_size_index, block_size in enumerate(block_sizes):
            block_size_groups[block_size[0] * block_size[1]].append(block_size_index)

        P_new = []
        W_new = []
        index_to_block_sizes = []
        for k, v in block_size_groups.items():
            cur_block_sizes = block_sizes[v]

            P_same_weight = np.stack([P[row_index] for row_index in v])
            argmax = P_same_weight.argmax(axis=0)
            max_ = P_same_weight.max(axis=0)

            index_to_block_sizes.append(cur_block_sizes[argmax])
            P_new.append(max_)
            W_new.append(W[v[0]])
        layer_index_to_index_to_block_sizes.append(np.array(index_to_block_sizes))
        P = np.array(P_new).T
        W = np.array(W_new).T
        Ps.append(P)
        Ws.append(W)

    channels = sum(params.LAYER_NAME_TO_CHANNELS[layer_name] for layer_name in layer_names)
    real_num_relus = sum(params.LAYER_NAME_TO_RELU_COUNT[layer_name] for layer_name in layer_names)
    num_relus = int(real_num_relus * group_to_relevant_ratio[group_index]) + 1
    arange = np.arange(num_relus)

    dp = - np.inf * np.ones(shape=(num_relus,))
    dp[Ws[0]] = Ps[0][0]

    dp_arg = np.zeros(shape=(channels, num_relus), dtype=np.uint8)
    dp_arg[0, Ws[0]] = np.arange(Ws[0].shape[0])


    channels_batch = [params.LAYER_NAME_TO_CHANNELS[layer_name] for layer_name in layer_names]
    layer_index = 0
    all_indices = np.maximum(np.arange(num_relus)[:, np.newaxis] - Ws[layer_index], 0)
    orig_shape = all_indices.shape
    all_indices = all_indices.flatten()
    cur_P = Ps[layer_index]
    offset = 0
    for channel in tqdm(range(1, channels)):
        if channel == sum(channels_batch[:layer_index+1]):
            layer_index += 1
            all_indices = np.maximum(np.arange(num_relus)[:, np.newaxis] - Ws[layer_index], 0)
            orig_shape = all_indices.shape
            all_indices = all_indices.flatten()
            cur_P = Ps[layer_index]
            offset = sum(channels_batch[:layer_index])
        r = dp[all_indices].reshape(orig_shape) + cur_P[channel -offset]
        dp_arg[channel] = np.argmax(r, axis=1)
        dp = r[arange, dp_arg[channel]]


    relus_count = np.floor(np.arange(0, group_to_relevant_ratio[group_index] + 0.001, 0.001) * (real_num_relus - 1)).astype(np.int32)

    channels_stack = np.hstack([[layer_index] * params.LAYER_NAME_TO_CHANNELS[layer_name] for layer_index, layer_name in enumerate(layer_names)])
    layer_channel = np.hstack([np.arange(params.LAYER_NAME_TO_CHANNELS[layer_name]) for layer_name in layer_names])
    all_block_sizes = []
    for reduction_index, relu_count in enumerate(relus_count):
        block_sizes = []
        for channel in reversed(range(channels)):
            arg = dp_arg[channel, relu_count]

            channel_num_relus = Ws[channels_stack[channel]][arg]
            relu_count -= channel_num_relus
            block_sizes.append(layer_index_to_index_to_block_sizes[channels_stack[channel]][arg, layer_channel[channel]])
        all_block_sizes.append(block_sizes[::-1])

    all_block_sizes = np.array(all_block_sizes)

    cur_chan = 0
    for layer_name in layer_names:
        cur_block_sizes = all_block_sizes[:,cur_chan:cur_chan + params.LAYER_NAME_TO_CHANNELS[layer_name]]
        cur_chan += params.LAYER_NAME_TO_CHANNELS[layer_name]
        np.save(
            file=f"/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/channel_knapsack_resblocks/{layer_name}_reduction_to_block_sizes.npy",
            arr=cur_block_sizes)
