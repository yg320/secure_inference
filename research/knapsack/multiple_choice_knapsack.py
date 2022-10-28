import glob
import os.path

import numpy as np
import torch
from collections import defaultdict
from research.block_relu.params import ResNetParams
from tqdm import tqdm
import argparse
import pickle
from research.block_relu.params import ParamsFactory, MobileNetV2_256_Params

iter_ = 3
layer_names = [
    [
        "conv1",
        "layer1_0_0",
        "layer2_0_0",
        "layer2_0_1",
        "layer2_1_0",
        "layer2_1_1",
        "layer3_0_0",
        "layer3_0_1",
        "layer3_1_0",
        "layer3_1_1",
        "layer3_2_0",
        "layer3_2_1",
    ],
    [
        "layer4_0_0",
        "layer4_0_1",
        "layer4_1_0",
        "layer4_1_1",
        "layer4_2_0",
        "layer4_2_1",
        "layer4_3_0",
        "layer4_3_1",
        "layer5_0_0",
        "layer5_0_1",
        "layer5_1_0",
        "layer5_1_1",
        "layer5_2_0",
        "layer5_2_1",
    ],
    [
        "layer6_0_0",
        "layer6_0_1",
        "layer6_1_0",
        "layer6_1_1",
        "layer6_2_0",
        "layer6_2_1",
        "layer7_0_0",
        "layer7_0_1"
    ],
    [
        'decode_0',
        'decode_1',
        'decode_2',
        'decode_3',
        'decode_4',
        'decode_5'
    ]

][iter_]
ratio = 1 / 3 / 4

params = MobileNetV2_256_Params()
params.DATASET = "ade_20k"
params.CONFIG = "/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/deeplabv3_m-v2-d8_256x256_160k_ade20k.py"
params.CHECKPOINT = "/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_m-v2-d8_256x256_160k_ade20k/iter_160000.pth"

def get_block_index_to_num_relus(block_sizes, layer_dim):
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
    return W
Ps = []
Ws = []
layer_name_to_block_shape_index_and_channel_to_block_size = {}
for layer_name in layer_names:
    block_sizes = np.array(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])
    assert np.max(block_sizes) <= 255
    layer_dim = params.LAYER_NAME_TO_DIMS[layer_name][1]

    W = get_block_index_to_num_relus(block_sizes, layer_dim)

    files = glob.glob(
        f"/home/yakir/Data2/assets_v4/distortions/ade_20k/MobileNetV2_256/channels_distortion/{layer_name}_*.pickle")
    noise = np.stack([pickle.load(open(f, 'rb'))["Noise"] for f in files])
    noise = noise.mean(axis=0).mean(axis=2).T

    P = -np.array(noise)

    block_size_groups = defaultdict(list)
    for block_size_index, block_size in enumerate(block_sizes):
        block_size_groups[block_size[0] * block_size[1]].append(block_size_index)

    P_new = []
    W_new = []
    block_shape_index_and_channel_to_block_size = []
    for k, v in block_size_groups.items():
        cur_block_sizes = block_sizes[v]

        P_same_weight = np.stack([P[row_index] for row_index in v])
        argmax = P_same_weight.argmax(axis=0)
        max_ = P_same_weight.max(axis=0)

        block_shape_index_and_channel_to_block_size.append(cur_block_sizes[argmax])
        P_new.append(max_)
        W_new.append(W[v[0]])
    layer_name_to_block_shape_index_and_channel_to_block_size[layer_name] = np.array(block_shape_index_and_channel_to_block_size)
    P = np.array(P_new).T
    W = np.array(W_new).T
    Ps.append(P)
    Ws.append(W)

channels = sum(params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names)
num_relus = int(sum(np.prod(params.LAYER_NAME_TO_DIMS[layer_name]) for layer_name in layer_names) * ratio)
arange = np.arange(num_relus)

dp = - np.inf * np.ones(shape=(num_relus,))
dp[Ws[0]] = Ps[0][0]

dp_arg = np.zeros(shape=(channels, num_relus), dtype=np.uint8)
dp_arg[0, Ws[0]] = np.arange(Ws[0].shape[0])


channels_batch = [params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names]
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









arg_and_channel_to_block_size = np.concatenate([layer_name_to_block_shape_index_and_channel_to_block_size[layer_name] for layer_name in layer_names], axis=1)#.transpose([1,0,2])

channels_stack = np.hstack([[layer_index] * params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_index, layer_name in enumerate(layer_names)])
layer_channel = np.hstack([np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]) for layer_name in layer_names])

channel_order_to_layer_index = np.hstack([[layer_index] * params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_index, layer_name in enumerate(layer_names)])
channel_order_to_channel_index = np.hstack([np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]) for layer_name in layer_names])

all_block_sizes = []
relu_count = dp_arg.shape[1] - 1
block_sizes = []
for channel in reversed(range(channels)):
    arg = dp_arg[channel, relu_count]

    channel_num_relus = Ws[channel_order_to_layer_index[channel]][arg]
    relu_count -= channel_num_relus
    block_sizes.append(arg_and_channel_to_block_size[arg,channel])
block_sizes = np.array(block_sizes[::-1])


cur_chan = 0
layer_name_to_block_size = dict()
for layer_name in layer_names:
    layer_name_to_block_size[layer_name] = block_sizes[cur_chan:cur_chan + params.LAYER_NAME_TO_DIMS[layer_name][0], :]
    cur_chan += params.LAYER_NAME_TO_DIMS[layer_name][0]
pickle.dump(obj=layer_name_to_block_size, file=open(f"/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_iter_{iter_}_0.0833.pickle", 'wb'))