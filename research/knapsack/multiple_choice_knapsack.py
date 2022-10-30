import glob
import os.path
import time

import numpy as np
import torch
from collections import defaultdict
from research.block_relu.params import ResNetParams
from tqdm import tqdm
import argparse
import pickle

time.sleep(7200)
from research.block_relu.params import ParamsFactory, MobileNetV2_256_Params
from research.parameters.base import MobileNetV2_256_Params_2_Groups
import gc

params = MobileNetV2_256_Params_2_Groups()
params.DATASET = "ade_20k"
params.CONFIG = "/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/deeplabv3_m-v2-d8_256x256_160k_ade20k.py"
params.CHECKPOINT = "/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_m-v2-d8_256x256_160k_ade20k/iter_160000.pth"
channel_distortion_path = "/home/yakir/Data2/assets_v4/distortions/ade_20k/MobileNetV2_256/channels_distortion_2_groups"
iter_ = 1
layer_names = params.LAYER_GROUPS[iter_]


ratio = 1 / 3 / 4
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

    files = glob.glob(os.path.join(channel_distortion_path, f"{layer_name}_*.pickle"))
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
        files = glob.glob(f"/home/yakir/Data2/dp_data/*.npy")
        np.save(file=f"/home/yakir/Data2/dp_data/dp_arg_{channel}.npy", arr=dp_arg)
        np.save(file=f"/home/yakir/Data2/dp_data/dp_{channel}.npy", arr=dp)
        for f in files:
            os.remove(f)
        gc.collect()
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
pickle.dump(obj=layer_name_to_block_size, file=open(f"/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_2_groups_iter_{iter_}_0.0833.pickle", 'wb'))


import pickle

out = {
    **pickle.load(file=open(f"/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_2_groups_iter_0_0.0833.pickle", 'rb')),
    **pickle.load(file=open(f"/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_2_groups_iter_1_0.0833.pickle", 'rb')),
}

pickle.dump(out, open(f"/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_2_groups_iter_01_0.0833.pickle", 'wb'))