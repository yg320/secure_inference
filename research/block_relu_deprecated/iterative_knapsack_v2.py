import glob
import os.path

import numpy as np
import torch
from collections import defaultdict
from research.block_relu.params import ResNetParams
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--layer_index', type=int, default=0)

args = parser.parse_args()

params = ResNetParams(HIERARCHY_NAME=None,
                      LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS=None,
                      LAYER_HIERARCHY_SPEC=None,
                      DATASET="coco_stuff164k",
                      CONFIG="/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/my_resnet_coco-stuff_164k.py",
                      CHECKPOINT="/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k/iter_80000.pth")


layer_name = params.LAYER_NAMES[args.layer_index]

files = glob.glob(f"/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/block/noise_{layer_name}_batch_*_8.npy")
noise = np.stack([np.load(f) for f in files]).mean(axis=0)

block_sizes = np.array(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])
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

P = np.array(P_new).T
W = np.array(W_new).T












channels = params.LAYER_NAME_TO_CHANNELS[layer_name]
num_relus = params.LAYER_NAME_TO_RELU_COUNT[layer_name]


dp = - np.inf * np.ones(shape=(num_relus, ))
dp[W] = P[0]
assert np.max(block_sizes) <= 255
dp_arg = np.zeros(shape=(channels, num_relus), dtype=np.uint8)
dp_arg[0, W] = np.arange(W.shape[0])


arange = np.arange(num_relus)
all_indices = np.maximum(np.arange(num_relus)[:, np.newaxis] - W, 0)
orig_shape = all_indices.shape
all_indices = all_indices.flatten()

for channel in tqdm(range(1, channels)):

    r = dp[all_indices].reshape(orig_shape) + P[channel]
    dp_arg[channel] = np.argmax(r, axis=1)
    dp = r[arange, dp_arg[channel]]



index_to_block_sizes = np.array(index_to_block_sizes)



min_relus = int((layer_dim/64)**2*channels)
max_relus = int((layer_dim/1)**2*channels)
relus_count = np.arange(min_relus, max_relus)

all_block_sizes = []
for relu_count in tqdm(relus_count):
    block_sizes = []
    for channel in reversed(range(channels)):

        arg = dp_arg[channel, relu_count]

        channel_num_relus = W[arg]
        relu_count -= channel_num_relus
        block_sizes.append(index_to_block_sizes[arg, channel])
    all_block_sizes.append(block_sizes[::-1])

all_block_sizes = np.array(all_block_sizes)