from research.block_relu.params import MobileNetV2Params
import numpy as np
import os
import pickle
from collections import defaultdict
import torch
from multiprocessing import Pool
from multiprocessing import sharedctypes
from tqdm import tqdm

params = MobileNetV2Params()

layers = ['conv1', 'layer1_0_0', 'layer2_0_0', 'layer2_0_1', 'layer2_1_0', 'layer2_1_1', 'layer3_0_0', 'layer3_0_1',
          'layer3_1_0', 'layer3_1_1', 'layer3_2_0', 'layer3_2_1', 'layer4_0_0', 'layer4_0_1', 'layer4_1_0',
          'layer4_1_1', 'layer4_2_0', 'layer4_2_1', 'layer4_3_0', 'layer4_3_1']
block_sizes = np.array([[1, 1]] + [
    [1, 2],
    [2, 1],
    [2, 2],
    [2, 4],
    [4, 2],
    [3, 3],
    [4, 4],
    [3, 6],
    [6, 3],
    [5, 5],
    [4, 8],
    [8, 4],
    [6, 6],
    [7, 7],
    [5, 10],
    [10, 5],
    [8, 8],
    [6, 12],
    [12, 6],
    [9, 9],
    [7, 14],
    [14, 7],
    [10, 10],
    [11, 11],
    [8, 16],
    [16, 8],
    [12, 12],
    [13, 13],
    [14, 14],
    [15, 15],
    [16, 16],
    [64, 64], [7, 3], [6, 9], [16, 6], [1, 6], [3, 7], [2, 5], [8, 5], [5, 8], [10, 8], [6, 10],
    [4, 10], [3, 2], [2, 6], [8, 2], [4, 5], [9, 3], [4, 16], [3, 12], [8, 12], [3, 1],
    [10, 14], [14, 8], [6, 16], [12, 8], [32, 32], [4, 12], [2, 12], [5, 1], [7, 2], [12, 2],
    [16, 10], [1, 5], [8, 6], [4, 1], [6, 4], [5, 4], [10, 4], [16, 4], [15, 10], [3, 5],
    [2, 7], [8, 3], [4, 6], [6, 1], [7, 4], [14, 12], [12, 4], [3, 15], [1, 3], [2, 8],
    [10, 15], [6, 2], [12, 9], [8, 10], [12, 3], [14, 10], [12, 14], [15, 3], [1, 4], [3, 9],
    [2, 3], [9, 6], [6, 5], [5, 3], [6, 8], [10, 16], [3, 4], [9, 12], [4, 7], [5, 6],
    [10, 6], [16, 2], [12, 5], [12, 16], [15, 5], [8, 14], [5, 12], [2, 16], [2, 10], [5, 15],
    [16, 12], [3, 8], [4, 3], [5, 2], [10, 2]])
dims = [16, 32, 64, 128, 256, 512]
num_relus_dict = {dim: [] for dim in dims}

for block_size_index, block_size in enumerate(block_sizes):
    for dim in dims:
        avg_pool = torch.nn.AvgPool2d(
            kernel_size=tuple(block_size),
            stride=tuple(block_size), ceil_mode=True)

        cur_input = torch.zeros(size=(1, 1, dim, dim))
        cur_relu_map = avg_pool(cur_input)
        num_relus = cur_relu_map.shape[2] * cur_relu_map.shape[3]
        num_relus_dict[dim].append(num_relus)

block_size_groups = defaultdict(list)
for block_size_index, block_size in enumerate(block_sizes):
    block_size_groups[block_size[0] * block_size[1]].append(block_size_index)

all_relus = []

for block_index in range(len(block_sizes)):

    relus = []
    for layer_name in layers:
        layer_dim = int(
            np.sqrt(params.LAYER_NAME_TO_RELU_COUNT[layer_name] // params.LAYER_NAME_TO_CHANNELS[layer_name]))
        N_channels = params.LAYER_NAME_TO_CHANNELS[layer_name]
        channel_relus = num_relus_dict[layer_dim][block_index]
        relus.extend([channel_relus] * N_channels)
    all_relus.append(relus)
W = np.array(all_relus)
# TODO: make W accurate

out_dir = f"/home/yakir/distortion_approximation_v2/extract_stats_v2/"
batch_index = 0
all_noises = [[0] * W.shape[1]]
for block_index, block_size in enumerate(block_sizes[1:]):
    noises = []
    for layer_name in layers:
        file_name = os.path.join(out_dir, f"{layer_name}_{batch_index}_{block_size[0]}_{block_size[1]}.pickle")
        content = pickle.load(open(file_name, 'rb'))
        noises.extend(list(np.array([x["noises_distorted"]['layer4_3'] for x in content]).mean(axis=1)))
    all_noises.append(noises)

P = -np.array(all_noises)

P_new = []
W_new = []
index_to_block_sizes = []
for k, v in block_size_groups.items():

    cur_block_sizes = block_sizes[v]

    t = np.stack([P[row] for row in v])
    index_to_block_sizes.append(cur_block_sizes[t.argmax(axis=0)])
    P_new.append(t.max(axis=0))
    W_new.append(W[v[0]])

P = np.array(P_new).T
W = np.array(W_new).T
index_to_block_sizes = np.array(index_to_block_sizes)

np.save('/home/yakir/Data2/DP/W.npy', W)

CHANNEL_BATCH_SIZE = 100
MAX_RELU = 4000000
NUM_PROC = 8
RELU_BATCH = MAX_RELU // NUM_PROC

data = - np.inf * np.ones(shape=(CHANNEL_BATCH_SIZE, MAX_RELU))
data[0, W[0, :]] = P[0, :]

data_arg = np.zeros(shape=(CHANNEL_BATCH_SIZE, MAX_RELU), dtype=np.int32)
data_arg[0, W[0, :]] = np.arange(W[0, :].shape[0])

dp = np.ctypeslib.as_ctypes(data)
dp_arg = np.ctypeslib.as_ctypes(data_arg)

shared_dp = sharedctypes.RawArray(dp._type_, dp)
shared_dp_arg = sharedctypes.RawArray(dp_arg._type_, dp_arg)


def fill_up(indices):
    i, j_batch = indices
    i = i % CHANNEL_BATCH_SIZE
    j_index_start = j_batch * RELU_BATCH
    j_index_end = j_index_start + RELU_BATCH

    shared_memory = np.ctypeslib.as_array(shared_dp)
    shared_memory_arg = np.ctypeslib.as_array(shared_dp_arg)

    for j in range(j_index_start, j_index_end):
        indices = np.maximum((j - W[i]), 0)
        shared_memory[i, j] = (shared_memory[i - 1, indices] + P[i]).max()
        shared_memory_arg[i, j] = (shared_memory[i - 1, indices] + P[i]).argmax()

# for i in tqdm(range(1,100)):
#     fill_up((i, 0))

np.save('/home/yakir/Data2/DP/index_to_block_sizes.npy', index_to_block_sizes)

with Pool(processes=NUM_PROC) as p:

    for i in tqdm(range(1, P.shape[0])):
        if i % CHANNEL_BATCH_SIZE == 0:

            np.save(f"/home/yakir/Data2/DP/dp_{i}.npy", np.ctypeslib.as_array(shared_dp))
            np.save(f"/home/yakir/Data2/DP/dp_arg_{i}.npy", np.ctypeslib.as_array(shared_dp_arg))

        for _, _ in enumerate(p.imap_unordered(fill_up, [(i, x) for x in range(NUM_PROC)])):
            pass

dp = np.ctypeslib.as_array(shared_dp)
dp_arg = np.ctypeslib.as_array(dp_arg)
np.save(f"/home/yakir/Data2/DP/dp_{i}.npy", dp)
np.save(f"/home/yakir/Data2/DP/dp_arg_{i}.npy", dp_arg)







# layer_name_to_relu_count = [params.LAYER_NAME_TO_RELU_COUNT[layer_name] for layer_name in layers]
# layer_name_to_channel_count = [params.LAYER_NAME_TO_CHANNELS[layer_name] for layer_name in layers]

# dp = - np.inf * np.ones(shape=(100, 4000000))
# dp[0, W[0, :]] = P[0, :]
#
# for i in tqdm(range(1, 100)):
#
#     assert dp[i-1,0] == -np.inf
#     for j in range(dp[i].shape[0]):
#         indices = np.maximum((j - W[i]), 0)
#         dp[i, j] = (dp[i-1, indices] + P[i]).max()
#