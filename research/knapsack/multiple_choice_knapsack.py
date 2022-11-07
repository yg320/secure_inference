import glob
import os.path
import time

import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import shutil

from research.parameters.base import ParamsFactory
import gc


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

def get_matrix_data(channel_distortion_path):
    Ps = []
    Ws = []
    layer_name_to_block_shape_index_and_channel_to_block_size = {}
    for layer_name in layer_names:
        block_sizes = np.array(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])
        assert np.max(block_sizes) <= 255
        layer_dim = params.LAYER_NAME_TO_DIMS[layer_name][1]

        W = get_block_index_to_num_relus(block_sizes, layer_dim)

        files = glob.glob(os.path.join(channel_distortion_path, f"{layer_name}_*.pickle"))
        assert len(files) == 2
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
    arg_and_channel_to_block_size = np.concatenate([layer_name_to_block_shape_index_and_channel_to_block_size[layer_name] for layer_name in layer_names], axis=1)

    return Ps, Ws, arg_and_channel_to_block_size

def main_dp(Ws, Ps, channels, num_relus):

    arange = np.arange(num_relus)[:, np.newaxis]

    dp = - np.inf * np.ones(shape=(num_relus,))
    dp[Ws[0]] = Ps[0][0]

    dp_arg = np.zeros(shape=(channels, num_relus), dtype=np.uint8)
    dp_arg[0, Ws[0]] = np.arange(Ws[0].shape[0])

    channels_batch = [params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names]

    layer_index = 0
    all_indices = np.maximum(arange - Ws[layer_index], 0)
    orig_shape = all_indices.shape
    all_indices = all_indices.reshape(-1)
    cur_P = Ps[layer_index]
    offset = 0

    buffer = np.zeros(shape=orig_shape, dtype=np.float64)

    for channel in tqdm(range(1, channels)):
        gc.collect()
        if channel == sum(channels_batch[:layer_index+1]):
            layer_index += 1
            all_indices = all_indices.reshape(orig_shape)
            all_indices[:] = np.maximum(arange - Ws[layer_index], 0)
            all_indices = all_indices.reshape(-1)
            cur_P = Ps[layer_index]
            offset = sum(channels_batch[:layer_index])

        buffer = buffer.reshape(-1)
        buffer[:] = dp[all_indices]
        buffer = buffer.reshape(orig_shape)
        buffer[:] = buffer + cur_P[channel - offset]

        dp_arg[channel] = np.argmax(buffer, axis=1)
        dp = buffer[arange[:, 0], dp_arg[channel]]

    return dp_arg

def convert_dp_arg_to_block_size_spec(dp_arg, Ws, arg_and_channel_order_to_block_size):
    channel_order_to_layer_index = np.hstack([[layer_index] * params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_index, layer_name in enumerate(layer_names)])
    relu_count = dp_arg.shape[1] - 1
    num_channels = dp_arg.shape[0]
    block_sizes = []
    for channel_order in reversed(range(num_channels)):
        arg = dp_arg[channel_order, relu_count]
        channel_num_relus = Ws[channel_order_to_layer_index[channel_order]][arg]
        relu_count -= channel_num_relus
        block_sizes.append(arg_and_channel_order_to_block_size[arg, channel_order])
    block_sizes = np.array(block_sizes[::-1])


    channel_order = 0
    layer_name_to_block_size = dict()
    for layer_name in layer_names:
        layer_name_to_block_size[layer_name] = block_sizes[channel_order:channel_order + params.LAYER_NAME_TO_DIMS[layer_name][0], :]
        channel_order += params.LAYER_NAME_TO_DIMS[layer_name][0]

    return layer_name_to_block_size


def get_block_size_spec(layer_names, params, channel_distortion_path, ratio):

    num_channels = sum(params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names)
    num_relus = int(sum(np.prod(params.LAYER_NAME_TO_DIMS[layer_name]) for layer_name in layer_names) * ratio)
    print(num_relus)
    Ps, Ws, arg_and_channel_order_to_block_size = get_matrix_data(channel_distortion_path)
    dp_arg = main_dp(Ws, Ps, num_channels, num_relus)

    layer_name_to_block_size = convert_dp_arg_to_block_size_spec(dp_arg, Ws, arg_and_channel_order_to_block_size)
    return layer_name_to_block_size, dp_arg






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default="ade_20k_256x256")
    parser.add_argument('--config', type=str, default="/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py")
    parser.add_argument('--checkpoint', type=str, default="/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth")
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--block_size_spec_file_name', type=str, default=f"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_0.0625/block_spec.pickle")
    parser.add_argument('--output_path', type=str, default="/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_0.0625/channel_distortions")
    parser.add_argument('--ratio', type=float, default=0.0625)
    parser.add_argument('--params_name', type=str, default="MobileNetV2_256_Params_2_Groups")
    args = parser.parse_args()

    params = ParamsFactory()(args.params_name)

    params.DATASET = args.dataset
    params.CONFIG = args.config
    params.CHECKPOINT = args.checkpoint

    channel_distortion_path = args.output_path
    layer_names = params.LAYER_GROUPS[args.iter]
    ratio = args.ratio
    block_size_spec_file_name = args.block_size_spec_file_name
    assert os.path.exists(block_size_spec_file_name) or args.iter == 0
    layer_name_to_block_size, dp_arg = get_block_size_spec(layer_names, params, channel_distortion_path, ratio)

    if os.path.exists(block_size_spec_file_name):
        older_block_size_spec = pickle.load(file=open(block_size_spec_file_name, 'rb'))
        shutil.copyfile(block_size_spec_file_name, block_size_spec_file_name + f".{int(time.time())}")
    else:
        older_block_size_spec = dict()

    new_block_size_spec = {**layer_name_to_block_size, **older_block_size_spec}
    pickle.dump(obj=new_block_size_spec, file=open(block_size_spec_file_name, 'wb'))
