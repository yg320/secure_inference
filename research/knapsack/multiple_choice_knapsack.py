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

def get_matrix_data(channel_distortion_path, params):
    Ps = []
    Ws = []
    layer_name_to_block_shape_index_and_channel_to_block_size = {}
    for layer_name in layer_names:
        block_sizes = np.array(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])
        assert np.max(block_sizes) < 255
        layer_dim = params.LAYER_NAME_TO_DIMS[layer_name][1]

        W = get_block_index_to_num_relus(block_sizes, layer_dim)

        files = glob.glob(os.path.join(channel_distortion_path, f"{layer_name}_*.pickle"))
        assert len(files) == 2
        noise = np.stack([pickle.load(open(f, 'rb'))["Noise"] for f in files])
        noise = noise.mean(axis=0).mean(axis=2).T  # noise.shape = [N-block-sizes, N-channels]

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
    channel_and_arg_to_block_size = arg_and_channel_to_block_size.transpose([1, 0, 2])
    Ps = np.concatenate(Ps, axis=0)
    Ws = np.concatenate([np.repeat(Ws[layer_index][:,np.newaxis], params.LAYER_NAME_TO_DIMS[layer_name][0], axis=1) for layer_index, layer_name in enumerate(layer_names)], axis=1).T
    return Ps, Ws, channel_and_arg_to_block_size


def main_dp_not_efficient(Ws, Ps, channels, num_relus):

    dp_arg = 255 * np.ones(shape=(channels, num_relus), dtype=np.uint8)
    # dp_arg = np.zeros(shape=(channels, num_relus), dtype=np.uint8)
    dp_arg[0, Ws[0]] = np.arange(Ws[0].shape[0])

    dp = - np.inf * np.ones(shape=(num_relus,))
    dp[Ws[0]] = Ps[0]

    for channel in tqdm(range(1, channels)):
        gc.collect()
        dp_prev = dp.copy()
        for desired_relu_count in range(num_relus):
            indices = np.maximum((desired_relu_count - Ws[channel]), 0)
            dp[desired_relu_count] = (dp_prev[indices] + Ps[channel]).max()
            dp_arg[channel, desired_relu_count] = (dp_prev[indices] + Ps[channel]).argmax()
        assert dp[0] == -np.inf
    return dp_arg

def main_dp(Ws, Ps, channels, num_relus):

    assert num_relus < np.iinfo(np.int32).max
    arange = np.arange(num_relus, dtype=np.int32)[:, np.newaxis]
    indices = np.zeros(shape=(num_relus, Ws[0].shape[0]), dtype=np.int32)
    buffer = np.zeros(shape=(num_relus * Ws[0].shape[0], ), dtype=np.float64)
    # dp_arg = 255 * np.ones(shape=(channels, num_relus), dtype=np.uint8)
    dp_arg = np.zeros(shape=(channels, num_relus), dtype=np.uint8)
    dp = - np.inf * np.ones(shape=(num_relus,))

    buffer_orig_shape = buffer.shape
    indices_orig_shape = indices.shape
    dp_arg[0, Ws[0]] = np.arange(Ws[0].shape[0])
    dp[Ws[0]] = Ps[0]

    for channel in tqdm(range(1, channels)):
        gc.collect()
        np.subtract(arange, Ws[channel], out=indices)
        np.maximum(indices, 0, out=indices)
        indices = indices.reshape(-1)
        np.take(dp, indices, out=buffer)
        indices = indices.reshape(indices_orig_shape)
        buffer = buffer.reshape(indices_orig_shape)
        np.add(buffer, Ps[channel], out=buffer)

        np.argmax(buffer, axis=1, out=dp_arg[channel])
        dp = buffer[arange[:, 0], dp_arg[channel]]
        buffer = buffer.reshape(buffer_orig_shape)  # Consider: buffer.shape = buffer_orig_shape to avoid rare case of copying

    return dp_arg


def convert_dp_arg_to_block_size_spec(dp_arg, Ws, arg_and_channel_order_to_block_size):
    relu_count = dp_arg.shape[1] - 1
    num_channels = dp_arg.shape[0]
    block_sizes = []
    for channel_order in reversed(range(num_channels)):
        arg = dp_arg[channel_order, relu_count]
        if arg == 255:
            print('j')
        channel_num_relus = Ws[channel_order, arg]
        relu_count -= channel_num_relus
        block_sizes.append(arg_and_channel_order_to_block_size[channel_order, arg])
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

    Ps, Ws, channel_and_arg_to_block_size = get_matrix_data(channel_distortion_path, params)

    # dp_arg_0 = main_dp_not_efficient(Ws, Ps, 100, num_relus)
    # dp_arg_1 = main_dp(Ws, Ps, 100, num_relus)
    dp_arg = main_dp(Ws, Ps, num_channels, num_relus)



    layer_name_to_block_size = convert_dp_arg_to_block_size_spec(dp_arg, Ws, channel_and_arg_to_block_size)
    return layer_name_to_block_size, dp_arg






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--block_size_spec_file_name', type=str, default=f"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_0.0625/block_spec.pickle")
    parser.add_argument('--output_path', type=str, default="/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_0.0625/channel_distortions")
    parser.add_argument('--ratio', type=float, default=0.0625)
    parser.add_argument('--params_name', type=str, default="MobileNetV2_256_Params_2_Groups")
    args = parser.parse_args()
    # import time
    # assert time.time() <= 1667812589.9087472 + 1800
    params = ParamsFactory()(args.params_name)

    channel_distortion_path = args.output_path
    layer_names = params.LAYER_GROUPS[args.iter]#[1:]
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
