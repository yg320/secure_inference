import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
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
from research.distortion.distortion_utils import get_num_relus, get_brelu_bandwidth
from research.parameters.base import ParamsFactory
# from research.share_array import make_shared_array, get_shared_array
import gc
import time
import torch


def get_cost(block_size, activation_dim, cost_type, division=1):

    if cost_type == "Bandwidth":
        cost_func = get_brelu_bandwidth
    elif cost_type == "ReLU":
        cost_func = get_num_relus
    else:
        cost_func = None

    cost = cost_func(tuple(block_size), activation_dim) // division

    return cost




def get_matrix_data(channel_distortion_path, params, cost_type, division):
    Ps = []
    Ws = []
    layer_name_to_block_shape_index_and_channel_to_block_size = {}
    # TODO: replace the 56
    for layer_name in layer_names:
        block_sizes = np.array(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])
        assert np.max(block_sizes) < 255
        layer_dim = params.LAYER_NAME_TO_DIMS[layer_name][1]

        W = np.array([get_cost(tuple(block_size), layer_dim, cost_type, division) for block_size in block_sizes])

        files = glob.glob(os.path.join(channel_distortion_path, f"{layer_name}_*.pickle"))
        assert len(files) == 2
        noise = np.stack([pickle.load(open(f, 'rb'))["Noise"] for f in files])
        noise = noise.mean(axis=0).mean(axis=2).T  # noise.shape = [N-block-sizes, N-channels]

        P = -np.array(noise)

        block_size_groups = defaultdict(list)
        for block_size_index, block_size in enumerate(block_sizes):
            cur_cost = get_cost(tuple(block_size), layer_dim, cost_type, division) #1 here to avoid weird stuff
            block_size_groups[cur_cost].append(block_size_index)

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

        block_shape_index_and_channel_to_block_size = np.array(block_shape_index_and_channel_to_block_size)
        P = np.array(P_new).T
        W = np.array(W_new).T

        pad_W = np.zeros(shape=(56 - W.shape[0],), dtype=W.dtype)
        pad_P = -np.inf * np.ones(shape=(P.shape[0], 56 - P.shape[1]), dtype=P.dtype)
        shape = block_shape_index_and_channel_to_block_size.shape
        pad_block_sizes_tracker = np.zeros(shape=(56 - shape[0], shape[1], shape[2]), dtype=block_shape_index_and_channel_to_block_size.dtype)

        P = np.concatenate([P, pad_P], axis=1)
        W = np.concatenate([W, pad_W], axis=0)
        block_shape_index_and_channel_to_block_size = np.concatenate([block_shape_index_and_channel_to_block_size, pad_block_sizes_tracker], axis=0)
        layer_name_to_block_shape_index_and_channel_to_block_size[layer_name] = block_shape_index_and_channel_to_block_size
        Ps.append(P)
        Ws.append(W)
    arg_and_channel_to_block_size = np.concatenate([layer_name_to_block_shape_index_and_channel_to_block_size[layer_name] for layer_name in layer_names], axis=1)
    channel_and_arg_to_block_size = arg_and_channel_to_block_size.transpose([1, 0, 2])
    Ps = np.concatenate(Ps, axis=0)
    Ws = np.concatenate([np.repeat(Ws[layer_index][:,np.newaxis], params.LAYER_NAME_TO_DIMS[layer_name][0], axis=1) for layer_index, layer_name in enumerate(layer_names)], axis=1).T
    return Ps, Ws, channel_and_arg_to_block_size

def main_dp_not_efficient(Ws, Ps, channels, num_relus):

    dp_arg = 255 * np.ones(shape=(channels, num_relus), dtype=np.uint8)
    dp_arg[0, Ws[0]] = np.arange(Ws[0].shape[0])

    # dp[-1] should always hold -inf, so in line # indices = np.maximum((desired_relu_count - Ws[channel]), -1) we can be sure that bad indices will get -inf
    dp = - np.inf * np.ones(shape=(num_relus + 1,))

    dp[Ws[0]] = Ps[0]

    for channel in tqdm(range(1, channels)):
        gc.collect()
        dp_prev = dp.copy()
        for desired_relu_count in range(num_relus):
            indices = np.maximum((desired_relu_count - Ws[channel]), -1)
            dp[desired_relu_count] = (dp_prev[indices] + Ps[channel]).max()
            if np.any((dp_prev[indices] + Ps[channel]) > -np.inf):
                dp_arg[channel, desired_relu_count] = (dp_prev[indices] + Ps[channel]).argmax()

        dp[-1] = -np.inf
    return dp_arg, dp[:-1]

def main_dp_super_not_efficient(Ws, Ps, channels, num_relus):

    dp_arg = 255 * np.ones(shape=(channels, num_relus), dtype=np.uint8)
    dp_arg[0, Ws[0]] = np.arange(Ws[0].shape[0])

    dp = - np.inf * np.ones(shape=(num_relus,))
    dp[Ws[0]] = Ps[0]

    for channel in range(1, channels):
        gc.collect()
        dp_prev = dp.copy()
        for desired_relu_count in tqdm(range(num_relus)):

            max_val = -np.inf
            argmax = None

            # Go over the cost (num relus) for each block configuration of channel
            for cur_block_size_index, cur_num_relus in enumerate(Ws[channel]):

                # If you use desired_relu_count amount of relus, and current configuration cost cur_num_relus, then we
                # should examine the previous channel cost of desired_relu_count - cur_num_relus (i.e. dp_prev[desired_relu_count - cur_num_relus] )
                # Obviously, we should add  Ps[channel][cur_block_size_index] to the cost and pick the best configuration
                index = desired_relu_count - cur_num_relus
                if index >= 0:
                    cur_v = dp_prev[index] + Ps[channel][cur_block_size_index]
                    if cur_v > max_val:
                        max_val = cur_v
                        argmax = cur_block_size_index

            if argmax is not None:
                dp[desired_relu_count] = max_val
                dp_arg[channel, desired_relu_count] = argmax

            # indices = np.maximum((desired_relu_count - Ws[channel]), -1)
            # dp[desired_relu_count] = (dp_prev[indices] + Ps[channel]).max()
            # dp_arg[channel, desired_relu_count] = (dp_prev[indices] + Ps[channel]).argmax()

    return dp_arg, dp

# #
class IO_Buffer:
    def __init__(self, word_size, package="numpy", load=False):
        self.buffer_size = 10
        self.buffer_dir = f"/home/yakir/Data2/buffer_dir/{package}"

        self.word_size = word_size
        self.buffer_init_value = 255
        self.cur_frame = 0
        self.dirty = False
        self.package = package

        if self.package == "numpy":
            self.buffer_path_format = os.path.join(self.buffer_dir, "{}.npy")
            self.buffer = self.buffer_init_value * np.ones(shape=(self.buffer_size, self.word_size), dtype=np.uint8)
        elif self.package == "torch":
            self.buffer_path_format = os.path.join(self.buffer_dir, "{}.pt")
            self.buffer = self.buffer_init_value * torch.ones(size=(self.buffer_size, self.word_size), dtype=torch.uint8)

        if not load:
            if os.path.exists(self.buffer_dir):
                shutil.rmtree(self.buffer_dir)

            os.makedirs(self.buffer_dir)
        else:
            self.reload(0)

    def get_channel_frame(self, channel):
        return channel // self.buffer_size

    def reload(self, channel_frame):
        if self.dirty:
            if self.package == "numpy":
                np.save(file=self.buffer_path_format.format(self.cur_frame), arr=self.buffer)
            elif self.package == "torch":
                torch.save(f=self.buffer_path_format.format(self.cur_frame), obj=self.buffer)
            self.dirty = False

        if os.path.exists(self.buffer_path_format.format(channel_frame)):
            if self.package == "numpy":
                self.buffer = np.load(self.buffer_path_format.format(channel_frame))
            elif self.package == "torch":
                self.buffer = torch.load(self.buffer_path_format.format(channel_frame))
        else:
            self.buffer[:] = self.buffer_init_value
        self.cur_frame = channel_frame

    def flush(self):
        self.reload(self.cur_frame)

    def __setitem__(self, channel, value):
        channel_frame = self.get_channel_frame(channel)
        if channel_frame != self.cur_frame:
            self.reload(channel_frame)
        self.buffer[channel % self.buffer_size] = value
        self.dirty = True

    def __getitem__(self, channel):
        channel_frame = self.get_channel_frame(channel)
        if channel_frame != self.cur_frame:
            self.reload(channel_frame)
        return self.buffer[channel % self.buffer_size]


def main_dp(Ws, Ps, channels, num_relus):

    assert num_relus < np.iinfo(np.int32).max
    arange = np.arange(num_relus, dtype=np.int32)[:, np.newaxis]
    indices = np.zeros(shape=(num_relus, Ws[0].shape[0]), dtype=np.int32)
    buffer = np.zeros(shape=(num_relus * Ws[0].shape[0], ), dtype=np.float64)

    dp_arg = IO_Buffer(num_relus)
    dp = - np.inf * np.ones(shape=(num_relus + 1,))

    buffer_orig_shape = buffer.shape
    indices_orig_shape = indices.shape
    init_row = np.copy(dp_arg[0])
    init_row[Ws[0]] = np.arange(Ws[0].shape[0])
    dp_arg[0] = init_row
    dp[Ws[0]] = Ps[0]

    for channel in tqdm(range(1, channels)):
        gc.collect()

        np.subtract(arange, Ws[channel], out=indices)
        np.maximum(indices, -1, out=indices)
        indices = indices.reshape(-1)
        np.take(dp, indices, out=buffer)
        indices = indices.reshape(indices_orig_shape)
        buffer = buffer.reshape(indices_orig_shape)
        np.add(buffer, Ps[channel], out=buffer)

        dp_arg[channel] = np.argmax(buffer, axis=1)
        dp[:-1] = buffer[arange[:, 0], dp_arg[channel]]
        dp_arg[channel][np.all(buffer == -np.inf, axis=1)] = 255
        buffer = buffer.reshape(buffer_orig_shape)  # Consider: buffer.shape = buffer_orig_shape to avoid rare case of copying

    dp_arg.flush()
    return dp_arg, dp[:-1]


def main_dp_torch(Ws, Ps, channels, num_relus):
    Ws = torch.from_numpy(Ws)
    Ps = torch.from_numpy(Ps)

    assert num_relus < np.iinfo(np.int32).max
    arange = torch.arange(num_relus, dtype=torch.int64).unsqueeze(dim=1)
    indices = torch.zeros(size=(num_relus, Ws[0].shape[0]), dtype=torch.int64)
    buffer = torch.zeros(size=(num_relus * Ws[0].shape[0], ), dtype=torch.float64)

    dp_arg = IO_Buffer(num_relus, package="torch")
    dp = - float("Inf") * torch.ones(size=(num_relus + 1,), dtype=torch.float64)

    buffer_orig_shape = buffer.shape
    indices_orig_shape = indices.shape
    init_row = dp_arg[0].clone()
    init_row[Ws[0]] = torch.arange(Ws[0].shape[0], dtype=torch.uint8)
    dp_arg[0] = init_row
    dp[Ws[0]] = Ps[0]

    negative_one = -torch.ones(size=(1,), dtype=torch.int64)

    device = torch.device("cuda:1")
    Ws = Ws.to(device)  # (torch.Size([14272, 56]), torch.int64)                6.39M
    Ps = Ps.to(device)  # (torch.Size([14272, 56]), torch.float64)              6.39M
    arange = arange.to(device)  # (torch.Size([15656345, 1]), torch.int64)      125.25M
    indices = indices.to(device)  # (torch.Size([15656345, 56]), torch.int64)
    negative_one = negative_one.to(device)
    dp = dp.to(device)  # (torch.Size([15656346]), torch.float64)
    buffer = buffer.to(device)  # (torch.Size([876755320]), torch.float64)
    dp_arg.buffer = dp_arg.buffer.to(device)  # (torch.Size([10, 15656345]), torch.uint8)

    for channel in tqdm(range(1, channels)):
        gc.collect()

        torch.sub(arange, Ws[channel], out=indices)
        torch.max(indices, negative_one, out=indices)
        indices = indices.reshape(-1)
        torch.take(dp, indices, out=buffer)
        indices = indices.reshape(indices_orig_shape)
        buffer = buffer.reshape(indices_orig_shape)
        torch.add(buffer, Ps[channel], out=buffer)

        dp_arg[channel] = torch.argmax(buffer, dim=1)
        dp[:-1] = buffer[arange[:, 0], dp_arg[channel].to(torch.int64)]
        dp_arg[channel][torch.all(buffer == -float("Inf"), dim=1)] = 255
        buffer = buffer.reshape(buffer_orig_shape)  # Consider: buffer.shape = buffer_orig_shape to avoid rare case of copying

    dp_arg.flush()
    return dp_arg, dp[:-1]

# def worker_function(s, e, channel, indices_orig_shape):
#
#     Ws = get_shared_array('Ws')
#     Ps = get_shared_array('Ps')
#     dp = get_shared_array('dp')
#     arange = get_shared_array('arange')[s:e]
#     indices = get_shared_array('indices')[s:e]
#     buffer = get_shared_array('buffer')
#
#
#     np.subtract(arange, Ws[channel], out=indices)
#     np.maximum(indices, -1, out=indices)
#     indices = indices.reshape(-1)
#     np.take(dp, indices, out=buffer)
#     indices = indices.reshape(indices_orig_shape)
#     buffer = buffer.reshape(indices_orig_shape)
#     np.add(buffer, Ps[channel], out=buffer)
#
#
# def main_dp_multiprocess(Ws, Ps, channels, num_relus):
#     assert num_relus < np.iinfo(np.int32).max
#     arange = np.arange(num_relus, dtype=np.int32)[:, np.newaxis]
#     indices = np.zeros(shape=(num_relus, Ws[0].shape[0]), dtype=np.int32)
#     buffer = np.zeros(shape=(num_relus * Ws[0].shape[0], ), dtype=np.float64)
#
#     dp_arg = IO_Buffer(num_relus)
#     dp = - np.inf * np.ones(shape=(num_relus + 1,))
#
#     buffer_orig_shape = buffer.shape
#     indices_orig_shape = indices.shape
#     init_row = np.copy(dp_arg[0])
#     init_row[Ws[0]] = np.arange(Ws[0].shape[0])
#     dp_arg[0] = init_row
#     dp[Ws[0]] = Ps[0]
#
#     make_shared_array(Ws, name='Ws')
#     make_shared_array(Ps, name='Ps')
#     make_shared_array(dp, name='dp')
#     make_shared_array(arange, name='arange')
#
#     make_shared_array(indices, name='indices')
#     make_shared_array(buffer, name='buffer')
#
#     indices = get_shared_array('indices')
#     buffer = get_shared_array('buffer')
#
#     for channel in tqdm(range(1, channels)):
#         gc.collect()
#
#         np.subtract(arange, Ws[channel], out=indices)
#         np.maximum(indices, -1, out=indices)
#         indices = indices.reshape(-1)
#         np.take(dp, indices, out=buffer)
#         indices = indices.reshape(indices_orig_shape)
#         buffer = buffer.reshape(indices_orig_shape)
#         np.add(buffer, Ps[channel], out=buffer)
#
#         dp_arg[channel] = np.argmax(buffer, axis=1)
#         dp[:-1] = buffer[arange[:, 0], dp_arg[channel]]
#         dp_arg[channel][np.all(buffer == -np.inf, axis=1)] = 255
#         buffer = buffer.reshape(buffer_orig_shape)  # Consider: buffer.shape = buffer_orig_shape to avoid rare case of copying
#
#     dp_arg.flush()
#     return dp_arg, dp[:-1]

def convert_dp_arg_to_block_size_spec(dp_arg, Ws, arg_and_channel_order_to_block_size, relu_count):

    num_channels = Ws.shape[0]
    block_sizes = []
    relu_count = int(torch.nonzero(dp_arg[num_channels-1] != 255).max().cpu().numpy())
    for channel_order in tqdm(reversed(range(num_channels))):
        arg = dp_arg[channel_order][relu_count]
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


def get_block_size_spec(layer_names, params, channel_distortion_path, ratio, cost_type, division):

    num_channels = sum(params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names)

    max_cost = int(sum(get_cost((1,1), params.LAYER_NAME_TO_DIMS[layer_name][1], cost_type, division) * params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names) * ratio)
    print(max_cost)
    Ps, Ws, channel_and_arg_to_block_size = get_matrix_data(channel_distortion_path, params, cost_type, division)

    # dp_arg_2, dp_2 = main_dp(Ws, Ps, 50, 100000)
    # dp_arg_0, dp_0 = main_dp_super_not_efficient(Ws, Ps, 50, 100000)
    # dp_arg_1, dp_1 = main_dp_not_efficient(Ws, Ps, 50, 100000)
    # assert np.all(dp_arg_2.buffer[:50] == dp_arg_0)
    # assert np.all(dp_2 == dp_0)
    # assert False
    # dp_arg_0, dp_0 = main_dp_torch(Ws, Ps, 6, max_cost)
    # dp_arg_1, dp_1 = main_dp(Ws, Ps, 6, max_cost)
    #
    # dp_arg_0 = dp_arg_0.buffer.cpu().numpy()
    # dp_arg_1 = dp_arg_1.buffer
    # l = np.argwhere(dp_arg_0[5] != dp_arg_1[5])[:,0]
    # dp_arg_0[5, [302592, 304356, 307968]]
    # dp_arg_1[5, [302592, 304356, 307968]]
    # print('ej')

    dp_arg, dp_0 = main_dp_torch(Ws, Ps, num_channels, max_cost)
    #
    # dp_arg = IO_Buffer(max_cost, package="torch", load=True)
    layer_name_to_block_size = convert_dp_arg_to_block_size_spec(dp_arg, Ws, channel_and_arg_to_block_size, max_cost - 1)
    return layer_name_to_block_size, dp_arg






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--block_size_spec_file_name', type=str, default=f"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity/block_spec.pickle")
    parser.add_argument('--output_path', type=str, default="/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity/channel_distortions")
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--params_name', type=str, default="MobileNetV2_256_Params_2_Groups")
    # parser.add_argument('--cost_type', type=str, default="ReLU")
    parser.add_argument('--cost_type', type=str, default="Bandwidth")
    parser.add_argument('--division', type=int, default=128)
    args = parser.parse_args()
    params = ParamsFactory()(args.params_name)

    channel_distortion_path = args.output_path
    layer_names = params.LAYER_GROUPS[args.iter]#[1:]
    ratio = args.ratio
    block_size_spec_file_name = args.block_size_spec_file_name
    assert os.path.exists(block_size_spec_file_name) or args.iter == 0
    layer_name_to_block_size, dp_arg = get_block_size_spec(layer_names, params, channel_distortion_path, ratio, args.cost_type, args.division)

    if os.path.exists(block_size_spec_file_name):
        older_block_size_spec = pickle.load(file=open(block_size_spec_file_name, 'rb'))
        shutil.copyfile(block_size_spec_file_name, block_size_spec_file_name + f".{int(time.time())}")
    else:
        older_block_size_spec = dict()

    new_block_size_spec = {**layer_name_to_block_size, **older_block_size_spec}
    pickle.dump(obj=new_block_size_spec, file=open(block_size_spec_file_name, 'wb'))
