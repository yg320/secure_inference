import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import glob
import os.path
import time
import mmcv
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import shutil
from research.distortion.distortion_utils import get_num_relus, get_brelu_bandwidth
from research.distortion.utils import get_channel_order_statistics
from research.distortion.parameters.factory import param_factory
from research.distortion.utils import get_channels_subset
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
    cost = cost_func(tuple(block_size), activation_dim)

    # 536 for decode_0
    assert (cost == 536 and cost_type == "Bandwidth") or cost % division == 0, cost
    cost = cost // division

    return cost

#
#
#
# def get_matrix_data(channel_distortion_path, params, cost_type, division):
#     Ps = []
#     Ws = []
#     block_size_trackers = []
#     # TODO: replace the 56
#     for layer_name in layer_names:
#         block_sizes = np.array(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])
#         if params.exclude_special_blocks:
#             assert np.prod(block_sizes[-1]) == np.prod(block_sizes[-2]) == 0
#             block_sizes = block_sizes[:-2]
#         else:
#             block_sizes = block_sizes[:-1]
#         assert np.max(block_sizes) < 255
#         layer_dim = params.LAYER_NAME_TO_DIMS[layer_name][1]
#
#         W = np.array([get_cost(tuple(block_size), layer_dim, cost_type, division) for block_size in block_sizes])
#         glob_pattern = os.path.join(channel_distortion_path, f"{layer_name}_*.pickle")
#         files = glob.glob(glob_pattern)
#         assert len(files) == 6, glob_pattern
#
#         noise = np.stack([pickle.load(open(f, 'rb'))["Noise"] for f in files])
#         noise = noise.mean(axis=0).mean(axis=2).T  # noise.shape = [N-block-sizes, N-channels]
#
#         signal = np.stack([pickle.load(open(f, 'rb'))["Signal"] for f in files])
#         signal = signal.mean(axis=0).mean(axis=2).T
#         if layer_name == "decode_0":
#             signal[:] = signal[0]
#         assert signal.min() > 0
#
#         noise = noise / signal
#         P = -np.array(noise)
#
#         block_size_groups = defaultdict(list)
#         for block_size_index, block_size in enumerate(block_sizes):
#             cur_cost = get_cost(tuple(block_size), layer_dim, cost_type, division) #1 here to avoid weird stuff
#             block_size_groups[cur_cost].append(block_size_index)
#
#         P_new = []
#         W_new = []
#         cur_block_size_tracker = []
#         for k, v in block_size_groups.items():
#             cur_block_sizes = block_sizes[v]
#
#             P_same_weight = np.stack([P[row_index] for row_index in v])
#             argmax = P_same_weight.argmax(axis=0)
#             max_ = P_same_weight.max(axis=0)
#
#             cur_block_size_tracker.append(cur_block_sizes[argmax])
#             P_new.append(max_)
#             W_new.append(W[v[0]])
#
#         cur_block_size_tracker = np.array(cur_block_size_tracker)
#         P = np.array(P_new).T
#         W = np.array(W_new).T
#
#         pad_W = np.zeros(shape=(56 - W.shape[0],), dtype=W.dtype)
#         pad_P = -np.inf * np.ones(shape=(P.shape[0], 56 - P.shape[1]), dtype=P.dtype)
#         shape = cur_block_size_tracker.shape
#         pad_block_sizes_tracker = np.zeros(shape=(56 - shape[0], shape[1], shape[2]), dtype=cur_block_size_tracker.dtype)
#
#         P = np.concatenate([P, pad_P], axis=1)
#         W = np.concatenate([W, pad_W], axis=0)
#         cur_block_size_tracker = np.concatenate([cur_block_size_tracker, pad_block_sizes_tracker], axis=0)
#         block_size_trackers.append(cur_block_size_tracker)
#         Ps.append(P)
#         Ws.append(W)
#     arg_and_channel_to_block_size = np.concatenate(block_size_trackers, axis=1)
#     channel_and_arg_to_block_size = arg_and_channel_to_block_size.transpose([1, 0, 2])
#     Ps = np.concatenate(Ps, axis=0)
#     Ws = np.concatenate([np.repeat(Ws[layer_index][:,np.newaxis], params.LAYER_NAME_TO_DIMS[layer_name][0], axis=1) for layer_index, layer_name in enumerate(layer_names)], axis=1).T
#     return Ps, Ws, channel_and_arg_to_block_size
#
# def main_dp_not_efficient(Ws, Ps, channels, num_relus):
#
#     dp_arg = 255 * np.ones(shape=(channels, num_relus), dtype=np.uint8)
#     dp_arg[0, Ws[0]] = np.arange(Ws[0].shape[0])
#
#     # dp[-1] should always hold -inf, so in line # indices = np.maximum((desired_relu_count - Ws[channel]), -1) we can be sure that bad indices will get -inf
#     dp = - np.inf * np.ones(shape=(num_relus + 1,))
#
#     dp[Ws[0]] = Ps[0]
#
#     for channel in tqdm(range(1, channels)):
#         gc.collect()
#         dp_prev = dp.copy()
#         for desired_relu_count in range(num_relus):
#             indices = np.maximum((desired_relu_count - Ws[channel]), -1)
#             dp[desired_relu_count] = (dp_prev[indices] + Ps[channel]).max()
#             if np.any((dp_prev[indices] + Ps[channel]) > -np.inf):
#                 dp_arg[channel, desired_relu_count] = (dp_prev[indices] + Ps[channel]).argmax()
#
#         dp[-1] = -np.inf
#     return dp_arg, dp[:-1]
#
# def main_dp_super_not_efficient(Ws, Ps, channels, num_relus):
#
#     dp_arg = 255 * np.ones(shape=(channels, num_relus), dtype=np.uint8)
#     dp_arg[0, Ws[0]] = np.arange(Ws[0].shape[0])
#
#     dp = - np.inf * np.ones(shape=(num_relus,))
#     dp[Ws[0]] = Ps[0]
#
#     for channel in range(1, channels):
#         gc.collect()
#         dp_prev = dp.copy()
#         for desired_relu_count in tqdm(range(num_relus)):
#
#             max_val = -np.inf
#             argmax = None
#
#             # Go over the cost (num relus) for each block configuration of channel
#             for cur_block_size_index, cur_num_relus in enumerate(Ws[channel]):
#
#                 # If you use desired_relu_count amount of relus, and current configuration cost cur_num_relus, then we
#                 # should examine the previous channel cost of desired_relu_count - cur_num_relus (i.e. dp_prev[desired_relu_count - cur_num_relus] )
#                 # Obviously, we should add  Ps[channel][cur_block_size_index] to the cost and pick the best configuration
#                 index = desired_relu_count - cur_num_relus
#                 if index >= 0:
#                     cur_v = dp_prev[index] + Ps[channel][cur_block_size_index]
#                     if cur_v > max_val:
#                         max_val = cur_v
#                         argmax = cur_block_size_index
#
#             if argmax is not None:
#                 dp[desired_relu_count] = max_val
#                 dp_arg[channel, desired_relu_count] = argmax
#
#             # indices = np.maximum((desired_relu_count - Ws[channel]), -1)
#             # dp[desired_relu_count] = (dp_prev[indices] + Ps[channel]).max()
#             # dp_arg[channel, desired_relu_count] = (dp_prev[indices] + Ps[channel]).argmax()
#
#     return dp_arg, dp
#
# # #

#
#
# def main_dp(Ws, Ps, channels, num_relus):
#
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
#
#
# def main_dp_torch(Ws, Ps, channels, num_relus):
#     Ws = torch.from_numpy(Ws)
#     Ps = torch.from_numpy(Ps)
#
#     assert num_relus < np.iinfo(np.int32).max
#     arange = torch.arange(num_relus, dtype=torch.int64).unsqueeze(dim=1)
#     indices = torch.zeros(size=(num_relus, Ws[0].shape[0]), dtype=torch.int64)
#     buffer = torch.zeros(size=(num_relus * Ws[0].shape[0], ), dtype=torch.float64)
#
#     dp_arg = IO_Buffer(num_relus, package="torch")
#     dp = - float("Inf") * torch.ones(size=(num_relus + 1,), dtype=torch.float64)
#
#     buffer_orig_shape = buffer.shape
#     indices_orig_shape = indices.shape
#     init_row = dp_arg[0].clone()
#     init_row[Ws[0]] = torch.arange(Ws[0].shape[0], dtype=torch.uint8)
#     dp_arg[0] = init_row
#     dp[Ws[0]] = Ps[0]
#
#     negative_one = -torch.ones(size=(1,), dtype=torch.int64)
#
#     device = torch.device("cuda:1")
#     Ws = Ws.to(device)  # (torch.Size([14272, 56]), torch.int64)                6.39M
#     Ps = Ps.to(device)  # (torch.Size([14272, 56]), torch.float64)              6.39M
#     arange = arange.to(device)  # (torch.Size([15656345, 1]), torch.int64)      125.25M
#     indices = indices.to(device)  # (torch.Size([15656345, 56]), torch.int64)
#     negative_one = negative_one.to(device)
#     dp = dp.to(device)  # (torch.Size([15656346]), torch.float64)
#     buffer = buffer.to(device)  # (torch.Size([876755320]), torch.float64)
#     dp_arg.buffer = dp_arg.buffer.to(device)  # (torch.Size([10, 15656345]), torch.uint8)
#
#     for channel in tqdm(range(1, channels)):
#         gc.collect()
#
#         torch.sub(arange, Ws[channel], out=indices)
#         torch.max(indices, negative_one, out=indices)
#         indices = indices.reshape(-1)
#         torch.take(dp, indices, out=buffer)
#         indices = indices.reshape(indices_orig_shape)
#         buffer = buffer.reshape(indices_orig_shape)
#         torch.add(buffer, Ps[channel], out=buffer)
#
#         dp_arg[channel] = torch.argmax(buffer, dim=1)
#         dp[:-1] = buffer[arange[:, 0], dp_arg[channel].to(torch.int64)]
#         dp_arg[channel][torch.all(buffer == -float("Inf"), dim=1)] = 255
#         buffer = buffer.reshape(buffer_orig_shape)  # Consider: buffer.shape = buffer_orig_shape to avoid rare case of copying
#
#     dp_arg.flush()
#     return dp_arg, dp[:-1]
#
# def main_dp_torch_memory(Ws, Ps, channels, num_relus):
#
#     device = torch.device("cuda:1")
#     Ws = torch.from_numpy(Ws)
#     Ps = torch.from_numpy(Ps)
#
#     assert num_relus < np.iinfo(np.int32).max
#     arange = torch.arange(num_relus, dtype=torch.int64)
#     indices = torch.zeros(size=(num_relus,), dtype=torch.int64)
#     opt_buffer = - float("Inf") * torch.ones(size=(num_relus, ), dtype=torch.float64)
#     buffer = torch.zeros(size=(num_relus, ), dtype=torch.float64)
#     boolean_index_buffer = torch.zeros(size=(num_relus, ), dtype=torch.bool)
#
#     dp_arg = IO_Buffer(num_relus, package="torch", buffer_size=1)
#     dp = - float("Inf") * torch.ones(size=(num_relus + 1,), dtype=torch.float64)
#
#     init_row = dp_arg[0].clone()
#     init_row[Ws[0]] = torch.arange(Ws[0].shape[0], dtype=torch.uint8)
#     dp_arg[0] = init_row
#     dp[Ws[0]] = Ps[0]
#
#     negative_one = -torch.ones(size=(1,), dtype=torch.int64)
#
#     Ws = Ws.to(device)  # (torch.Size([14272, 56]), torch.int64)                6.39M
#     Ps = Ps.to(device)  # (torch.Size([14272, 56]), torch.float64)              6.39M
#     arange = arange.to(device)  # (torch.Size([15656345, 1]), torch.int64)      125.25M
#     indices = indices.to(device)  # (torch.Size([15656345, 56]), torch.int64)
#     negative_one = negative_one.to(device)
#     dp = dp.to(device)  # (torch.Size([15656346]), torch.float64)
#     opt_buffer = opt_buffer.to(device)  # (torch.Size([876755320]), torch.float64)
#     buffer = buffer.to(device)  # (torch.Size([876755320]), torch.float64)
#     dp_arg.buffer = dp_arg.buffer.to(device)  # (torch.Size([10, 15656345]), torch.uint8)
#     boolean_index_buffer = boolean_index_buffer.to(device)  # (torch.Size([10, 15656345]), torch.uint8)
#
#     for channel in tqdm(range(1, channels)):
#         opt_buffer[:] = -float("Inf")
#         for index in range(Ws.shape[1]):
#
#             torch.sub(arange, Ws[channel][index], out=indices)
#             torch.max(indices, negative_one, out=indices)
#
#             torch.take(dp, indices, out=buffer)
#             torch.add(buffer, Ps[channel][index], out=buffer)
#
#             torch.gt(buffer, opt_buffer, out=boolean_index_buffer)
#             dp_arg[channel][boolean_index_buffer] = index
#             opt_buffer[boolean_index_buffer] = buffer[boolean_index_buffer]
#
#         dp[:-1] = opt_buffer
#
#     dp_arg.flush()
#     return dp_arg, dp[:-1]
#
#
# def convert_dp_arg_to_block_size_spec(dp_arg, Ws, arg_and_channel_order_to_block_size, relu_count):
#
#     num_channels = Ws.shape[0]
#     block_sizes = []
#     relu_count = int(torch.nonzero(dp_arg[num_channels-1] != 255).max().cpu().numpy())
#     for channel_order in tqdm(reversed(range(num_channels))):
#         arg = dp_arg[channel_order][relu_count]
#         channel_num_relus = Ws[channel_order, arg]
#         relu_count -= channel_num_relus
#         block_sizes.append(arg_and_channel_order_to_block_size[channel_order, arg])
#     block_sizes = np.array(block_sizes[::-1])
#
#
#     channel_order = 0
#     layer_name_to_block_size = dict()
#     for layer_name in layer_names:
#         layer_name_to_block_size[layer_name] = block_sizes[channel_order:channel_order + params.LAYER_NAME_TO_DIMS[layer_name][0], :]
#         channel_order += params.LAYER_NAME_TO_DIMS[layer_name][0]
#
#     return layer_name_to_block_size
#
#
# def get_block_size_spec(layer_names, params, channel_distortion_path, ratio, cost_type, division):
#
#     num_channels = sum(params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names)
#
#     max_cost = int(sum(get_cost((1,1), params.LAYER_NAME_TO_DIMS[layer_name][1], cost_type, division) * params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names) * ratio)
#     print(max_cost)
#     Ps, Ws, channel_and_arg_to_block_size = get_matrix_data(channel_distortion_path, params, cost_type, division)
#
#     # dp_arg_2, dp_2 = main_dp(Ws, Ps, 50, 100000)
#     # dp_arg_0, dp_0 = main_dp_super_not_efficient(Ws, Ps, 50, 100000)
#     # dp_arg_1, dp_1 = main_dp_not_efficient(Ws, Ps, 50, 100000)
#     # assert np.all(dp_arg_2.buffer[:50] == dp_arg_0)
#     # assert np.all(dp_2 == dp_0)
#     # assert False
#     # dp_arg_0, dp_0 = main_dp_torch(Ws, Ps, 6, max_cost)
#     # dp_arg_1, dp_1 = main_dp(Ws, Ps, 6, max_cost)
#     #
#     # dp_arg_0 = dp_arg_0.buffer.cpu().numpy()
#     # dp_arg_1 = dp_arg_1.buffer
#     # l = np.argwhere(dp_arg_0[5] != dp_arg_1[5])[:,0]
#     # dp_arg_0[5, [302592, 304356, 307968]]
#     # dp_arg_1[5, [302592, 304356, 307968]]
#     # print('ej')
#
#
#     # dp_arg_0, dp_0 = main_dp_torch(Ws, Ps, 10, max_cost)
#     # dp_arg_0 = dp_arg_0.buffer.cpu().numpy()
#     # dp_0 = dp_0.cpu().numpy()
#
#     dp_arg, dp = main_dp_torch_memory(Ws, Ps, num_channels, max_cost)
#     np.save(file="/home/yakir/dp.npy", arr=dp.cpu().numpy())
#     # print('hey')
#     #
#     # dp_arg_1 = dp_arg_1.buffer.cpu().numpy()
#     # dp_1 = dp_1.cpu().numpy()
#     #
#     # dp_arg_2, dp_2 = main_dp(Ws, Ps, 25, max_cost)
#     # np.all(dp_arg_2.buffer == dp_arg_1)
#     # np.all(dp_2 == dp_1)
#     # #
#     # # dp_arg = IO_Buffer(max_cost, package="torch", load=True)
#     layer_name_to_block_size = convert_dp_arg_to_block_size_spec(dp_arg, Ws, channel_and_arg_to_block_size, max_cost - 1)
#     return layer_name_to_block_size, dp_arg

class IO_Buffer:
    def __init__(self, word_size, package="numpy", load=False, buffer_size=10):
        self.buffer_size = buffer_size
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
                self.buffer[:] = np.load(self.buffer_path_format.format(channel_frame))
            elif self.package == "torch":
                self.buffer[:] = torch.load(self.buffer_path_format.format(channel_frame))
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
        self.dirty = True  # The user can change the buffer later on
        return self.buffer[channel % self.buffer_size]


class MultipleChoiceKnapsack:
    def __init__(self, params, cost_type, division, ratio, seed, channel_distortion_path, cur_iter, num_iters, max_cost=None):
        self.params = params
        self.cost_type = cost_type
        self.division = division
        self.ratio = ratio
        self.seed = seed
        self.channel_distortion_path = channel_distortion_path
        self.cur_iter = cur_iter
        self.num_iters = num_iters

        self.channel_order_to_layer, self.channel_order_to_channel, self.channel_order_to_dim = \
            get_channel_order_statistics(self.params)

        self.layer_name_to_noise = dict()
        self.read_noise_files()

        _, self.channel_orders = get_channels_subset(self.seed, self.params, self.cur_iter, self.num_iters)
        self.num_channels = len(self.channel_orders)

        get_baseline_cost = lambda channel_order: get_cost(block_size=(1, 1),
                                                           activation_dim=self.channel_order_to_dim[channel_order],
                                                           cost_type=self.cost_type,
                                                           division=self.division)

        if max_cost is None:
            max_cost = sum(get_baseline_cost(channel_order) for channel_order in self.channel_orders)
            self.max_cost = int(max_cost * self.ratio)
        else:
            self.max_cost = max_cost

    @staticmethod
    def run_multiple_choice(Ws, Ps, num_rows, num_columns):
        device = torch.device("cuda:0")
        Ws = torch.from_numpy(Ws)
        Ps = torch.from_numpy(Ps)

        assert num_columns < np.iinfo(np.int32).max
        arange = torch.arange(num_columns, dtype=torch.int64)
        indices = torch.zeros(size=(num_columns,), dtype=torch.int64)
        opt_buffer = - float("Inf") * torch.ones(size=(num_columns,), dtype=torch.float64)
        buffer = torch.zeros(size=(num_columns,), dtype=torch.float64)
        boolean_index_buffer = torch.zeros(size=(num_columns,), dtype=torch.bool)

        dp_arg = IO_Buffer(num_columns, package="torch", buffer_size=1)
        dp = - float("Inf") * torch.ones(size=(num_columns + 1,), dtype=torch.float64)

        init_row = dp_arg[0].clone()
        init_row[Ws[0]] = torch.arange(Ws[0].shape[0], dtype=torch.uint8)
        dp_arg[0] = init_row
        dp[Ws[0]] = Ps[0]

        negative_one = -torch.ones(size=(1,), dtype=torch.int64)

        Ws = Ws.to(device)  # (torch.Size([14272, 56]), torch.int64)                6.39M
        Ps = Ps.to(device)  # (torch.Size([14272, 56]), torch.float64)              6.39M
        arange = arange.to(device)  # (torch.Size([15656345, 1]), torch.int64)      125.25M
        indices = indices.to(device)  # (torch.Size([15656345, 56]), torch.int64)
        negative_one = negative_one.to(device)
        dp = dp.to(device)  # (torch.Size([15656346]), torch.float64)
        opt_buffer = opt_buffer.to(device)  # (torch.Size([876755320]), torch.float64)
        buffer = buffer.to(device)  # (torch.Size([876755320]), torch.float64)
        dp_arg.buffer = dp_arg.buffer.to(device)  # (torch.Size([10, 15656345]), torch.uint8)
        boolean_index_buffer = boolean_index_buffer.to(device)  # (torch.Size([10, 15656345]), torch.uint8)

        for channel in tqdm(range(1, num_rows)):
            opt_buffer[:] = -float("Inf")
            for index in range(Ws.shape[1]):
                torch.sub(arange, Ws[channel][index], out=indices)
                torch.max(indices, negative_one, out=indices)

                torch.take(dp, indices, out=buffer)
                torch.add(buffer, Ps[channel][index], out=buffer)

                torch.gt(buffer, opt_buffer, out=boolean_index_buffer)
                dp_arg[channel][boolean_index_buffer] = index
                opt_buffer[boolean_index_buffer] = buffer[boolean_index_buffer]

            dp[:-1] = opt_buffer

        dp_arg.flush()
        return dp_arg, dp[:-1]

    def read_noise_files(self):
        for layer_name in self.params.LAYER_NAMES:
            glob_pattern = os.path.join(self.channel_distortion_path, f"{layer_name}_*.pickle")
            files = glob.glob(glob_pattern)
            assert len(files) > 0, glob_pattern

            noise = np.stack([pickle.load(open(f, 'rb'))["Noise"] for f in files])
            noise = noise.mean(axis=0).mean(axis=2).T  # noise.shape = [N-block-sizes, N-channels]

            signal = np.stack([pickle.load(open(f, 'rb'))["Signal"] for f in files])
            signal = signal.mean(axis=0).mean(axis=2).T

            noise = noise / signal
            noise = noise[:-1]
            self.layer_name_to_noise[layer_name] = -np.array(noise).T

    def prepare_matrices(self):
        Ps = []
        Ws = []
        block_size_trackers = []

        for channel_order in self.channel_orders:
            layer_name = self.channel_order_to_layer[channel_order]
            channel_index = self.channel_order_to_channel[channel_order]
            layer_dim = self.params.LAYER_NAME_TO_DIMS[layer_name][1]

            block_sizes = np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])[:-1]  # TODO: either use [1,0] or don't infer it at all
            assert np.max(block_sizes) < 255

            W = np.array([get_cost(tuple(block_size), layer_dim, self.cost_type, self.division) for block_size in block_sizes])
            P = self.layer_name_to_noise[layer_name][channel_index]

            block_size_groups = defaultdict(list)
            for block_size_index, block_size in enumerate(block_sizes):
                cur_cost = get_cost(tuple(block_size), layer_dim, self.cost_type, self.division)  # 1 here to avoid weird stuff
                block_size_groups[cur_cost].append(block_size_index)

            P_new = []
            W_new = []
            cur_block_size_tracker = []
            for k, v in block_size_groups.items():
                cur_block_sizes = block_sizes[v]

                P_same_weight = np.stack([P[row_index] for row_index in v])
                argmax = P_same_weight.argmax(axis=0)
                max_ = P_same_weight.max(axis=0)

                cur_block_size_tracker.append(cur_block_sizes[argmax])
                P_new.append(max_)
                W_new.append(W[v[0]])

            cur_block_size_tracker = np.array(cur_block_size_tracker)
            P = np.array(P_new)
            W = np.array(W_new)

            block_size_trackers.append(cur_block_size_tracker)
            Ps.append(P)
            Ws.append(W)

        padding_factor = max(set([x.shape for x in Ps]))[0]

        Ps = np.stack([np.pad(P, (0, padding_factor - P.shape[0]), mode="constant", constant_values=-np.inf) for P in Ps])
        Ws = np.stack([np.pad(W, (0, padding_factor - W.shape[0]), mode="constant", constant_values=0) for W in Ws])
        block_size_trackers = np.stack([np.pad(X, ((0, padding_factor - X.shape[0]), (0, 0))) for X in block_size_trackers])
        return Ps, Ws, block_size_trackers

    def get_optimal_block_sizes(self):

        Ps, Ws, block_size_tracker = self.prepare_matrices()

        dp_arg, dp = self.run_multiple_choice(Ws, Ps, self.num_channels, self.max_cost)

        block_size_spec = self.convert_dp_arg_to_block_size_spec(dp_arg, Ws, block_size_tracker)

        return block_size_spec

    def convert_dp_arg_to_block_size_spec(self, dp_arg, Ws, block_size_tracker):

        num_channels = Ws.shape[0]
        block_sizes = []
        column = int(torch.nonzero(dp_arg[num_channels - 1] != 255).max().cpu().numpy())

        for channel_index in tqdm(reversed(range(num_channels))):
            arg = dp_arg[channel_index][column]
            channel_cost = Ws[channel_index, arg]
            column -= channel_cost
            block_sizes.append(block_size_tracker[channel_index, arg])
        block_sizes = np.array(block_sizes[::-1])

        block_size_spec = {layer_name: np.ones(shape=(self.params.LAYER_NAME_TO_DIMS[layer_name][0], 2), dtype=np.int32)
                           for layer_name in self.params.LAYER_NAMES}

        for channel_order, block_size in zip(self.channel_orders, block_sizes):
            channel_index = self.channel_order_to_channel[channel_order]
            layer_name = self.channel_order_to_layer[channel_order]
            block_size_spec[layer_name][channel_index] = block_size

        return block_size_spec


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--block_size_spec_file_name', type=str, default="/home/yakir/block_size_spec_4x4_algo.pickle")
    parser.add_argument('--channel_distortion_path', type=str, default="/home/yakir/resnet50_8xb32_in1k_finetune_0.0001_avg_pool_dummy")
    parser.add_argument('--config', type=str, default="/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k_finetune_0.0001_avg_pool.py")
    parser.add_argument('--ratio', type=float, default=None)
    parser.add_argument('--cost_type', type=str, default="ReLU")
    parser.add_argument('--division', type=int, default=1)
    parser.add_argument('--cur_iter', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--max_cost', type=int, default=644224)

    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    params = param_factory(cfg)
    layer_names = params.LAYER_NAMES
    ratio = args.ratio
    mck = MultipleChoiceKnapsack(params=params,
                                 cost_type=args.cost_type,
                                 division=args.division,
                                 ratio=args.ratio,
                                 seed=123,
                                 channel_distortion_path=args.channel_distortion_path,
                                 cur_iter=args.cur_iter,
                                 num_iters=args.num_iters,
                                 max_cost=args.max_cost)

    block_size_spec = mck.get_optimal_block_sizes()

    if not os.path.exists(os.path.dirname(args.block_size_spec_file_name)):
        os.makedirs(os.path.dirname(args.block_size_spec_file_name))

    with open(args.block_size_spec_file_name, "wb") as f:
        pickle.dump(obj=block_size_spec, file=f)