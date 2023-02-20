import os.path
import mmcv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import torch

from research.distortion.parameters.factory import param_factory
from research.distortion.utils import get_num_relus
from research.distortion.utils import get_channel_order_statistics
from research.distortion.knapsack.multiple_choice_knapsack_solver import MultipleChoiceKnapsackSolver


class MultipleChoiceKnapsackPatchSizeExtractor:
    def __init__(self, params, channel_distortion_path, ratio=None, max_cost=None, device=0):

        assert max_cost is None or ratio is None
        self.params = params
        self.ratio = ratio
        self.channel_distortion_path = channel_distortion_path

        self.channel_order_to_layer, self.channel_order_to_channel, self.channel_order_to_dim = \
            get_channel_order_statistics(self.params)

        self.layer_name_to_noise = dict()
        self.read_noise_files()

        self.num_channels = len(self.channel_order_to_layer)
        self.channel_orders = np.arange(self.num_channels)

        if max_cost is None:
            get_baseline_cost = lambda channel_order: get_num_relus(block_size=(1, 1),
                                                                    activation_dim=self.channel_order_to_dim[ channel_order])
            max_cost = sum(get_baseline_cost(channel_order) for channel_order in self.channel_orders)
            self.max_cost = int(max_cost * self.ratio)
        else:
            self.max_cost = max_cost

        self.device = device

    def read_noise_files(self):
        for layer_name in self.params.LAYER_NAMES:
            self.layer_name_to_noise[layer_name] = np.load(
                os.path.join(self.channel_distortion_path, f"{layer_name}.npy"))

    def prepare_matrices(self):
        Ps = []
        Ws = []
        block_size_trackers = []

        for channel_order in self.channel_orders:
            layer_name = self.channel_order_to_layer[channel_order]
            channel_index = self.channel_order_to_channel[channel_order]
            layer_dim = self.params.LAYER_NAME_TO_DIMS[layer_name][1]

            block_sizes = np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])[
                          :-1]  # TODO: either use [1,0] or don't infer it at all

            W = np.array([get_num_relus(tuple(block_size), layer_dim) for block_size in block_sizes])
            P = self.layer_name_to_noise[layer_name][channel_index]

            block_size_groups = defaultdict(list)
            for block_size_index, block_size in enumerate(block_sizes):
                cur_cost = get_num_relus(tuple(block_size), layer_dim)
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

        Ps = np.stack(
            [np.pad(P, (0, padding_factor - P.shape[0]), mode="constant", constant_values=-np.inf) for P in Ps])
        Ws = np.stack([np.pad(W, (0, padding_factor - W.shape[0]), mode="constant", constant_values=0) for W in Ws])
        block_size_trackers = np.stack(
            [np.pad(X, ((0, padding_factor - X.shape[0]), (0, 0))) for X in block_size_trackers])
        return Ps, Ws, block_size_trackers

    def get_optimal_block_sizes(self):

        Ps, Ws, block_size_tracker = self.prepare_matrices()

        dp_arg, dp = MultipleChoiceKnapsackSolver.run_multiple_choice_knapsack(Ws, Ps, self.num_channels, self.max_cost,
                                                                               self.device)

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

    parser.add_argument('--block_size_spec_file_name', type=str, default="/home/yakir/deepreduce_comparison/distortions/superlightweight/block_sizes/14.33K.pickle")
    parser.add_argument('--channel_distortion_path', type=str, default="/home/yakir/deepreduce_comparison/distortions/superlightweight/distortion_collected/")
    parser.add_argument('--config', type=str, default="/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet18_cifar100/superlightweight.py")
    parser.add_argument('--ratio', type=float, default=None)
    parser.add_argument('--max_cost', type=int, default=14330)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    params = param_factory(cfg)

    mck = MultipleChoiceKnapsackPatchSizeExtractor(
        params=params,
        channel_distortion_path=args.channel_distortion_path,
        ratio=args.ratio,
        max_cost=args.max_cost,
        device=args.device)

    block_size_spec = mck.get_optimal_block_sizes()

    if not os.path.exists(os.path.dirname(args.block_size_spec_file_name)):
        os.makedirs(os.path.dirname(args.block_size_spec_file_name))

    with open(args.block_size_spec_file_name, "wb") as f:
        pickle.dump(obj=block_size_spec, file=f)
