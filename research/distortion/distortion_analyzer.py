import matplotlib
matplotlib.use("TkAgg")
import argparse
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import pickle

from research.parameters.factory import ParamsFactory
from research.distortion.distortion_utils import DistortionUtils, get_num_relus
from research.distortion.utils import get_channel_order_statistics, get_channels_component, get_num_of_channels
from matplotlib import pyplot as plt


class RandomDistortionHandler:
    def __init__(self, gpu_id, output_path, params):
        self.device = gpu_id
        self.params = params
        self.distortion_utils = DistortionUtils(gpu_id=gpu_id, params=self.params)
        self.output_path = output_path

        self.keys = ["Noise", "Signal", "Distorted Loss", "Baseline Loss"]

    def foo_bar3(self, layer_start, layer_end):

        channel_distortion_path = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/one_group/channel_distortions"
        out_path = f"/home/yakir/Data2/tmp_stats/layer_{layer_start}-{layer_end}"
        os.makedirs(out_path, exist_ok=True)
        input_block_name = "conv1"
        output_block_name = "decode"
        general_seed = 13

        layer_names = self.params.LAYER_NAMES[layer_start:layer_end]
        # layer_names = self.params.LAYER_NAMES[:2] + self.params.LAYER_NAMES[3:7]

        channel_order_to_dim = np.hstack([[self.params.LAYER_NAME_TO_DIMS[layer_name][1]] * self.params.LAYER_NAME_TO_DIMS[layer_name][0]  for layer_name in layer_names])
        channel_order_to_block_size_len = np.hstack([[len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])] * self.params.LAYER_NAME_TO_DIMS[layer_name][0]  for layer_name in layer_names])
        channel_order_to_channel = np.hstack([np.arange(self.params.LAYER_NAME_TO_DIMS[layer_name][0]) for layer_name in layer_names])
        channel_order_to_layer = np.hstack([[layer_name] * self.params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names])

        tot_channel_count = sum(self.params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names)
        layer_to_noise_statistics = {layer_name:pickle.load(open(os.path.join(channel_distortion_path,f"{layer_name}_{self.device}.pickle"), 'rb')) for layer_name in layer_names}

        if 'decode_0' in layer_to_noise_statistics:
            layer_to_noise_statistics['decode_0']["Signal"][:, :, :] = layer_to_noise_statistics['decode_0']["Signal"][:, :1, :]

        noises = {layer_name: layer_to_noise_statistics[layer_name]['Noise'].mean(axis=2) for layer_name in layer_names}
        signals = {layer_name: layer_to_noise_statistics[layer_name]['Signal'].mean(axis=2) for layer_name in layer_names}
        distorted_losses = {layer_name: layer_to_noise_statistics[layer_name]['Distorted Loss'].mean(axis=2) for layer_name in layer_names}

        # snr = {layer_name: noises[layer_name]/signals[layer_name] for layer_name in layer_names}
        # block_to_test = [[1, 1], [1, 2], [2, 1], [2, 2], [2, 4], [4, 2], [3, 3], [4, 4], [3, 6], [6, 3]]
        block_to_test = [[1, 1],
                         [1, 2],
                         [2, 1],
                         [1, 3],
                         [3, 1],
                         [1, 4],
                         [2, 2],
                         [4, 1],
                         [1, 5],
                         [5, 1],
                         [1, 6],
                         [2, 3],
                         [3, 2],
                         [6, 1],
                         # [1, 7],
                         # [7, 1],
                         # [1, 8],
                         [2, 4],
                         [4, 2],
                         # [8, 1],
                         # [1, 9],
                         [3, 3],
                         # [9, 1],
                         # [1, 10],
                         [2, 5],
                         [5, 2],
                         # [10, 1],
                         # [1, 11],
                         # [11, 1],
                         # [1, 12],
                         [2, 6],
                         [3, 4],
                         [4, 3],
                         [6, 2],
                         # [12, 1],
                         # [1, 13],
                         # [13, 1],
                         # [1, 14],
                         [2, 7],
                         [7, 2],
                         # [14, 1],
                         # [1, 15],
                         [3, 5],
                         [5, 3],
                         # [15, 1],
                         # [1, 16],
                         [2, 8],
                         [4, 4],
                         [8, 2],
                         # [16, 1],
                         # [2, 9],
                         [3, 6],
                         [6, 3],
                         # [9, 2],
                         [2, 10],
                         [4, 5],
                         [5, 4],
                         [10, 2],
                         [3, 7],
                         [7, 3],
                         # [2, 11],
                         # [11, 2],
                         [2, 12],
                         [3, 8],
                         [4, 6],
                         [6, 4],
                         [8, 3],
                         [12, 2],
                         [5, 5],
                         # [2, 13],
                         # [13, 2],
                         [3, 9],
                         [9, 3],
                         # [2, 14],
                         [4, 7],
                         [7, 4],
                         # [14, 2],
                         # [2, 15],
                         # [3, 10],
                         [5, 6],
                         [6, 5],
                         # [10, 3],
                         # [15, 2],
                         [2, 16],
                         [4, 8],
                         [8, 4],
                         [16, 2],
                         # [3, 11],
                         # [11, 3],
                         # [5, 7],
                         # [7, 5],
                         [3, 12],
                         # [4, 9],
                         [6, 6],
                         # [9, 4],
                         [12, 3],
                         # [3, 13],
                         # [13, 3],
                         [4, 10],
                         [5, 8],
                         [8, 5],
                         [10, 4],
                         # [3, 14],
                         # [6, 7],
                         # [7, 6],
                         # [14, 3],
                         # [4, 11],
                         # [11, 4],
                         [3, 15],
                         # [5, 9],
                         # [9, 5],
                         [15, 3],
                         # [3, 16],
                         [4, 12],
                         [6, 8],
                         [8, 6],
                         [12, 4],
                         # [16, 3],
                         [7, 7],
                         [5, 10],
                         [10, 5],
                         # [4, 13],
                         # [13, 4],
                         [6, 9],
                         [9, 6],
                         # [5, 11],
                         # [11, 5],
                         # [4, 14],
                         # [7, 8],
                         # [8, 7],
                         # [14, 4],
                         # [4, 15],
                         [5, 12],
                         [6, 10],
                         [10, 6],
                         [12, 5],
                         # [15, 4],
                         # [7, 9],
                         # [9, 7],
                         [4, 16],
                         [8, 8],
                         [16, 4],
                         # [5, 13],
                         # [13, 5],
                         # [6, 11],
                         # [11, 6],
                         # [5, 14],
                         # [7, 10],
                         # [10, 7],
                         # [14, 5],
                         [6, 12],
                         # [8, 9],
                         # [9, 8],
                         [12, 6],
                         [5, 15],
                         [15, 5],
                         # [7, 11],
                         # [11, 7],
                         # [6, 13],
                         # [13, 6],
                         # [5, 16],
                         [8, 10],
                         [10, 8],
                         # [16, 5],
                         [9, 9],
                         # [6, 14],
                         # [7, 12],
                         # [12, 7],
                         # [14, 6],
                         # [8, 11],
                         # [11, 8],
                         # [6, 15],
                         # [9, 10],
                         # [10, 9],
                         # [15, 6],
                         # [7, 13],
                         # [13, 7],
                         [6, 16],
                         [8, 12],
                         [12, 8],
                         [16, 6],
                         # [7, 14],
                         # [14, 7],
                         # [9, 11],
                         # [11, 9],
                         [10, 10],
                         # [8, 13],
                         # [13, 8],
                         # [7, 15],
                         # [15, 7],
                         [9, 12],
                         [12, 9],
                         # [10, 11],
                         # [11, 10],
                         # [7, 16],
                         [8, 14],
                         [14, 8],
                         # [16, 7],
                         # [9, 13],
                         # [13, 9],
                         # [8, 15],
                         # [10, 12],
                         # [12, 10],
                         # [15, 8],
                         [11, 11],
                         # [9, 14],
                         # [14, 9],
                         [8, 16],
                         [16, 8],
                         # [10, 13],
                         # [13, 10],
                         # [11, 12],
                         # [12, 11],
                         # [9, 15],
                         # [15, 9],
                         [10, 14],
                         [14, 10],
                         # [11, 13],
                         # [13, 11],
                         # [9, 16],
                         [12, 12],
                         # [16, 9],
                         [10, 15],
                         [15, 10],
                         # [11, 14],
                         # [14, 11],
                         # [12, 13],
                         # [13, 12],
                         [10, 16],
                         [16, 10],
                         # [11, 15],
                         # [15, 11],
                         [12, 14],
                         [14, 12],
                         [13, 13],
                         # [11, 16],
                         # [16, 11],
                         # [12, 15],
                         # [15, 12],
                         # [13, 14],
                         # [14, 13],
                         [12, 16],
                         [16, 12],
                         # [13, 15],
                         # [15, 13],
                         [14, 14],
                         # [13, 16],
                         # [16, 13],
                         # [14, 15],
                         # [15, 14],
                         # [14, 16],
                         # [16, 14],
                         [15, 15],
                         # [15, 16],
                         # [16, 15],
                         [16, 16],
                         [32, 32],
                         [0, 1],
                         [1, 0]
                         ]
        relevant_block_sizes = {layer_name: [block_index for block_index, block_size in enumerate(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]) if block_size in block_to_test] for layer_name in layer_names}
        all_channels = np.arange(tot_channel_count)
        for i in tqdm(range(100000)):
            seed = i #2*i+self.device
            block_size_spec = {layer_name: np.ones((self.params.LAYER_NAME_TO_DIMS[layer_name][0], 2), dtype=np.uint8) for layer_name in layer_names}
            block_size_indices_spec = {layer_name: np.zeros((self.params.LAYER_NAME_TO_DIMS[layer_name][0],), dtype=np.uint8) for layer_name in layer_names}

            for dim in np.unique(channel_order_to_dim):
                for n_blocks in np.unique(np.unique(channel_order_to_block_size_len)):
                    np.random.seed(seed)
                    cur_channels_orders = all_channels[np.logical_and(channel_order_to_dim == dim, channel_order_to_block_size_len == n_blocks)]
                    np.random.shuffle(cur_channels_orders)
                    np.random.seed(general_seed)
                    for channel_order in cur_channels_orders:
                        layer_name = channel_order_to_layer[channel_order]
                        channel = channel_order_to_channel[channel_order]

                        block_size_index = np.random.choice(relevant_block_sizes[layer_name])

                        block_size = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name][block_size_index]
                        block_size_indices_spec[layer_name][channel] = block_size_index
                        block_size_spec[layer_name][channel] = block_size

            cur_assets = self.distortion_utils.get_batch_distortion(
                baseline_block_size_spec=dict(),
                block_size_spec=block_size_spec,
                batch_index=self.device,
                batch_size=16,
                input_block_name=input_block_name,
                output_block_name=output_block_name)

            noise = np.hstack([noises[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            signal = np.hstack([signals[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            distorted_loss = np.hstack([distorted_losses[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            loss = cur_assets['Distorted Loss'].mean()
            relus_count = sum([sum([get_num_relus(tuple(block_size), self.params.LAYER_NAME_TO_DIMS[layer_name][1]) for block_size in block_size_spec[layer_name]]) for layer_name in layer_names])

            np.save(file=os.path.join(out_path, f"noise_{seed}_{self.device }.npy"), arr=noise)
            np.save(file=os.path.join(out_path, f"signal_{seed}_{self.device }.npy"), arr=signal)
            np.save(file=os.path.join(out_path, f"loss_{seed}_{self.device }.npy"), arr=loss)
            np.save(file=os.path.join(out_path, f"relus_count_{seed}_{self.device }.npy"), arr=relus_count)
            np.save(file=os.path.join(out_path, f"distorted_loss_{seed}_{self.device }.npy"), arr=distorted_loss)

    def foo_bar4(self, layer_start, layer_end, num_channels, general_seed=123):

        channel_distortion_path = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/one_group_mini_4000_channels/channel_distortions"
        out_path = f"/home/yakir/Data2/tmp_stats/foo_bar4_{num_channels}/layer_{layer_start}-{layer_end}"
        os.makedirs(out_path, exist_ok=True)
        input_block_name = "conv1"
        output_block_name = "decode"
        num_channels = num_channels

        layer_names = self.params.LAYER_NAMES[layer_start:layer_end]

        channel_order_to_channel = np.hstack([np.arange(self.params.LAYER_NAME_TO_DIMS[layer_name][0]) for layer_name in layer_names])
        channel_order_to_layer = np.hstack([[layer_name] * self.params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names])

        tot_channel_count = sum(self.params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names)
        layer_to_noise_statistics = {layer_name:pickle.load(open(os.path.join(channel_distortion_path,f"{layer_name}_{self.device}.pickle"), 'rb')) for layer_name in layer_names}
        if 'decode_0' in layer_to_noise_statistics:
            layer_to_noise_statistics['decode_0']["Signal"][:, :, :] = layer_to_noise_statistics['decode_0']["Signal"][:, :1, :]

        noises = {layer_name: layer_to_noise_statistics[layer_name]['Noise'].mean(axis=2) for layer_name in layer_names}
        signals = {layer_name: layer_to_noise_statistics[layer_name]['Signal'].mean(axis=2) for layer_name in layer_names}
        distorted_losses = {layer_name: layer_to_noise_statistics[layer_name]['Distorted Loss'].mean(axis=2) for layer_name in layer_names}

        all_channels = np.arange(tot_channel_count)
        np.random.seed(general_seed)
        np.random.shuffle(all_channels)

        channels_to_use = all_channels[:num_channels]
        constant_channels_to_use = all_channels[num_channels:]
        constant_block_sizes_to_use = np.random.choice(num_of_blocks, size=constant_channels_to_use.shape[0], replace=True)

        for seed in tqdm(range(100000)):

            block_size_spec = {layer_name: np.ones((self.params.LAYER_NAME_TO_DIMS[layer_name][0], 2), dtype=np.uint8) for layer_name in layer_names}
            block_size_indices_spec = {layer_name: np.zeros((self.params.LAYER_NAME_TO_DIMS[layer_name][0],), dtype=np.uint8) for layer_name in layer_names}
            np.random.seed(general_seed)

            block_sizes_to_use = np.random.choice(num_of_blocks, size=num_channels, replace=True)
            np.random.seed(seed)
            np.random.shuffle(block_sizes_to_use)

            for index, channel_order in enumerate(channels_to_use):
                layer_name = channel_order_to_layer[channel_order]
                channel = channel_order_to_channel[channel_order]

                assert len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]) == num_of_blocks

                block_size_index = block_sizes_to_use[index]

                block_size = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name][block_size_index]
                block_size_indices_spec[layer_name][channel] = block_size_index
                block_size_spec[layer_name][channel] = block_size
            #
            # for index, channel_order in enumerate(constant_channels_to_use):
            #     layer_name = channel_order_to_layer[channel_order]
            #     channel = channel_order_to_channel[channel_order]
            #
            #     assert len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]) == num_of_blocks
            #
            #     block_size_index = constant_block_sizes_to_use[index]
            #
            #     block_size = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name][block_size_index]
            #     assert block_size_indices_spec[layer_name][channel]  == 0
            #     assert block_size_spec[layer_name][channel].sum() == 2
            #     block_size_indices_spec[layer_name][channel] = block_size_index
            #     block_size_spec[layer_name][channel] = block_size

            cur_assets = self.distortion_utils.get_batch_distortion(
                baseline_block_size_spec=dict(),
                block_size_spec=block_size_spec,
                batch_index=self.device,
                batch_size=16,
                input_block_name=input_block_name,
                output_block_name=output_block_name)

            noise = np.hstack([noises[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            signal = np.hstack([signals[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            distorted_loss = np.hstack([distorted_losses[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            loss = cur_assets['Distorted Loss'].mean()
            relus_count = sum([sum([get_num_relus(tuple(block_size), self.params.LAYER_NAME_TO_DIMS[layer_name][1]) for block_size in block_size_spec[layer_name]]) for layer_name in layer_names])

            noise = noise[channels_to_use]
            signal = signal[channels_to_use]
            distorted_loss = distorted_loss[channels_to_use]
            loss = loss
            relus_count = relus_count
            np.save(file=os.path.join(out_path, f"noise_{seed}_{self.device }.npy"), arr=noise)
            np.save(file=os.path.join(out_path, f"signal_{seed}_{self.device }.npy"), arr=signal)
            np.save(file=os.path.join(out_path, f"loss_{seed}_{self.device }.npy"), arr=loss)
            np.save(file=os.path.join(out_path, f"relus_count_{seed}_{self.device }.npy"), arr=relus_count)
            np.save(file=os.path.join(out_path, f"distorted_loss_{seed}_{self.device }.npy"), arr=distorted_loss)

    def foo_bar5(self, layer_start, layer_end, num_channels, general_seed=123, iter_=0):
        channel_distortion_path = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/one_group_mini_4000_channels/channel_distortions_1"
        block_size_spec_file_name = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/one_group_mini_4000_channels/block_size_spec_0.pickle"
        out_path = f"/home/yakir/Data2/tmp_stats/foo_bar5_{num_channels}_{iter_}/layer_{layer_start}-{layer_end}"

        os.makedirs(out_path, exist_ok=True)
        input_block_name = "conv1"
        output_block_name = "decode"
        num_channels = num_channels

        layer_names = self.params.LAYER_NAMES[layer_start:layer_end]
        assert len(set([len(set(tuple(x) for x in self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])) for layer_name in layer_names])) == 1
        num_of_blocks = len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_names[0]])
        index_dict = {tuple(block_size): index for index, block_size in enumerate(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_names[0]])}
        channel_order_to_channel = np.hstack([np.arange(self.params.LAYER_NAME_TO_DIMS[layer_name][0]) for layer_name in layer_names])
        channel_order_to_layer = np.hstack([[layer_name] * self.params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names])

        tot_channel_count = sum(self.params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names)
        layer_to_noise_statistics = {layer_name:pickle.load(open(os.path.join(channel_distortion_path,f"{layer_name}_{self.device}.pickle"), 'rb')) for layer_name in layer_names}
        if 'decode_0' in layer_to_noise_statistics:
            layer_to_noise_statistics['decode_0']["Signal"][:, :, :] = layer_to_noise_statistics['decode_0']["Signal"][:, :1, :]

        noises = {layer_name: layer_to_noise_statistics[layer_name]['Noise'].mean(axis=2) for layer_name in layer_names}
        signals = {layer_name: layer_to_noise_statistics[layer_name]['Signal'].mean(axis=2) for layer_name in layer_names}
        distorted_losses = {layer_name: layer_to_noise_statistics[layer_name]['Distorted Loss'].mean(axis=2) for layer_name in layer_names}
        channel_order_to_dim = np.hstack([[self.params.LAYER_NAME_TO_DIMS[layer_name][1]] * self.params.LAYER_NAME_TO_DIMS[layer_name][0]  for layer_name in layer_names])

        all_channels = np.arange(tot_channel_count)
        np.random.seed(general_seed)
        np.random.shuffle(all_channels)

        channels_to_use = all_channels[num_channels*iter_:num_channels*(iter_+1)]
        channels_to_use.sort()
        dims = np.array([channel_order_to_dim[channel_order] for channel_order in channels_to_use])
        for seed in tqdm(range(100000)):

            if block_size_spec_file_name is not None:
                block_size_spec = pickle.load(open(block_size_spec_file_name, "rb"))
                block_size_indices_spec = {layer_name: np.zeros((self.params.LAYER_NAME_TO_DIMS[layer_name][0],), dtype=np.uint8) for layer_name in layer_names}
                for layer_name in block_size_spec:
                    for block_size_index, block_size in enumerate(block_size_spec[layer_name]):
                        block_size_indices_spec[layer_name][block_size_index] = index_dict[tuple(block_size)]
            else:
                block_size_spec = {layer_name: np.ones((self.params.LAYER_NAME_TO_DIMS[layer_name][0], 2), dtype=np.uint8) for layer_name in layer_names}
                block_size_indices_spec = {layer_name: np.zeros((self.params.LAYER_NAME_TO_DIMS[layer_name][0],), dtype=np.uint8) for layer_name in layer_names}
            np.random.seed(general_seed)

            block_sizes_to_use = np.random.choice(num_of_blocks, size=num_channels, replace=True)

            np.random.seed(seed)
            b_128 = block_sizes_to_use[dims == 128]
            b_64 = block_sizes_to_use[dims == 64]
            b_32 = block_sizes_to_use[dims == 32]
            b_1 = block_sizes_to_use[dims == 1]
            np.random.shuffle(b_128)
            np.random.shuffle(b_64)
            np.random.shuffle(b_32)
            np.random.shuffle(b_1)
            block_sizes_to_use = np.zeros_like(block_sizes_to_use)
            block_sizes_to_use[dims == 128] = b_128
            block_sizes_to_use[dims == 64] = b_64
            block_sizes_to_use[dims == 32] = b_32
            block_sizes_to_use[dims == 1] = b_1

            for index, channel_order in enumerate(channels_to_use):
                layer_name = channel_order_to_layer[channel_order]
                channel = channel_order_to_channel[channel_order]

                assert len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]) == num_of_blocks

                block_size_index = block_sizes_to_use[index]

                block_size = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name][block_size_index]
                block_size_indices_spec[layer_name][channel] = block_size_index
                assert np.all(block_size_spec[layer_name][channel] == [1,1])
                block_size_spec[layer_name][channel] = block_size

            cur_assets = self.distortion_utils.get_batch_distortion(
                baseline_block_size_spec=dict(),
                block_size_spec=block_size_spec,
                batch_index=self.device,
                batch_size=16,
                input_block_name=input_block_name,
                output_block_name=output_block_name)

            noise = np.hstack([noises[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            signal = np.hstack([signals[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            distorted_loss = np.hstack([distorted_losses[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            loss = cur_assets['Distorted Loss'].mean()
            relus_count = sum([sum([get_num_relus(tuple(block_size), self.params.LAYER_NAME_TO_DIMS[layer_name][1]) for block_size in block_size_spec[layer_name]]) for layer_name in layer_names])

            noise = noise[channels_to_use]
            signal = signal[channels_to_use]
            distorted_loss = distorted_loss[channels_to_use]
            loss = loss
            relus_count = relus_count
            np.save(file=os.path.join(out_path, f"noise_{seed}_{self.device }.npy"), arr=noise)
            np.save(file=os.path.join(out_path, f"signal_{seed}_{self.device }.npy"), arr=signal)
            np.save(file=os.path.join(out_path, f"loss_{seed}_{self.device }.npy"), arr=loss)
            np.save(file=os.path.join(out_path, f"relus_count_{seed}_{self.device }.npy"), arr=relus_count)
            np.save(file=os.path.join(out_path, f"distorted_loss_{seed}_{self.device }.npy"), arr=distorted_loss)

    def get_stats(self, channel_distortion_path, layer_names):
        layer_to_noise_statistics = {layer_name: pickle.load(
            open(os.path.join(channel_distortion_path, f"{layer_name}_{self.device}.pickle"), 'rb')) for layer_name in
                                     layer_names}

        if 'decode_0' in layer_to_noise_statistics:
            layer_to_noise_statistics['decode_0']["Signal"][:, :, :] = layer_to_noise_statistics['decode_0']["Signal"][
                                                                       :, :1, :]

        noises = {layer_name: layer_to_noise_statistics[layer_name]['Noise'].mean(axis=2) for layer_name in layer_names}
        signals = {layer_name: layer_to_noise_statistics[layer_name]['Signal'].mean(axis=2) for layer_name in
                   layer_names}
        distorted_losses = {layer_name: layer_to_noise_statistics[layer_name]['Distorted Loss'].mean(axis=2) for
                            layer_name in layer_names}

        return noises, signals, distorted_losses

    def dump_random_channel_stats(self, channel_distortion_path, out_path, num_channels, general_seed, num_of_sample_points, block_to_test=None):
        block_size_spec_file_name = None

        os.makedirs(out_path, exist_ok=True)
        input_block_name = "conv1"
        output_block_name = "decode"

        layer_names = self.params.LAYER_NAMES

        assert len(set([len(set(tuple(x) for x in self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])) for layer_name in layer_names])) == 1

        num_of_blocks = len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_names[0]])
        index_dict = {tuple(block_size): index for index, block_size in enumerate(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_names[0]])}
        channel_order_to_layer, channel_order_to_channel, channel_order_to_dim = get_channel_order_statistics(params)

        noises, signals, distorted_losses = self.get_stats(channel_distortion_path, layer_names)

        channels_to_use = get_channels_component(params, group=0, group_size=num_channels, seed=general_seed, shuffle=True)
        channels_to_use.sort()
        if block_to_test is not None:
            relevant_block_sizes = [block_index for block_index, block_size in enumerate(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_names[0]]) if block_size in block_to_test]
        else:
            relevant_block_sizes = num_of_blocks
        dims = np.array([channel_order_to_dim[channel_order] for channel_order in channels_to_use])
        for seed in tqdm(range(num_of_sample_points)):

            if block_size_spec_file_name is not None:
                block_size_spec = pickle.load(open(block_size_spec_file_name, "rb"))
                block_size_indices_spec = {layer_name: np.zeros((self.params.LAYER_NAME_TO_DIMS[layer_name][0],), dtype=np.uint8) for layer_name in layer_names}
                for layer_name in block_size_spec:
                    for block_size_index, block_size in enumerate(block_size_spec[layer_name]):
                        block_size_indices_spec[layer_name][block_size_index] = index_dict[tuple(block_size)]
            else:
                block_size_spec = {layer_name: np.ones((self.params.LAYER_NAME_TO_DIMS[layer_name][0], 2), dtype=np.uint8) for layer_name in layer_names}
                block_size_indices_spec = {layer_name: np.zeros((self.params.LAYER_NAME_TO_DIMS[layer_name][0],), dtype=np.uint8) for layer_name in layer_names}

            np.random.seed(general_seed)
            block_sizes_to_use = np.random.choice(relevant_block_sizes, size=num_channels, replace=True)

            np.random.seed(seed)
            b_128 = block_sizes_to_use[dims == 128]
            b_64 = block_sizes_to_use[dims == 64]
            b_32 = block_sizes_to_use[dims == 32]
            b_1 = block_sizes_to_use[dims == 1]
            np.random.shuffle(b_128)
            np.random.shuffle(b_64)
            np.random.shuffle(b_32)
            np.random.shuffle(b_1)
            block_sizes_to_use = np.zeros_like(block_sizes_to_use)
            block_sizes_to_use[dims == 128] = b_128
            block_sizes_to_use[dims == 64] = b_64
            block_sizes_to_use[dims == 32] = b_32
            block_sizes_to_use[dims == 1] = b_1

            for index, channel_order in enumerate(channels_to_use):
                layer_name = channel_order_to_layer[channel_order]
                channel = channel_order_to_channel[channel_order]

                assert len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]) == num_of_blocks

                block_size_index = block_sizes_to_use[index]

                block_size = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name][block_size_index]
                block_size_indices_spec[layer_name][channel] = block_size_index
                assert np.all(block_size_spec[layer_name][channel] == [1,1])
                block_size_spec[layer_name][channel] = block_size

            cur_assets = self.distortion_utils.get_batch_distortion(
                baseline_block_size_spec=dict(),
                block_size_spec=block_size_spec,
                batch_index=self.device,
                batch_size=16,
                input_block_name=input_block_name,
                output_block_name=output_block_name)

            noise = np.hstack([noises[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            signal = np.hstack([signals[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            distorted_loss = np.hstack([distorted_losses[layer_name][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]] for layer_name in layer_names])
            loss = cur_assets['Distorted Loss'].mean()
            relus_count = sum([sum([get_num_relus(tuple(block_size), self.params.LAYER_NAME_TO_DIMS[layer_name][1]) for block_size in block_size_spec[layer_name]]) for layer_name in layer_names])

            noise = noise[channels_to_use]
            signal = signal[channels_to_use]
            distorted_loss = distorted_loss[channels_to_use]
            loss = loss
            relus_count = relus_count
            np.save(file=os.path.join(out_path, f"noise_{seed}_{self.device }.npy"), arr=noise)
            np.save(file=os.path.join(out_path, f"signal_{seed}_{self.device }.npy"), arr=signal)
            np.save(file=os.path.join(out_path, f"loss_{seed}_{self.device }.npy"), arr=loss)
            np.save(file=os.path.join(out_path, f"relus_count_{seed}_{self.device }.npy"), arr=relus_count)
            np.save(file=os.path.join(out_path, f"distorted_loss_{seed}_{self.device }.npy"), arr=distorted_loss)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="ade_20k_256x256")
    parser.add_argument('--config', type=str, default="/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py")
    parser.add_argument('--checkpoint', type=str, default=f"/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth")

    args = parser.parse_args()

    gpu_id = args.gpu_id
    params = ParamsFactory()("MobileNetV2_256_Params_1_Groups_mini")
    params.DATASET = args.dataset
    params.CONFIG = args.config
    params.CHECKPOINT = args.checkpoint

    chd = RandomDistortionHandler(gpu_id=gpu_id, output_path=None, params=params)

    # for seed in range(100):
    #     for channel_ratio in [0.1,0.25,0.33,0.5,0.9,1.0]:
    #         num_channels = int(channel_ratio * get_num_of_channels(params))
    #
    #         channel_distortion_path = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/one_group_mini/channel_distortions/"
    #         out_path = f"/home/yakir/Data2/random_channel_stats/small_blocks/{channel_ratio}_{seed}_v2/"
    #         chd.dump_random_channel_stats(channel_distortion_path, out_path, num_channels, seed,
    #                                       num_of_sample_points=1000, block_to_test=[[1, 1], [1, 2], [2, 1], [2, 2]])


    for seed in range(100):
        # for channel_ratio in [0.1,0.25,0.33,0.5,0.9,1.0]:
        for channel_ratio in [0.2]:
            num_channels = int(channel_ratio * get_num_of_channels(params))
            # /home/yakir/Data2/random_channel_stats/baseline/random_channel_stats_{channel_ratio}_{seed}_v2
            channel_distortion_path = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/one_group_mini/channel_distortions/"
            out_path = f"/home/yakir/Data2/random_channel_stats/baseline/random_channel_stats_{channel_ratio}_{seed}_v2/"
            chd.dump_random_channel_stats(channel_distortion_path, out_path, num_channels, seed, num_of_sample_points = 1000)
