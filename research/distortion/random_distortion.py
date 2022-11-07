import matplotlib
matplotlib.use("TkAgg")
import argparse
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import pickle

from research.parameters.base import MobileNetV2_256_Params_2_Groups
from research.distortion.distortion_utils import DistortionUtils

from matplotlib import pyplot as plt

# def
class RandomDistortionHandler:
    def __init__(self, gpu_id, output_path, params):

        self.params = params
        self.distortion_utils = DistortionUtils(gpu_id=gpu_id, params=self.params)
        self.output_path = output_path

        self.keys = ["Noise", "Signal", "Distorted Loss", "Baseline Loss"]

    def get_current_reduction(self, block_size_spec):
        return sum(np.prod(params.LAYER_NAME_TO_DIMS[layer_name]) * np.sum(
            1 / np.prod(block_size_spec[layer_name], axis=1)) / len(block_size_spec[layer_name]) for layer_name in
            block_size_spec.keys()) / sum(
            np.prod(params.LAYER_NAME_TO_DIMS[layer_name]) for layer_name in block_size_spec.keys())

    def foo_bar(self):
        noises = []
        losses = []
        reductions = []
        num_channels_to_noise = []

        input_block_name = "conv1"
        output_block_name = "decode"
        layer_names = self.params.LAYER_NAMES[7:34]
        relus = {layer_name: np.prod(params.LAYER_NAME_TO_DIMS[layer_name]) for layer_name in layer_names}
        total_relus = sum(relus.values())

        tot_channel_count = sum(self.params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names)
        channel_order_to_channel = np.hstack([np.arange(self.params.LAYER_NAME_TO_DIMS[layer_name][0]) for layer_name in layer_names])
        channel_order_to_layer = np.hstack([[layer_name] * self.params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names])
        layer_to_noise_statistics = {layer_name:pickle.load(open(f"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_0.0625/channel_distortions/{layer_name}_0.pickle", 'rb')) for layer_name in layer_names}


        for i in tqdm(range(100000)):

            block_size_spec = {layer_name: np.ones((self.params.LAYER_NAME_TO_DIMS[layer_name][0], 2), dtype=np.uint8) for layer_name in layer_names}
            block_size_indices_spec = {layer_name: np.zeros((self.params.LAYER_NAME_TO_DIMS[layer_name][0],), dtype=np.uint8) for layer_name in layer_names}

            num_of_channels_to_noise = 5000 #np.random.randint(tot_channel_count)
            np.random.seed(i)
            channels_to_noise = np.random.choice(tot_channel_count, size=num_of_channels_to_noise, replace=False)

            np.random.seed(123)
            for channel_order in channels_to_noise:
                layer_name = channel_order_to_layer[channel_order]
                channel = channel_order_to_channel[channel_order]
                block_size_index = np.random.randint(len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]))
                block_size = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name][block_size_index]
                block_size_indices_spec[layer_name][channel] = block_size_index
                block_size_spec[layer_name][channel] = block_size

            cur_assets = self.distortion_utils.get_batch_distortion(
                baseline_block_size_spec=dict(),
                block_size_spec=block_size_spec,
                batch_index=0,
                batch_size=16,
                input_block_name=input_block_name,
                output_block_name=output_block_name)

            additive_noise = sum([layer_to_noise_statistics[layer_name]["Noise"][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]].sum(axis=0).mean() for layer_name in layer_names])
            reduction = sum(np.mean(1/np.prod(block_size_spec[layer_name], axis=1)) * relus[layer_name] for layer_name in layer_names) / total_relus
            noises.append(additive_noise)
            losses.append(cur_assets['Distorted Loss'].mean())
            reductions.append(reduction)
            num_channels_to_noise.append(num_of_channels_to_noise)


    def bar(self):
        noises = []
        losses = []

        input_block_name = "conv1"
        output_block_name = "decode"
        layer_names = ['layer3_0_1',
                       'layer3_1_0',
                       'layer3_1_1',
                       'layer3_2_0',
                       'layer3_2_1',
                       'layer4_0_0',
                       'layer4_0_1',
                       'layer4_1_0',
                       'layer4_1_1',
                       'layer4_2_0',
                       'layer4_2_1',
                       'layer4_3_0',
                       'layer4_3_1',
                       'layer5_0_0',
                       'layer5_0_1',
                       'layer5_1_0',
                       'layer5_1_1',
                       'layer5_2_0',
                       'layer5_2_1',
                       'layer6_0_0',
                       'layer6_0_1',
                       'layer6_1_0',
                       'layer6_1_1',
                       'layer6_2_0',
                       'layer6_2_1',
                       'layer7_0_0',
                       'layer7_0_1',
                       ]
        layer_noises = {layer_name:pickle.load(open(f"/home/yakir/Data2/assets_v4/distortions/ade_20k/MobileNetV2_256/2_groups_160k/channel_distortions/{layer_name}_0.pickle", 'rb')) for layer_name in layer_names}


        for _ in tqdm(range(100000)):
            num_channels_to_noise = np.random.randint(0,20000)
            block_sizes = np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES["conv1"])
            block_sizes_indices_to_use = np.random.choice(len(block_sizes), size=num_channels_to_noise, replace=True)

            block_sizes_to_use = block_sizes[block_sizes_indices_to_use]
            block_size_spec = {layer_name: np.ones((self.params.LAYER_NAME_TO_DIMS[layer_name][0], 2), dtype=np.uint8) for layer_name in layer_names}
            block_size_indices_spec = {layer_name: np.zeros((self.params.LAYER_NAME_TO_DIMS[layer_name][0],), dtype=np.uint8) for layer_name in layer_names}

            for index in range(num_channels_to_noise):
                layer_name = np.random.choice(layer_names)
                channel = np.random.randint(params.LAYER_NAME_TO_DIMS[layer_name][0])
                block_size_indices_spec[layer_name][channel] = block_sizes_indices_to_use[index]
                block_size_spec[layer_name][channel] = block_sizes_to_use[index]

            cur_assets = self.distortion_utils.get_batch_distortion(
                baseline_block_size_spec=dict(),
                block_size_spec=block_size_spec,
                batch_index=0,
                batch_size=16,
                input_block_name=input_block_name,
                output_block_name=output_block_name)

            additive_noise = sum([layer_noises[layer_name]["Noise"][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_size_indices_spec[layer_name]].sum(axis=0).mean() for layer_name in layer_names])

            noises.append(additive_noise)
            losses.append(cur_assets['Distorted Loss'].mean())
            np.random.shuffle(block_sizes_indices_to_use)

    def foo(self):
        noises = []
        losses = []
        reductions = []

        layer_names = ['layer3_0_1',
                       'layer3_1_0',
                       'layer3_1_1',
                       'layer3_2_0',
                       'layer3_2_1',
                       'layer4_0_0',
                       'layer4_0_1',
                       'layer4_1_0',
                       'layer4_1_1',
                       'layer4_2_0',
                       'layer4_2_1',
                       'layer4_3_0',
                       'layer4_3_1',
                       'layer5_0_0',
                       'layer5_0_1',
                       'layer5_1_0',
                       'layer5_1_1',
                       'layer5_2_0',
                       'layer5_2_1',
                       'layer6_0_0',
                       'layer6_0_1',
                       'layer6_1_0',
                       'layer6_1_1',
                       'layer6_2_0',
                       'layer6_2_1',
                       'layer7_0_0',
                       'layer7_0_1',
                       ]


        content = {layer_name:pickle.load(open(f"/home/yakir/Data2/assets_v4/distortions/ade_20k/MobileNetV2_256/2_groups_160k/channel_distortions/{layer_name}_0.pickle", 'rb')) for layer_name in layer_names}

        # block_size_orig_order = np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES["conv1"])
        block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES["conv1"]
        # np.all(block_sizes == block_size_orig_order)
        # [np.argwhere(np.all(cur_block_size == block_size_orig_order, axis=1))[0,0] for cur_block_size in block_sizes]

        ratios = np.array([1 / x[0] / x[1] for x in block_sizes])
        relus = np.array([np.prod(params.LAYER_NAME_TO_DIMS[layer_name]) for layer_name in layer_names])
        total_relus = relus.sum()
        num_layers = len(layer_names)

        block_size_spec = pickle.load(open("/home/yakir/Data2/assets_v4/distortions/ade_20k/MobileNetV2_256/2_groups_160k/block_spec.pickle", 'rb'))
        block_sizes_index_spec = {layer_name:np.argwhere(np.all((block_size_spec[layer_name][:,np.newaxis] == np.array(block_sizes)), axis=2))[:,1] for layer_name in block_size_spec.keys()}
        for _ in tqdm(range(100000)):

            # # iterations = np.random.randint(10000, 1000000)
            # iterations = 6000
            # block_sizes_index_spec = [np.zeros((self.params.LAYER_NAME_TO_DIMS[layer_name][0],), dtype=np.uint8) for layer_name in layer_names]
            #
            # layer_indices = np.random.randint(low=0, high=num_layers, size=iterations)
            #
            # for iteration in range(iterations):
            #     layer_index = layer_indices[iteration]
            #     channel = np.random.randint(block_sizes_index_spec[layer_index].shape[0])
            #     cur_value = block_sizes_index_spec[layer_index][channel]
            #     block_sizes_index_spec[layer_index][channel] = min(cur_value + 1, len(block_sizes) -1) #np.random.randint(cur_value, len(block_sizes))
            # reduction = sum(ratios[block_sizes_index_spec[layer_index]].mean() * relus[layer_index] for layer_index in range(num_layers)) / total_relus
            #
            block_size_spec = {layer_name:np.array(block_sizes)[ block_sizes_index_spec[layer_name]] for layer_name in layer_names}

            input_block_name = "conv1"
            output_block_name = "decode"


            cur_assets = self.distortion_utils.get_batch_distortion(
                baseline_block_size_spec=dict(),
                block_size_spec=block_size_spec,
                batch_index=0,
                batch_size=16,
                input_block_name=input_block_name,
                output_block_name=output_block_name)
            additive_noise = sum([content[layer_name]["Noise"][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_sizes_index_spec[layer_name]].sum(axis=0).mean() for  layer_name in layer_names])

            noises.append(additive_noise)
            losses.append(cur_assets['Distorted Loss'].mean())

            for k in block_sizes_index_spec.keys():
                np.random.shuffle(block_sizes_index_spec[k])
            # reductions.append(reduction)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="ade_20k")
    parser.add_argument('--config', type=str, default="/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py")
    parser.add_argument('--checkpoint', type=str, default=f"/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth")
    args = parser.parse_args()

    gpu_id = args.gpu_id
    params = MobileNetV2_256_Params_2_Groups()
    params.DATASET = args.dataset
    params.CONFIG = args.config
    params.CHECKPOINT = args.checkpoint


    chd = RandomDistortionHandler(gpu_id=gpu_id,
                                   output_path=None,
                                   params=params)

    chd.foo_bar()
