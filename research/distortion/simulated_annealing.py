import argparse
import copy
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import pickle
import mmcv

from research.distortion.parameters.factory import param_factory
from research.distortion.distortion_utils import DistortionUtils, get_block_spec_num_relus, get_num_relus
from research.distortion.utils import get_channel_order_statistics, get_num_of_channels
from research.distortion.utils import get_channels_subset
from research.mmlab_extension.classification.resnet import MyResNet  # TODO: why is this needed?

class SimulatedAnnealingHandler:
    def __init__(self, gpu_id, params, cfg, input_block_spec_path, output_block_spec_path):

        self.params = params
        self.cfg = cfg
        self.distortion_utils = DistortionUtils(gpu_id=gpu_id, params=self.params, cfg=self.cfg)

        self.keys = ["Noise", "Signal"]
        self.block_size_spec = pickle.load(open(input_block_spec_path, "rb"))
        self.output_block_spec_path = output_block_spec_path
        self.channel_order_to_layer, self.channel_order_to_channel, self.channel_order_to_dim = get_channel_order_statistics(self.params)

        self.num_channels = get_num_of_channels(self.params)
        self.num_of_drelus = get_block_spec_num_relus(self.block_size_spec, self.params)

        self.dim_to_channels = {dim: np.argwhere(self.channel_order_to_dim == dim)[:,0] for dim in np.unique(self.channel_order_to_dim)}

    def get_sibling_channels(self):
        random_channel_a = np.random.choice(self.num_channels)
        channels_b = self.dim_to_channels[self.channel_order_to_dim[random_channel_a]]
        random_channel_b = np.random.choice(channels_b)
        return random_channel_a, random_channel_b

    # def single_channel_proposal(self):
    #
    #     random_channel = np.random.choice(self.num_channels)
    #     layer_name = self.channel_order_to_layer[random_channel]
    #     block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]
    #     channel = self.channel_order_to_channel[random_channel]
    #     old_block_size_index = np.argwhere((self.block_size_spec[layer_name][channel] == block_sizes).all(axis=1))[0,0]
    #     old_block_size = block_sizes[old_block_size_index]
    #     new_block_size_index = old_block_size_index
    #
    #     while new_block_size_index == old_block_size_index:
    #         min_val = max(0, old_block_size_index - 4)
    #         max_val = min(len(block_sizes) - 1, old_block_size_index + 4)
    #         new_block_size_index = np.random.randint(min_val, max_val)
    #
    #     new_block_size = block_sizes[new_block_size_index]
    #     # print(old_block_size, new_block_size)
    #     layer_num_channels = self.params.LAYER_NAME_TO_DIMS[layer_name][0]
    #     if channel >= layer_num_channels:
    #         assert False
    #     return layer_name, channel, new_block_size
    #
    def get_suggested_block_size(self, iteration):
        if iteration == 0:
            return self.block_size_spec
        while True:
            suggest_block_size_spec = copy.deepcopy(self.block_size_spec)
            sibling_channel_a, sibling_channel_b = self.get_sibling_channels()
            layer_name_a = self.channel_order_to_layer[sibling_channel_a]
            layer_name_b = self.channel_order_to_layer[sibling_channel_b]
            channel_a = self.channel_order_to_channel[sibling_channel_a]
            channel_b = self.channel_order_to_channel[sibling_channel_b]

            if not np.all(suggest_block_size_spec[layer_name_a][channel_a] == suggest_block_size_spec[layer_name_b][channel_b]):

                tmp = suggest_block_size_spec[layer_name_a][channel_a]
                suggest_block_size_spec[layer_name_a][channel_a] = suggest_block_size_spec[layer_name_b][channel_b]
                suggest_block_size_spec[layer_name_b][channel_b] = tmp
                return suggest_block_size_spec
    # def get_suggested_block_size(self, iteration):
    #     suggest_block_size_spec = copy.deepcopy(self.block_size_spec)
    #     layer_name, channel, new_block_size = self.single_channel_proposal()
    #     suggest_block_size_spec[layer_name][channel] = new_block_size
    #
    #     while True:
    #         layer_name, channel, new_block_size = self.single_channel_proposal()
    #         old_block_size = suggest_block_size_spec[layer_name][channel]
    #         suggest_block_size_spec[layer_name][channel] = new_block_size
    #
    #         if get_block_spec_num_relus(suggest_block_size_spec, params) == self.num_of_drelus:
    #             break
    #         else:
    #             suggest_block_size_spec[layer_name][channel] = old_block_size
    #     return suggest_block_size_spec

    def get_batch_size(self, iteration):
        return 128

    def get_batch_index(self, iteration):
        return 0
        batch_size = self.get_batch_size(iteration)
        return np.random.choice(len(self.distortion_utils.dataset) // batch_size)

    def extract_deformation_channel_ord(self, iteations):
        baseline_block_size_spec = dict()
        first_layer = self.params.LAYER_NAMES[0]
        input_block_name = self.params.LAYER_NAME_TO_BLOCK_NAME[first_layer]
        output_block_name = self.params.BLOCK_NAMES[-2]

        noise = np.inf
        for iteration in tqdm(range(iteations)):
            batch_size = self.get_batch_size(iteration)
            batch_index = self.get_batch_index(iteration)
            suggest_block_size_spec = self.get_suggested_block_size(iteration)

            cur_assets = self.distortion_utils.get_batch_distortion(
                baseline_block_size_spec=baseline_block_size_spec,
                block_size_spec=suggest_block_size_spec,
                batch_index=batch_index,
                batch_size=batch_size,
                input_block_name=input_block_name,
                output_block_name=output_block_name)

            suggest_block_size_noise = cur_assets["Noise"].mean()

            if suggest_block_size_noise < noise:
                self.block_size_spec = suggest_block_size_spec
                noise = suggest_block_size_noise
                pickle.dump(obj=suggest_block_size_spec, file=open(self.output_block_spec_path, "wb"))
                print(noise)



if __name__ == "__main__":
    checkpoint = "/home/yakir/epoch_14.pth"
    input_block_spec_path = "/home/yakir/block_size_spec_4x4_algo.pickle"
    output_block_spec_path = "/home/yakir/block_size_spec_4x4_algo_out.pickle"
    config = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k_finetune_0.0001_avg_pool.py"

    # block_size_spec = pickle.load(open(input_block_spec, 'rb'))
    gpu_id = 0

    cfg = mmcv.Config.fromfile(config)
    gpu_id = gpu_id
    params = param_factory(cfg)

    params.CHECKPOINT = checkpoint

    SimulatedAnnealingHandler(gpu_id=gpu_id,
                              params=params,
                              cfg=cfg,
                              input_block_spec_path=input_block_spec_path,
                              output_block_spec_path=output_block_spec_path).extract_deformation_channel_ord(iteations=1000000)