import argparse
import copy
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import pickle

from research.parameters.factory import ParamsFactory
from research.distortion.distortion_utils import DistortionUtils


class ChannelDistortionHandler:
    def __init__(self, gpu_id, output_path, params):

        self.params = params
        self.distortion_utils = DistortionUtils(gpu_id=gpu_id, params=self.params)
        self.output_path = output_path

        self.keys = ["Noise", "Signal", "Distorted Loss", "Baseline Loss"]

    def extract_deformation_channel_v2(self,
                                       batch_index: int,
                                       layer_names: List[str],
                                       batch_size: int,
                                       baseline_block_size_spec: Dict[str, np.array],
                                       seed: int = 123,
                                       num_channels_to_run: int = None):

        np.random.seed(seed)
        total_num_channels = sum([self.params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in self.params.LAYER_NAMES])
        channel_order_to_channel_in_layer_index = np.hstack([np.arange(self.params.LAYER_NAME_TO_DIMS[layer_name][0]) for layer_name in layer_names])
        channel_order_to_layer = np.hstack([[layer_name] * self.params.LAYER_NAME_TO_DIMS[layer_name][0] for layer_name in layer_names])

        all_channels = np.arange(total_num_channels)
        np.random.shuffle(all_channels)
        channels_to_use = all_channels[:num_channels_to_run]

        channels_to_run = {layer_name: [] for layer_name in self.params.LAYER_NAMES}
        for channel_order in channels_to_use:
            layer_name = channel_order_to_layer[channel_order]
            channel_in_layer_index = channel_order_to_channel_in_layer_index[channel_order]
            channels_to_run[layer_name].append(channel_in_layer_index)

        for layer_name in self.params.LAYER_NAMES:
            channels_to_run[layer_name].sort()

        os.makedirs(self.output_path, exist_ok=True)

        for layer_name in layer_names:
            file_name = os.path.join(self.output_path, f"{layer_name}_{batch_index}.pickle")
            if os.path.exists(file_name):
                continue
            block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]
            layer_num_channels = self.params.LAYER_NAME_TO_DIMS[layer_name][0]
            input_block_name = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
            output_block_name = self.params.IN_LAYER_PROXY_SPEC[layer_name]

            layer_assets = {key: np.zeros(shape=(layer_num_channels, len(block_sizes), batch_size)) for key in
                            self.keys}

            block_size_spec = copy.deepcopy(baseline_block_size_spec)

            if layer_name not in block_size_spec:
                block_size_spec[layer_name] = np.ones(shape=(layer_num_channels, 2), dtype=np.int32)

            for channel in tqdm(channels_to_run[layer_name], desc=f"Batch={batch_index} Layer={layer_name}"):
                for block_size_index, block_size in enumerate(block_sizes):
                    if layer_name == "decode_0":
                        if np.prod(block_size) > 1:
                            continue

                    orig_block_size = block_size_spec[layer_name][channel].copy()
                    block_size_spec[layer_name][channel] = block_size

                    cur_assets = self.distortion_utils.get_batch_distortion(
                        baseline_block_size_spec=baseline_block_size_spec,
                        block_size_spec=block_size_spec,
                        batch_index=batch_index,
                        batch_size=batch_size,
                        input_block_name=input_block_name,
                        output_block_name=output_block_name)

                    block_size_spec[layer_name][channel] = orig_block_size

                    for key in self.keys:
                        layer_assets[key][channel, block_size_index] = cur_assets[key]

            with open(file_name, "wb") as file:
                pickle.dump(obj=layer_assets, file=file)


    def extract_deformation_by_blocks(self,
                                      batch_index: int,
                                      layer_names: List[str],
                                      batch_size: int,
                                      baseline_block_size_spec: Dict[str, np.array]):

        os.makedirs(self.output_path, exist_ok=True)

        for layer_name in layer_names:
            file_name = os.path.join(self.output_path, f"{layer_name}_{batch_index}.pickle")
            if os.path.exists(file_name):
                continue
            block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]
            layer_num_channels = self.params.LAYER_NAME_TO_DIMS[layer_name][0]
            input_block_name = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
            output_block_name = self.params.IN_LAYER_PROXY_SPEC[layer_name]

            layer_assets = {key: np.zeros(shape=(layer_num_channels, len(block_sizes), batch_size)) for key in
                            self.keys}

            block_size_spec = copy.deepcopy(baseline_block_size_spec)

            if layer_name not in block_size_spec:
                block_size_spec[layer_name] = np.ones(shape=(layer_num_channels, 2), dtype=np.int32)

            for channel in tqdm(range(layer_num_channels), desc=f"Batch={batch_index} Layer={layer_name}"):
                for block_size_index, block_size in enumerate(block_sizes):
                    if layer_name == "decode_0":
                        if np.prod(block_size) > 1:
                            continue

                    orig_block_size = block_size_spec[layer_name][channel].copy()
                    block_size_spec[layer_name][channel] = block_size

                    cur_assets = self.distortion_utils.get_batch_distortion(
                        baseline_block_size_spec=baseline_block_size_spec,
                        block_size_spec=block_size_spec,
                        batch_index=batch_index,
                        batch_size=batch_size,
                        input_block_name=input_block_name,
                        output_block_name=output_block_name)

                    block_size_spec[layer_name][channel] = orig_block_size

                    for key in self.keys:
                        layer_assets[key][channel, block_size_index] = cur_assets[key]

            with open(file_name, "wb") as file:
                pickle.dump(obj=layer_assets, file=file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--batch_index', type=int)
    # parser.add_argument('--gpu_id', type=int)
    # parser.add_argument('--dataset', type=str)
    # parser.add_argument('--config', type=str)
    # parser.add_argument('--checkpoint', type=str)
    # parser.add_argument('--iter', type=int)
    # parser.add_argument('--block_size_spec_file_name', type=str, default=None)
    # parser.add_argument('--output_path', type=str, default=None)
    # parser.add_argument('--params_name', type=str, default=None)

    parser.add_argument('--batch_index', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="ade_20k_256x256")
    parser.add_argument('--config', type=str, default="/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py")
    parser.add_argument('--checkpoint', type=str, default="/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth")
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--block_size_spec_file_name', type=str, default=None)
    parser.add_argument('--output_path', type=str, default="/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/one_group/channel_distortions")
    parser.add_argument('--params_name', type=str, default="MobileNetV2_256_Params_1_Groups")
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    params = ParamsFactory()(args.params_name)
    params.DATASET = args.dataset
    params.CONFIG = args.config
    params.CHECKPOINT = args.checkpoint

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    iteration = args.iter
    layer_names = params.LAYER_GROUPS[iteration]

    if args.block_size_spec_file_name and os.path.exists(args.block_size_spec_file_name):
        baseline_block_size_spec = pickle.load(open(args.block_size_spec_file_name, 'rb'))
    else:
        baseline_block_size_spec = dict()

    chd = ChannelDistortionHandler(gpu_id=gpu_id,
                                   output_path=output_path,
                                   params=params)

    chd.extract_deformation_by_blocks(batch_index=args.batch_index,
                                      layer_names=layer_names,
                                      batch_size=args.batch_size,
                                      baseline_block_size_spec=baseline_block_size_spec)
