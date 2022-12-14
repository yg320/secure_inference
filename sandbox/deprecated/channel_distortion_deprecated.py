import argparse
import copy
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import pickle

from research.parameters.factory import ParamsFactory
from research.distortion.distortion_utils import DistortionUtils
from research.distortion.utils import get_channels_component, get_channel_order_statistics

from research.pipeline.backbones.secure_resnet import AvgPoolResNet

class ChannelDistortionHandler:
    def __init__(self, gpu_id, output_path, params):

        self.params = params
        self.distortion_utils = DistortionUtils(gpu_id=gpu_id, params=self.params)
        self.output_path = output_path

        self.keys = ["Noise", "Signal", "Distorted Loss", "Baseline Loss"]

    def extract_channels_distortion(self,
                                    batch_index: int,
                                    batch_size: int,
                                    baseline_block_size_spec: Dict[str, np.array],
                                    channels: List[int]):

        channel_order_to_layer, channel_order_to_channel, _ = get_channel_order_statistics(params)

        channels_to_run = {layer_name: [] for layer_name in self.params.LAYER_NAMES}

        for channel_order in channels:
            layer_name = channel_order_to_layer[channel_order]
            channel_in_layer_index = channel_order_to_channel[channel_order]
            channels_to_run[layer_name].append(channel_in_layer_index)

        for layer_name in self.params.LAYER_NAMES:
            channels_to_run[layer_name].sort()

        os.makedirs(self.output_path, exist_ok=True)

        block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[self.params.LAYER_NAMES[0]]

        # Remove this flexibility
        for layer_name in self.params.LAYER_NAMES:
            tuplify_block_size = set(tuple(x) for x in block_sizes)
            tuplify_layer_block_size = set(tuple(x) for x in self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])
            assert tuplify_layer_block_size == tuplify_block_size

        for layer_name in self.params.LAYER_NAMES:
            file_name = os.path.join(self.output_path, f"{layer_name}_{batch_index}.pickle")
            if os.path.exists(file_name):
                continue

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
    parser.add_argument('--params_name', type=str, default="ResNet18_Params_192x192")
    parser.add_argument('--dataset', type=str, default="ade_20k_192x192")
    parser.add_argument('--config', type=str, default="/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16.py")
    parser.add_argument('--checkpoint', type=str, default="/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/iter_80000.pth")
    parser.add_argument('--output_path', type=str, default="/home/yakir/Data2/assets_v4/distortions/ade_20k_192x192/ResNet18/channel_distortions")
    parser.add_argument('--block_size_spec_file_name', type=str, default=None)

    parser.add_argument('--group_size', type=int, default=None)
    parser.add_argument('--group_index', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--channel_ord_range', type=tuple, default=(0, 4736))

    args = parser.parse_args()

    gpu_id = args.gpu_id
    params = ParamsFactory()(args.params_name)
    params.DATASET = args.dataset
    params.CONFIG = args.config
    params.CHECKPOINT = args.checkpoint

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    if args.block_size_spec_file_name and os.path.exists(args.block_size_spec_file_name):
        baseline_block_size_spec = pickle.load(open(args.block_size_spec_file_name, 'rb'))
    else:
        baseline_block_size_spec = dict()

    chd = ChannelDistortionHandler(gpu_id=gpu_id,
                                   output_path=output_path,
                                   params=params)

    channels = get_channels_component(params=params,
                                      group=args.group_index,
                                      group_size=args.group_size,
                                      seed=args.seed,
                                      shuffle=args.shuffle,
                                      channel_ord_range=args.channel_ord_range)

    chd.extract_channels_distortion(batch_index=args.batch_index,
                                    batch_size=args.batch_size,
                                    baseline_block_size_spec=baseline_block_size_spec,
                                    channels=channels)
