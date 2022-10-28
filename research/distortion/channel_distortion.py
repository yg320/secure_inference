import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
from typing import Dict, List
import glob
import pickle
import shutil

from research.block_relu.consts import TARGET_REDUCTIONS
from research.block_relu.utils import get_model, get_data, center_crop, ArchUtilsFactory
from research.block_relu.params import ParamsFactory, MobileNetV2_256_Params
from research.distortion.distortion_utils import DistortionUtils
from research.pipeline.backbones.secure_resnet import MyResNet  # TODO: find better way to init

from mmseg.ops import resize
import torch.nn.functional as F
from mmseg.core import intersect_and_union
import pandas as pd

class ChannelDistortionHandler:
    def __init__(self, gpu_id, output_path):

        self.params = MobileNetV2_256_Params()
        self.params.DATASET = "ade_20k"
        self.params.CONFIG = "/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/deeplabv3_m-v2-d8_256x256_160k_ade20k.py"
        self.params.CHECKPOINT = "/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_m-v2-d8_256x256_160k_ade20k/iter_160000.pth"
        self.distortion_utils = DistortionUtils(gpu_id=gpu_id, params=self.params)
        self.output_path = output_path

        self.keys = ["Noise", "Signal", "Distorted Loss", "Baseline Loss"]

    def extract_deformation_by_blocks(self,
                                      batch_index: int,
                                      layer_names: List[str],
                                      batch_size: int,
                                      baseline_block_size_spec: Dict[str, np.array]):

        os.makedirs(self.output_path, exist_ok=True)

        for layer_name in layer_names:

            block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]
            layer_num_channels = self.params.LAYER_NAME_TO_DIMS[layer_name][0]
            input_block_name = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
            output_block_name = self.params.IN_LAYER_PROXY_SPEC[layer_name]

            layer_assets = {key: np.zeros(shape=(layer_num_channels, len(block_sizes), batch_size)) for key in self.keys}
            for channel in tqdm(range(layer_num_channels), desc=f"Batch={batch_index} Layer={layer_name}"):
                for block_size_index, block_size in enumerate(block_sizes):
                    layer_block_sizes = np.ones(shape=(layer_num_channels, 2), dtype=np.int32)  # TODO: reuse
                    layer_block_sizes[channel] = block_size
                    block_size_spec = {layer_name: layer_block_sizes}

                    cur_assets = self.distortion_utils.get_batch_distortion(
                        baseline_block_size_spec=baseline_block_size_spec,
                        block_size_spec=block_size_spec,
                        batch_index=batch_index,
                        batch_size=batch_size,
                        input_block_name=input_block_name,
                        output_block_name=output_block_name)

                    for key in self.keys:
                        layer_assets[key][channel, block_size_index] = cur_assets[key]

            file_name = os.path.join(self.output_path, f"{layer_name}_{batch_index}.pickle")
            with open(file_name, "wb") as file:
                pickle.dump(obj=layer_assets, file=file)

if __name__ == "__main__":
    # layer_names = [
    #     "conv1",
    #     "layer1_0_0",
    #     "layer2_0_0",
    #     "layer2_0_1",
    #     "layer2_1_0",
    #     "layer2_1_1",
    #     "layer3_0_0",
    #     "layer3_0_1",
    #     "layer3_1_0",
    #     "layer3_1_1",
    #     "layer3_2_0",
    #     "layer3_2_1",
    # ]
    # baseline_block_size_spec = dict()

    # layer_names = [
    #     "layer4_0_0",
    #     "layer4_0_1",
    #     "layer4_1_0",
    #     "layer4_1_1",
    #     "layer4_2_0",
    #     "layer4_2_1",
    #     "layer4_3_0",
    #     "layer4_3_1",
    #     "layer5_0_0",
    #     "layer5_0_1",
    #     "layer5_1_0",
    #     "layer5_1_1",
    #     "layer5_2_0",
    #     "layer5_2_1",
    # ]
    # baseline_block_size_spec = pickle.load(open('/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_iter_0_0.0833.pickle', 'rb'))

    # layer_names = [
    #     "layer6_0_0",
    #     "layer6_0_1",
    #     "layer6_1_0",
    #     "layer6_1_1",
    #     "layer6_2_0",
    #     "layer6_2_1",
    #     "layer7_0_0",
    #     "layer7_0_1"
    # ]
    #
    # baseline_block_size_spec = pickle.load(open('/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_iter_01_0.0833.pickle', 'rb'))

    layer_names = [
        'decode_0',
        # 'decode_1',
        # 'decode_2',
        # 'decode_3',
        # 'decode_4',
        # 'decode_5'
    ]

    baseline_block_size_spec = pickle.load(open('/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_iter_012_0.0833.pickle', 'rb'))


    chd = ChannelDistortionHandler(gpu_id=0, output_path="/home/yakir/Data2/assets_v4/distortions/ade_20k/MobileNetV2_256/channels_distortion")
    chd.extract_deformation_by_blocks(batch_index=0,
                                      layer_names=layer_names,
                                      batch_size=16,
                                      baseline_block_size_spec=baseline_block_size_spec)
