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
from research.parameters.base import MobileNetV2_256_Params_2_Groups
from research.distortion.distortion_utils import DistortionUtils
from research.pipeline.backbones.secure_resnet import MyResNet  # TODO: find better way to init

from mmseg.ops import resize
import torch.nn.functional as F
from mmseg.core import intersect_and_union
import pandas as pd

class ChannelDistortionHandler:
    def __init__(self, gpu_id, output_path, params):

        self.params = params
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

    gpu_id = 1
    params = MobileNetV2_256_Params_2_Groups()
    params.DATASET = "ade_20k"
    params.CONFIG = "/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/deeplabv3_m-v2-d8_256x256_160k_ade20k.py"
    params.CHECKPOINT = "/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_m-v2-d8_256x256_160k_ade20k/iter_160000.pth"

    block_size_spec_f_format = "/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_2_groups_iter_{}_0.0833.pickle"
    output_path = "/home/yakir/Data2/assets_v4/distortions/ade_20k/MobileNetV2_256/channels_distortion_2_groups"

    iter_ = 1
    layer_names = params.LAYER_GROUPS[iter_]
    if iter_ == 0:
        baseline_block_size_spec = dict()
    else:
        baseline_block_size_spec = pickle.load(open(block_size_spec_f_format.format(iter_-1), 'rb'))

    if "decode_0" in layer_names:
        layer_names.remove("decode_0")
    chd = ChannelDistortionHandler(gpu_id=gpu_id,
                                   output_path=output_path,
                                   params=params)

    chd.extract_deformation_by_blocks(batch_index=gpu_id,
                                      layer_names=layer_names,
                                      batch_size=16,
                                      baseline_block_size_spec=baseline_block_size_spec)
