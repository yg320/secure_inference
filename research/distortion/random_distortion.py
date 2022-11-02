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

    def foo(self):
        noises = []
        losses = []
        reductions = []

        layer_names = params.LAYER_NAMES[:34]

        content = {layer_name:pickle.load(open(f"/home/yakir/Data2/assets_v4/distortions/ade_20k/MobileNetV2_256/2_groups_160k/channel_distortions/{layer_name}_0.pickle", 'rb')) for layer_name in layer_names}

        # block_size_orig_order = np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES["conv1"])
        block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES["conv1"]
        # np.all(block_sizes == block_size_orig_order)
        # [np.argwhere(np.all(cur_block_size == block_size_orig_order, axis=1))[0,0] for cur_block_size in block_sizes]

        ratios = np.array([1 / x[0] / x[1] for x in block_sizes])
        relus = np.array([np.prod(params.LAYER_NAME_TO_DIMS[layer_name]) for layer_name in layer_names])
        total_relus = relus.sum()
        num_layers = len(layer_names)

        for _ in tqdm(range(100000)):

            iterations = np.random.randint(10000, 1000000)
            block_sizes_index_spec = [np.zeros((self.params.LAYER_NAME_TO_DIMS[layer_name][0],), dtype=np.uint8) for layer_name in layer_names]

            layer_indices = np.random.randint(low=0, high=num_layers, size=iterations)

            for iteration in range(iterations):
                layer_index = layer_indices[iteration]
                channel = np.random.randint(block_sizes_index_spec[layer_index].shape[0])
                cur_value = block_sizes_index_spec[layer_index][channel]
                block_sizes_index_spec[layer_index][channel] = min(cur_value + 1, len(block_sizes) -1) #np.random.randint(cur_value, len(block_sizes))
            reduction = sum(ratios[block_sizes_index_spec[layer_index]].mean() * relus[layer_index] for layer_index in range(num_layers)) / total_relus

            block_size_spec = {layer_name:np.array(block_sizes)[ block_sizes_index_spec[layer_index]] for layer_index, layer_name in enumerate(layer_names)}

            input_block_name = "conv1"
            output_block_name = "decode"


            cur_assets = self.distortion_utils.get_batch_distortion(
                baseline_block_size_spec=dict(),
                block_size_spec=block_size_spec,
                batch_index=0,
                batch_size=16,
                input_block_name=input_block_name,
                output_block_name=output_block_name)
            additive_noise = sum([content[layer_name]["Noise"][np.arange(params.LAYER_NAME_TO_DIMS[layer_name][0]), block_sizes_index_spec[layer_index]].sum(axis=0).mean() for layer_index, layer_name in enumerate(layer_names)])

            noises.append(additive_noise)
            losses.append(cur_assets['Distorted Loss'].mean())
            reductions.append(reduction)

if __name__ == "__main__":
    #         steps = 160
    #         checkpoint = f"/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_m-v2-d8_256x256_160k_ade20k/iter_{steps}000.pth"
    #         base_output_path = f"/home/yakir/Data2/assets_v4/distortions/{dataset}/MobileNetV2_256/2_groups_{steps}k"
    #         ratio = 0.08333333333333333
    #         output_path = os.path.join(base_output_path, "channel_distortions")
    #         block_size_spec_file_name = os.path.join(base_output_path, "block_spec.pickle")
    #         script_path = "/home/yakir/PycharmProjects/secure_inference/research/distortion/channel_distortion.py"
    #         knapsack_script_path = "/home/yakir/PycharmProjects/secure_inference/research/knapsack/multiple_choice_knapsack.py"
    #
    #         for iteration in range(1):
    #             jobs = []
    #             for gpu_id in range(2):
    #
    #                 job_params = [
    #                     "--batch_index", f"{gpu_id}",
    #                     "--gpu_id", f"{gpu_id}",
    #                     "--dataset", dataset,
    #                     "--config", config,
    #                     "--checkpoint", checkpoint,
    #                     "--iter", f"{iteration}",
    #                     "--block_size_spec_file_name", block_size_spec_file_name,
    #                     "--output_path", output_path
    #                 ]
    #                 jobs.append(["python", script_path] + job_params)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="ade_20k")
    parser.add_argument('--config', type=str, default="/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/deeplabv3_m-v2-d8_256x256_160k_ade20k.py")
    parser.add_argument('--checkpoint', type=str, default=f"/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_m-v2-d8_256x256_160k_ade20k/iter_160000.pth")
    args = parser.parse_args()

    gpu_id = args.gpu_id
    params = MobileNetV2_256_Params_2_Groups()
    params.DATASET = args.dataset
    params.CONFIG = args.config
    params.CHECKPOINT = args.checkpoint


    chd = RandomDistortionHandler(gpu_id=gpu_id,
                                   output_path=None,
                                   params=params)

    chd.foo()
