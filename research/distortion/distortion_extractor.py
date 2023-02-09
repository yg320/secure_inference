import argparse
import copy
import os
from tqdm import tqdm
from typing import Dict
import pickle
import mmcv
import torch
import numpy as np
import contextlib
from functools import lru_cache
import ctypes

from research.distortion.parameters.factory import param_factory
from research.distortion.utils import get_channels_subset
from research.distortion.utils import get_model
from research.utils import build_data
from research.distortion.arch_utils.factory import arch_utils_factory

# TODO: why are these import needed?
from research.mmlab_extension.classification.resnet_cifar_v2 import ResNet_CIFAR_V2
from research.mmlab_extension.classification.resnet import MyResNet
from research.mmlab_extension.transforms import CenterCrop


@contextlib.contextmanager
def model_block_relu_transform(model, relu_spec, arch_utils):
    layer_name_to_orig_layer = {}
    for layer_name, block_sizes in relu_spec.items():
        orig_layer = arch_utils.get_layer(model, layer_name)
        layer_name_to_orig_layer[layer_name] = orig_layer

        arch_utils.set_bReLU_layers(model, {layer_name: block_sizes})

    yield model

    for layer_name_, orig_layer in layer_name_to_orig_layer.items():
        arch_utils.set_layers(model, {layer_name_: orig_layer})


class DistortionUtils:
    def __init__(self, gpu_id, params, checkpoint, cfg, mode, seed=123):

        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.params = params
        self.checkpoint = checkpoint
        self.cfg = cfg
        self.arch_utils = arch_utils_factory(self.cfg)

        self.model = get_model(
            config=self.cfg,
            gpu_id=self.gpu_id,
            checkpoint_path=self.checkpoint
        )

        self.dataset = build_data(self.cfg, mode=mode)

        np.random.seed(seed)
        self.shuffled_indices = np.arange(len(self.dataset))
        np.random.shuffle(self.shuffled_indices)

    def get_loss(self, out, ground_truth):
        if self.cfg.model.type == 'ImageClassifier':
            return self.model.head.loss(out, ground_truth.to(torch.long))['loss'].cpu().numpy()
        elif self.cfg.model.type == 'EncoderDecoder':
            return self.model.decode_head.losses(out, ground_truth)['loss_ce'].cpu().numpy()
        else:
            raise NotImplementedError

    def get_samples(self, batch_index, batch_size):

        batch_indices = np.arange(batch_index * batch_size, batch_index * batch_size + batch_size)
        batch_indices = self.shuffled_indices[batch_indices]

        # TODO: find a more elegant way to do this
        if self.cfg.model.type == 'ImageClassifier':
            batch = torch.stack([self.dataset[sample_id]['img'].data for sample_id in batch_indices]).to(self.device)
            ground_truth = torch.Tensor([self.dataset[sample_id]["gt_label"] for sample_id in batch_indices]).to(self.device)
        elif self.cfg.model.type == 'EncoderDecoder':
            batch = torch.stack([self.dataset[sample_id]['img'][0].data for sample_id in batch_indices]).to(self.device)
            ground_truth = torch.stack([self.dataset[sample_id]['gt_semantic_seg'][0].to(torch.int64) for sample_id in batch_indices]).to(self.device)
        else:
            raise NotImplementedError

        return batch, ground_truth

    @staticmethod
    def get_distortion(block_name_to_activation_baseline, block_name_to_activation_distorted):

        noises = {}
        signals = {}

        for k in block_name_to_activation_distorted.keys():
            distorted = block_name_to_activation_distorted[k]
            baseline = block_name_to_activation_baseline[k]

            dim = [1] if distorted.dim() == 2 else [1, 2, 3]

            noises[k] = ((distorted - baseline) ** 2).mean(dim=dim).cpu().numpy()
            signals[k] = (baseline ** 2).mean(dim=dim).cpu().numpy()

        return noises, signals

    def get_activations(self, block_size_spec, input_block_name, input_tensor, output_block_names, ground_truth):
        with torch.no_grad():
            with model_block_relu_transform(self.model, block_size_spec, self.arch_utils) as model:
                torch.cuda.empty_cache()
                input_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == input_block_name)[0, 0]

                output_block_indices = [np.argwhere(np.array(self.params.BLOCK_NAMES) == output_block_name)[0, 0] for
                                        output_block_name in output_block_names]

                block_name_to_activation = dict()
                activation = input_tensor

                for block_index in range(input_block_index, max(output_block_indices) + 1):
                    block_name = self.params.BLOCK_NAMES[block_index]
                    activation = self.arch_utils.run_model_block(model, activation, self.params.BLOCK_NAMES[block_index])

                    if block_name in output_block_names:
                        block_name_to_activation[block_name] = activation

                if output_block_names[-1] == self.params.BLOCK_NAMES[-1]:
                    losses = self.get_loss(activation, ground_truth)
                else:
                    losses = np.nan * np.ones(shape=(input_tensor.shape[0],))

                return block_name_to_activation, losses

    @lru_cache(maxsize=2)
    def get_batch_data(self, batch_index, batch_size, block_size_spec_id):
        block_size_spec = ctypes.cast(block_size_spec_id, ctypes.py_object).value
        input_images, ground_truth = self.get_samples(batch_index, batch_size=batch_size)

        block_name_to_activation, losses = \
            self.get_activations(block_size_spec,
                                 input_block_name=self.params.BLOCK_NAMES[0],
                                 input_tensor=input_images,
                                 output_block_names=self.params.BLOCK_NAMES,
                                 ground_truth=ground_truth)
        block_name_to_activation["input_images"] = input_images
        return block_name_to_activation, ground_truth, losses

    def get_batch_distortion(self,
                             clean_block_size_spec,
                             baseline_block_size_spec,
                             block_size_spec,
                             batch_index,
                             batch_size,
                             input_block_name,
                             output_block_name):

        block_name_to_activation_clean, ground_truth_clean, losses_clean = \
            self.get_batch_data(batch_index, batch_size, id(clean_block_size_spec))

        block_name_to_activation_baseline, _, losses_baseline = \
            self.get_batch_data(batch_index, batch_size, id(baseline_block_size_spec))

        input_tensor = block_name_to_activation_baseline[self.params.BLOCK_INPUT_DICT[input_block_name]]

        block_name_to_activation_distorted, losses_distorted = \
            self.get_activations(block_size_spec,
                                 input_block_name=input_block_name,
                                 input_tensor=input_tensor,
                                 output_block_names=[output_block_name],
                                 ground_truth=ground_truth_clean)

        noises, signals = self.get_distortion(
            block_name_to_activation_baseline=block_name_to_activation_clean,
            block_name_to_activation_distorted=block_name_to_activation_distorted)

        assets = {
            "Baseline Loss": float(losses_clean),
            "Distorted Loss": float(losses_distorted),
            "Noise": noises[output_block_name].mean(),
            "Signal": signals[output_block_name].mean(),
        }
        return assets


class ChannelDistortionHandler:
    def __init__(self, gpu_id, output_path, checkpoint, config, is_train_mode=False, baseline_block_size_spec_path=None, clean_block_size_spec_path=None):

        self.gpu_id = gpu_id
        self.output_path = output_path
        self.checkpoint = checkpoint
        self.config = config
        self.is_train_mode = is_train_mode
        self.baseline_block_size_spec_path = baseline_block_size_spec_path
        self.clean_block_size_spec_path = clean_block_size_spec_path

        self.cfg = mmcv.Config.fromfile(self.config)
        self.params = param_factory(self.cfg)

        os.makedirs(output_path, exist_ok=True)
        self.distortion_utils = DistortionUtils(gpu_id=self.gpu_id,
                                                params=self.params,
                                                checkpoint=self.checkpoint,
                                                cfg=self.cfg,
                                                mode="distortion_extraction")

        if self.is_train_mode:
            self.distortion_utils.model.train()

        if self.baseline_block_size_spec_path:
            assert os.path.exists(self.baseline_block_size_spec_path)
            self.baseline_block_size_spec = pickle.load(open(self.baseline_block_size_spec_path, 'rb'))
        else:
            self.baseline_block_size_spec = dict()

        if self.clean_block_size_spec_path:
            assert os.path.exists(self.clean_block_size_spec_path)
            self.clean_block_size_spec = pickle.load(open(self.clean_block_size_spec_path, 'rb'))
        else:
            self.clean_block_size_spec = dict()

    def extract_deformation_channel_ord(self,
                                        batch_index: int,
                                        batch_size: int):

        channels_to_run, _ = get_channels_subset(params=self.params)

        os.makedirs(self.output_path, exist_ok=True)

        for layer_name in self.params.LAYER_NAMES:

            block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]
            layer_num_channels = self.params.LAYER_NAME_TO_DIMS[layer_name][0]
            input_block_name = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
            output_block_name = self.params.BLOCK_NAMES[-1]  # TODO: Why do we have None in last layer

            block_size_spec = copy.deepcopy(self.baseline_block_size_spec)

            if layer_name not in block_size_spec:
                block_size_spec[layer_name] = np.ones(shape=(layer_num_channels, 2), dtype=np.int32)

            for channel in tqdm(channels_to_run[layer_name], desc=f"Batch={batch_index} Layer={layer_name}"):

                file_name = os.path.join(self.output_path, f"{layer_name}_{channel}_{batch_index}.npy")
                if os.path.exists(file_name):
                    continue

                channel_noise = np.zeros(shape=(len(block_sizes), ))

                for block_size_index, block_size in enumerate(block_sizes):

                    orig_block_size = block_size_spec[layer_name][channel].copy()
                    block_size_spec[layer_name][channel] = block_size

                    cur_assets = self.distortion_utils.get_batch_distortion(
                        clean_block_size_spec=self.clean_block_size_spec,
                        baseline_block_size_spec=self.baseline_block_size_spec,
                        block_size_spec=block_size_spec,
                        batch_index=batch_index,
                        batch_size=batch_size,
                        input_block_name=input_block_name,
                        output_block_name=output_block_name)

                    block_size_spec[layer_name][channel] = orig_block_size
                    channel_noise[block_size_index] = cur_assets["Noise"]
                np.save(file_name, channel_noise)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--config', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--batch_index', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--baseline_block_size_spec_path', type=str, default=None)
    parser.add_argument('--clean_block_size_spec_path', type=str, default=None)
    parser.add_argument('--train_mode', action='store_true', default=False)

    args = parser.parse_args()

    chd = ChannelDistortionHandler(gpu_id=args.gpu_id,
                                   output_path=args.output_path,
                                   checkpoint=args.checkpoint,
                                   config=args.config,
                                   is_train_mode=args.train_mode,
                                   baseline_block_size_spec_path=args.baseline_block_size_spec_path,
                                   clean_block_size_spec_path=args.clean_block_size_spec_path)

    chd.extract_deformation_channel_ord(batch_index=args.batch_index,
                                        batch_size=args.batch_size)
