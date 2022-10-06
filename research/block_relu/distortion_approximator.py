import collections

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import torch
import numpy as np
import pickle
import time
from research.block_relu.utils import get_model, get_data, center_crop, ArchUtilsFactory
from research.block_relu.params import ParamsFactory

from research.pipeline.backbones.secure_resnet import MyResNet # TODO: find better way to init
from mmseg.ops import resize
import torch.nn.functional as F
from mmseg.core import intersect_and_union
import contextlib
from research.block_relu.params import MobileNetV2Params
from functools import lru_cache

LEVEL_TO_BLOCK_SIZE = \
    {
        0: [[i, j] for i in range(1, 3) for j in range(1, 3)],
        1: [[i, j] for i in range(1, 5) for j in range(1, 5)],
        2: [[i, j] for i in range(1, 9) for j in range(1, 9)],
        3: [[i, j] for i in range(1, 17) for j in range(1, 17)],
        4: [[i, j] for i in range(1, 33) for j in range(1, 33)],
    }


@contextlib.contextmanager
def model_block_relu_transform(model, relu_spec, arch_utils, params):

    layer_name_to_orig_layer = {}
    for layer_name, block_size_info in relu_spec.items():
        if type(block_size_info) is tuple:
            block_size_indices, block_sizes = block_size_info
        else:
            block_size_indices = block_size_info
            block_sizes = params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]

        orig_layer = arch_utils.get_layer(model, layer_name)
        layer_name_to_orig_layer[layer_name] = orig_layer

        arch_utils.set_bReLU_layers(model, {layer_name: (block_size_indices, block_sizes)})

    yield model

    for layer_name_, orig_layer in layer_name_to_orig_layer.items():
        arch_utils.set_layers(model, {layer_name_: orig_layer})

# def get_random_spec(self, seed, layers):
#     np.random.seed(seed)
#
#     channel_ord_to_layer_name = np.hstack([self.params.LAYER_NAME_TO_CHANNELS[layer_name] * [layer_name] for layer_name in self.params.LAYER_NAMES])
#     channel_ord_to_channel_index = np.hstack([np.arange(self.params.LAYER_NAME_TO_CHANNELS[layer_name]) for layer_name in self.params.LAYER_NAMES])
#     num_channels = len(channel_ord_to_layer_name)
#
#     num_channel_to_add_noise_to = num_channels
#     channel_sample = np.random.choice(num_channels, size=num_channel_to_add_noise_to, replace=False)
#     layer_and_channels = [(channel_ord_to_layer_name[x], channel_ord_to_channel_index[x]) for x in channel_sample]
#
#     block_size_indices = [np.random.randint(1, len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])) for
#                           layer_name, _ in layer_and_channels]
#     block_size_spec = {
#         layer_name: np.zeros(shape=(self.params.LAYER_NAME_TO_CHANNELS[layer_name],), dtype=np.int32) for
#         layer_name in self.params.LAYER_NAMES if layer_name in layers}
#
#     for index, (layer_name, channel) in zip(block_size_indices, layer_and_channels):
#         if layer_name in layers:
#             block_size_spec[layer_name][channel] = index
#
#     return block_size_spec

def split_size_spec(params, block_size_spec, num_specs, split_method="random"):
    if split_method == "random":
        channel_ord_to_layer_name = np.hstack(
            [params.LAYER_NAME_TO_CHANNELS[layer_name] * [layer_name] for layer_name in params.LAYER_NAMES])
        channel_ord_to_channel_index = np.hstack(
            [np.arange(params.LAYER_NAME_TO_CHANNELS[layer_name]) for layer_name in params.LAYER_NAMES])

        channels = np.arange(len(channel_ord_to_layer_name))
        np.random.shuffle(channels)
        channel_groups = np.array_split(channels, num_specs)

        split_specs = []

        for channel_group in channel_groups:
            new_block_size_spec = {k: np.zeros_like(v) for k, v in block_size_spec.items()}
            for channel in channel_group:
                layer_name = channel_ord_to_layer_name[channel]
                channel_index = channel_ord_to_channel_index[channel]
                new_block_size_spec[layer_name][channel_index] = block_size_spec[layer_name][channel_index]

            split_specs.append(new_block_size_spec)
        return split_specs
    elif split_method == "by_layer":
        return [{layer_name:block_size_spec[layer_name]} for layer_name in params.LAYER_NAMES]
    else:
        assert False

def get_random_spec(seed, layer_names, params, channel_count=None):
    np.random.seed(seed)

    if channel_count is None:
        block_size_spec = {layer_name: np.random.randint(low=0,
                                                         high=len(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]),
                                                         size=params.LAYER_NAME_TO_CHANNELS[layer_name]) for layer_name in
                           layer_names}
    else:
        block_size_spec = dict()
        for layer_name in layer_names:
            cur_channel_count = min(channel_count, params.LAYER_NAME_TO_CHANNELS[layer_name])
            channels = np.random.choice(params.LAYER_NAME_TO_CHANNELS[layer_name], size=cur_channel_count, replace=False)
            values = np.random.randint(low=0, high=len(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]), size=cur_channel_count)
            layer_spec = np.zeros(shape=params.LAYER_NAME_TO_CHANNELS[layer_name], dtype=np.int32)
            layer_spec[channels] = values
            block_size_spec[layer_name] = layer_spec

    return block_size_spec


def get_random_spec_v2(seed, layer_names, params, level=None):
    np.random.seed(seed)
    if level is None:
        level = np.random.randint(low=0, high=5)
    block_size_spec = {layer_name: (np.random.randint(low=0, high=len(LEVEL_TO_BLOCK_SIZE[level]),
                                                     size=params.LAYER_NAME_TO_CHANNELS[layer_name]), LEVEL_TO_BLOCK_SIZE[level]) for layer_name in
                       layer_names}

    return block_size_spec

IMAGES = "***"
class DistortionStatistics:
    def __init__(self, gpu_id):

        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.params = MobileNetV2Params()
        self.deformation_base_path = "/home/yakir/tmp_d"

        self.arch_utils = ArchUtilsFactory()("MobileNetV2")
        self.model = get_model(
            config="/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_m-v2-d8_512x512_160k_ade20k/deeplabv3_m-v2-d8_512x512_160k_ade20k.py",
            gpu_id=self.gpu_id,
            checkpoint_path="/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_m-v2-d8_512x512_160k_ade20k/iter_160000.pth"
        )

        self.ds_name = "ade_20k"
        self.dataset = get_data(self.ds_name)
        np.random.seed(123)
        self.shuffled_indices = np.arange(len(self.dataset))
        np.random.shuffle(self.shuffled_indices)

    def get_mIoU(self, out, ground_truth):
        seg_logit = resize(
            input=out,
            size=(ground_truth.shape[2], ground_truth.shape[3]),
            mode='bilinear',
            align_corners=self.model.decode_head.align_corners)
        output = F.softmax(seg_logit, dim=1)

        seg_pred = output.argmax(dim=1).cpu().numpy()
        gt = ground_truth[:, 0].cpu().numpy()

        mIoUs = []
        for sample in range(seg_pred.shape[0]):
            results = [intersect_and_union(
                seg_pred[sample:sample+1],
                gt[sample:sample+1],
                len(self.dataset.CLASSES),
                self.dataset.ignore_index,
                label_map=dict(),
                reduce_zero_label=False)]
            assert self.ds_name == "ade_20k"

            mIoUs.append(self.dataset.evaluate(results, logger='silent', **{'metric': ['mIoU']})['mIoU'])


        return mIoUs

    def get_loss(self, out, ground_truth):
        # loss_ce = self.model.decode_head.losses(out, ground_truth)['loss_ce'].cpu().numpy()
        loss_ce_list = []
        for sample_id in range(out.shape[0]):
            loss_ce_list.append(self.model.decode_head.losses(out[sample_id:sample_id+1], ground_truth[sample_id:sample_id+1])['loss_ce'].cpu().numpy())
        return loss_ce_list

    def get_samples(self, batch_indices, im_size=512):
        batch = torch.stack([center_crop(self.dataset[sample_id]['img'].data, im_size) for sample_id in batch_indices]).to(self.device)
        ground_truth = torch.stack([center_crop(self.dataset[sample_id]['gt_semantic_seg'].data, im_size) for sample_id in batch_indices]).to(self.device)
        return batch, ground_truth

    def get_activations(self, input_block_name, input_tensor, output_block_names, model, ground_truth=None):
        torch.cuda.empty_cache()
        input_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == input_block_name)[0, 0]

        output_block_indices = [np.argwhere(np.array(self.params.BLOCK_NAMES) == output_block_name)[0, 0] for output_block_name in output_block_names]

        resnet_block_name_to_activation = dict()
        activation = input_tensor

        for block_index in range(input_block_index, max(output_block_indices) + 1):
            block_name = self.params.BLOCK_NAMES[block_index]
            activation = self.arch_utils.run_model_block(model, activation, self.params.BLOCK_NAMES[block_index])

            if block_name in output_block_names:
                resnet_block_name_to_activation[block_name] = activation

        if output_block_names[-1] == "decode":
            mIoUs = self.get_mIoU(activation, ground_truth)
            losses = self.get_loss(activation, ground_truth)
        else:
            mIoUs = []
            losses = []

        return resnet_block_name_to_activation, losses, mIoUs

    def get_distortion(self, resnet_block_name_to_activation_baseline, resnet_block_name_to_activation_distorted):

        noises = {}
        signals = {}

        for k in resnet_block_name_to_activation_distorted.keys():
            distorted = resnet_block_name_to_activation_distorted[k]
            baseline = resnet_block_name_to_activation_baseline[k]

            noises[k] = ((distorted - baseline) ** 2).mean(dim=[1,2,3]).cpu().numpy()
            signals[k] = (baseline ** 2).mean(dim=[1,2,3]).cpu().numpy()

        return noises, signals

    def get_expected_distortion(self, seed, batch_index=0, batch_size=8, im_size=512):
        np.random.seed(seed)
        block_size_spec = get_random_spec(self.params.LAYER_NAMES, self.params)
        # block_size_spec = {self.params.LAYER_NAMES[15]:block_size_spec[self.params.LAYER_NAMES[15]]}
        with torch.no_grad():

            torch.cuda.empty_cache()

            # Baseline
            # batch, ground_truth, resnet_block_name_to_activation_baseline, losses_baseline, mIoUs_baseline =\
            #     self.get_samples_and_activations(batch=batch_index, batch_size=batch_size, im_size=im_size)

            # batch_indices = np.arange(batch_index * batch_size, batch_index * batch_size + batch_size)
            batch, ground_truth = self.get_samples([1], im_size=512)
            resnet_block_name_to_activation_baseline, losses_baseline, mIoUs_baseline = \
                self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                     input_tensor=batch,
                                     output_block_names=self.params.BLOCK_NAMES[:-1],
                                     model=self.model,
                                     ground_truth=ground_truth)

            with model_block_relu_transform(self.model, block_size_spec, self.arch_utils, self.params) as noisy_model:
                resnet_block_name_to_activation_distorted, losses_distorted, mIoUs_distorted = \
                    self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                         input_tensor=batch,
                                         output_block_names=['decode'],
                                         model=noisy_model,
                                         ground_truth=ground_truth)

            noises, signals = self.get_distortion(
                resnet_block_name_to_activation_baseline=resnet_block_name_to_activation_baseline,
                resnet_block_name_to_activation_distorted=resnet_block_name_to_activation_distorted)

            noises_distorted = noises["decode"]
            signals_distorted = signals["decode"]

        assets = {
            "losses_baseline": losses_baseline,
            "losses_distorted": losses_distorted,
            "mIoUs_baseline": mIoUs_baseline,
            "mIoUs_distorted": mIoUs_distorted,
            "noises_distorted": noises_distorted,
            "signals_distorted": signals_distorted,
        }
        return assets


    def block_size_based_approximation(self, seed):
        pass

    def decomposed_channel_distortion_baseline(self, seed, batch_index, batch_size, im_size, channel_count):
        block_size_spec = get_random_spec(seed, layer_names=self.params.LAYER_NAMES, params=self.params, channel_count=channel_count)
        layer_name_to_assets = dict()
        for layer_name in self.params.LAYER_NAMES:
            layer_block_size_spec = {layer_name: block_size_spec[layer_name]}
            layer_name_to_assets[layer_name] = self.get_batch_distortion(layer_block_size_spec, batch_index, batch_size, im_size)

        return layer_name_to_assets
    def decomposed_channel_distortion(self, seed, batch_index, batch_size, im_size, channel_count):
        block_size_spec = get_random_spec(seed, layer_names=self.params.LAYER_NAMES, params=self.params, channel_count=channel_count)

        layer_name_to_assets = dict()
        for layer_name in self.params.LAYER_NAMES:
            additive_assets = []
            additive_block_size_spec = {layer_name: np.zeros_like(block_size_spec[layer_name])}
            for channel_index, channel_value in enumerate(block_size_spec[layer_name]):
                if channel_value != 0:
                    additive_block_size_spec[layer_name][channel_index] = channel_value
                    additive_assets.append(self.get_batch_distortion(additive_block_size_spec, batch_index, batch_size, im_size))
                    additive_block_size_spec[layer_name][channel_index] = 0

            layer_name_to_assets[layer_name] = additive_assets

        return layer_name_to_assets

    def get_inputs(self, block_size_spec, batch, resnet_block_name_to_activation_baseline):
        assert len(block_size_spec) in [1, 40]
        earliest_layer_name = self.params.LAYER_NAMES[(np.argwhere(
            (np.array(list(block_size_spec.keys()))[:, np.newaxis] == np.array(self.params.LAYER_NAMES)[np.newaxis]))[:,
                                 1]).min()]
        layer_block = self.params.LAYER_NAME_TO_BLOCK_NAME[earliest_layer_name]
        layer_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == layer_block)[0, 0]
        input_block_name = self.params.BLOCK_NAMES[layer_block_index]
        output_block_names = self.params.BLOCK_NAMES[layer_block_index:-1]
        if layer_block_index == 0:
            input_tensor = batch
        else:
            prev_block_name = self.params.BLOCK_NAMES[layer_block_index - 1]
            input_tensor = resnet_block_name_to_activation_baseline[prev_block_name]

        return input_tensor, input_block_name, output_block_names

    def get_batch_distortion(self, block_size_spec, batch_index, batch_size=8, im_size=512, output_block_names=None):

        with torch.no_grad():
            torch.cuda.empty_cache()

            batch, ground_truth, resnet_block_name_to_activation_baseline, losses_baseline, mIoUs_baseline = \
                self.get_samples_and_activations(batch=batch_index, batch_size=batch_size, im_size=im_size)

            with model_block_relu_transform(self.model, block_size_spec, self.arch_utils, self.params) as noisy_model:

                input_tensor, input_block_name, suggested_output_block_names = self.get_inputs(block_size_spec, batch, resnet_block_name_to_activation_baseline)

                if output_block_names is None:
                    output_block_names = suggested_output_block_names
                resnet_block_name_to_activation_distorted, losses_distorted, mIoUs_distorted = \
                    self.get_activations(input_block_name=input_block_name,
                                         input_tensor=input_tensor,
                                         output_block_names=output_block_names,
                                         model=noisy_model,
                                         ground_truth=ground_truth)

            noises_distorted, signals_distorted = self.get_distortion(
                resnet_block_name_to_activation_baseline=resnet_block_name_to_activation_baseline,
                resnet_block_name_to_activation_distorted=resnet_block_name_to_activation_distorted)

        assets = {
            "losses_baseline": losses_baseline,
            "losses_distorted": losses_distorted,
            "mIoUs_baseline": mIoUs_baseline,
            "mIoUs_distorted": mIoUs_distorted,
            "noises_distorted": noises_distorted,
            "signals_distorted": signals_distorted,
        }
        return assets


    @lru_cache(maxsize=1)
    def get_samples_and_activations(self, batch, batch_size, im_size):
        batch_indices = np.arange(batch*batch_size, batch*batch_size + batch_size)
        batch, ground_truth = self.get_samples(self.shuffled_indices[batch_indices], im_size=im_size)
        resnet_block_name_to_activation_baseline, cur_losses, cur_mIoUs = \
            self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                 input_tensor=batch,
                                 output_block_names=self.params.BLOCK_NAMES[:-1],
                                 model=self.model,
                                 ground_truth=ground_truth)

        return batch, ground_truth, resnet_block_name_to_activation_baseline, cur_losses, cur_mIoUs

    def get_estimated_distortion(self, seed, batch_index=0, batch_size=64, im_size=128):
        num_of_splits = 40

        np.random.seed(seed)
        block_size_spec = get_random_spec(self.params.LAYER_NAMES, self.params)
        split_block_size_specs = split_size_spec(self.params, block_size_spec, num_of_splits, "by_layer")

        noises_additive = np.zeros(shape=(num_of_splits,  batch_size))
        signals_additive = np.zeros(shape=(num_of_splits, batch_size))
        loss_additive = np.zeros(shape=(num_of_splits, batch_size))
        mIoU_additive = np.zeros(shape=(num_of_splits, batch_size))

        with torch.no_grad():
            torch.cuda.empty_cache()

            # Baseline
            # batch, ground_truth, resnet_block_name_to_activation_baseline, estimated_losses_baseline, estimated_mIoUs_baseline =\
            #     self.get_samples_and_activations(batch=batch_index, batch_size=batch_size, im_size=im_size)
            batch, ground_truth = self.get_samples([1], im_size=512)
            resnet_block_name_to_activation_baseline, estimated_losses_baseline, estimated_mIoUs_baseline = \
                self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                     input_tensor=batch,
                                     output_block_names=self.params.BLOCK_NAMES[:-1],
                                     model=self.model,
                                     ground_truth=ground_truth)

            for spec_index, cur_spec in enumerate(split_block_size_specs):
                with model_block_relu_transform(self.model,
                                                cur_spec,
                                                self.arch_utils, self.params) as noisy_model:
                    # keys = list(cur_spec.keys())
                    # assert len(keys) == 1
                    # layer_name = keys[0]
                    # layer_block = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
                    # layer_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == layer_block)[0, 0]
                    # input_block_name = self.params.BLOCK_NAMES[layer_block_index]
                    #
                    # if layer_block_index == 0:
                    #     input_tensor = batch
                    # else:
                    #     prev_block_name = self.params.BLOCK_NAMES[layer_block_index - 1]
                    #     input_tensor = resnet_block_name_to_activation_baseline[prev_block_name]

                    resnet_block_name_to_activation_distorted, layer_loss_distorted, layer_mIoU_distorted = \
                        self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                             input_tensor=batch,
                                             output_block_names=['decode'],
                                             model=noisy_model,
                                             ground_truth=ground_truth)

                noises, signals = self.get_distortion(
                    resnet_block_name_to_activation_baseline=resnet_block_name_to_activation_baseline,
                    resnet_block_name_to_activation_distorted=resnet_block_name_to_activation_distorted)

                noises_additive[spec_index] = noises['decode']
                signals_additive[spec_index] = signals['decode']
                loss_additive[spec_index] = layer_loss_distorted
                mIoU_additive[spec_index] = layer_mIoU_distorted

        assets = {
            "estimated_losses_baseline": estimated_losses_baseline,
            "estimated_mIoUs_baseline": estimated_mIoUs_baseline,
            "noises_additive": noises_additive,
            "signals_additive": signals_additive,
            "loss_additive": loss_additive,
            "mIoU_additive": mIoU_additive,
            "block_size_spec":block_size_spec
        }

        return assets

    def get_network_additivity(self, seed, batch_size, num_batches, num_of_splits, split_type="random"):

        np.random.seed(seed)
        block_size_spec = get_random_spec(self.params.LAYER_NAMES, self.params)
        split_block_size_specs = split_size_spec(self.params, block_size_spec, num_of_splits, split_type)

        batch_indices = [0] #np.random.choice(len(self.dataset), batch_size * num_batches, replace=False)

        noises_additive_network_distorted = np.zeros(shape=(num_of_splits, num_batches, batch_size))
        signals_additive_network_distorted = np.zeros(shape=(num_of_splits, num_batches, batch_size))
        loss_additive_network_distorted = np.zeros(shape=(num_of_splits, num_batches, batch_size))
        mIoU_additive_network_distorted = np.zeros(shape=(num_of_splits, num_batches, batch_size))
        losses_baseline = np.zeros(shape=(num_batches, batch_size))
        losses_distorted = np.zeros(shape=(num_batches, batch_size))
        mIoUs_baseline = np.zeros(shape=(num_batches, batch_size))
        mIoUs_distorted = np.zeros(shape=(num_batches, batch_size))
        noises_distorted = np.zeros(shape=(num_batches, batch_size))
        signals_distorted = np.zeros(shape=(num_batches, batch_size))

        with torch.no_grad():
            for cur_batch_index, cur_batch_indices in enumerate(np.array_split(batch_indices, num_batches)):

                torch.cuda.empty_cache()

                # Baseline
                batch, ground_truth = self.get_samples(cur_batch_indices)
                resnet_block_name_to_activation_baseline, cur_losses, cur_mIoUs = \
                    self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                         input_tensor=batch,
                                         output_block_names=self.params.BLOCK_NAMES[:-1],
                                         model=self.model,
                                         ground_truth=ground_truth)

                losses_baseline[cur_batch_index] = cur_losses
                mIoUs_baseline[cur_batch_index] = cur_mIoUs

                # Real
                with model_block_relu_transform(self.model, block_size_spec, self.arch_utils, self.params) as noisy_model:

                    resnet_block_name_to_activation_distorted, cur_losses, cur_mIoUs = \
                        self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                             input_tensor=batch,
                                             output_block_names=['decode'],
                                             model=noisy_model,
                                             ground_truth=ground_truth)

                losses_distorted[cur_batch_index] = cur_losses
                mIoUs_distorted[cur_batch_index] = cur_mIoUs

                real_noises_distorted, real_signals_distorted = self.get_distortion(
                    resnet_block_name_to_activation_baseline=resnet_block_name_to_activation_baseline,
                    resnet_block_name_to_activation_distorted=resnet_block_name_to_activation_distorted)

                noises_distorted[cur_batch_index] = real_noises_distorted["decode"]
                signals_distorted[cur_batch_index] = real_signals_distorted["decode"]


                # By Channe

                for spec_index, cur_spec in enumerate(split_block_size_specs):
                    with model_block_relu_transform(self.model,
                                                    cur_spec,
                                                    self.arch_utils, self.params) as noisy_model:

                        resnet_block_name_to_activation_distorted, layer_loss_distorted, layer_mIoU_distorted = \
                            self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                                 input_tensor=batch,
                                                 output_block_names=['decode'],
                                                 model=noisy_model,
                                                 ground_truth=ground_truth)

                    noises, signals = self.get_distortion(
                        resnet_block_name_to_activation_baseline=resnet_block_name_to_activation_baseline,
                        resnet_block_name_to_activation_distorted=resnet_block_name_to_activation_distorted)

                    noises_additive_network_distorted[spec_index, cur_batch_index] = noises['decode']
                    signals_additive_network_distorted[spec_index, cur_batch_index] = signals['decode']
                    loss_additive_network_distorted[spec_index, cur_batch_index] = layer_loss_distorted
                    mIoU_additive_network_distorted[spec_index, cur_batch_index] = layer_mIoU_distorted

        assets = {
            "losses_baseline": losses_baseline,
            "losses_distorted": losses_distorted,
            "mIoUs_baseline": mIoUs_baseline,
            "mIoUs_distorted": mIoUs_distorted,
            "noises_distorted": noises_distorted,
            "signals_distorted": signals_distorted,
            "noises_additive_layer_distorted": noises_additive_network_distorted,
            "signals_additive_layer_distorted": signals_additive_network_distorted,

            "loss_additive_layer_distorted": loss_additive_network_distorted,
            "mIoU_additive_layer_distorted": mIoU_additive_network_distorted,
        }
        return assets

    def get_additive_sample_estimation(self, seed, layer_name, num_of_samples_to_approximate, batch_size,
                                       channel_count=None):

        assert batch_size == num_of_samples_to_approximate == 1

        np.random.seed(seed)
        block_size_spec = get_random_spec([layer_name], self.params, channel_count=channel_count)
        batch_indices = np.random.choice(len(self.dataset), num_of_samples_to_approximate, replace=False)

        with torch.no_grad():

            torch.cuda.empty_cache()

            # Baseline
            batch, ground_truth = self.get_samples(batch_indices)
            resnet_block_name_to_activation_baseline, loss_baseline, mIoU_baseline = \
                self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                     input_tensor=batch,
                                     output_block_names=self.params.BLOCK_NAMES[:-1],
                                     model=self.model,
                                     ground_truth=ground_truth)

            layer_block = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
            layer_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == layer_block)[0, 0]
            input_block_name = self.params.BLOCK_NAMES[layer_block_index]

            if layer_block_index == 0:
                input_tensor = batch
            else:
                prev_block_name = self.params.BLOCK_NAMES[layer_block_index - 1]
                input_tensor = resnet_block_name_to_activation_baseline[prev_block_name]

            # Real
            with model_block_relu_transform(self.model, block_size_spec, self.arch_utils, self.params) as noisy_model:

                resnet_block_name_to_activation_distorted, loss_distorted, mIoU_distorted = \
                    self.get_activations(input_block_name=input_block_name,
                                         input_tensor=input_tensor,
                                         output_block_names=['decode'],
                                         model=noisy_model,
                                         ground_truth=ground_truth)

            noises_distorted, signals_distorted = self.get_distortion(
                resnet_block_name_to_activation_baseline=resnet_block_name_to_activation_baseline,
                resnet_block_name_to_activation_distorted=resnet_block_name_to_activation_distorted)

            noises_distorted = noises_distorted["decode"]
            signals_distorted = signals_distorted["decode"]

            # By Channel

            noises_additive_channel_distorted = np.zeros(shape=self.params.LAYER_NAME_TO_CHANNELS[layer_name])
            signals_additive_channel_distorted = np.zeros(shape=self.params.LAYER_NAME_TO_CHANNELS[layer_name])
            loss_additive_channel_distorted = np.zeros(shape=self.params.LAYER_NAME_TO_CHANNELS[layer_name])
            mIoU_additive_channel_distorted = np.zeros(shape=self.params.LAYER_NAME_TO_CHANNELS[layer_name])

            for channel in range(self.params.LAYER_NAME_TO_CHANNELS[layer_name]):

                block_size_indices_channel = np.zeros(shape=self.params.LAYER_NAME_TO_CHANNELS[layer_name],
                                                      dtype=np.int32)
                block_size_indices_channel[channel] = block_size_spec[layer_name][channel]
                if block_size_indices_channel[channel] == 0:
                    continue
                with model_block_relu_transform(self.model,
                                                {layer_name: block_size_indices_channel},
                                                self.arch_utils, self.params) as noisy_model:

                    resnet_block_name_to_activation_distorted, loss_additive_channel_distorted[channel], \
                    mIoU_additive_channel_distorted[channel] = \
                        self.get_activations(input_block_name=input_block_name,
                                             input_tensor=input_tensor,
                                             output_block_names=['decode'],
                                             model=noisy_model,
                                             ground_truth=ground_truth)

                noises, signals = self.get_distortion(
                    resnet_block_name_to_activation_baseline=resnet_block_name_to_activation_baseline,
                    resnet_block_name_to_activation_distorted=resnet_block_name_to_activation_distorted)

                noises_additive_channel_distorted[channel] = noises["decode"]
                signals_additive_channel_distorted[channel] = signals["decode"]

        assets = {
            "loss_baseline": loss_baseline,
            "loss_distorted": loss_distorted,
            "loss_additive_channel_distorted": loss_additive_channel_distorted,
            "mIoU_baseline": mIoU_baseline,
            "mIoU_distorted": mIoU_distorted,
            "mIoU_additive_channel_distorted": mIoU_additive_channel_distorted,
            "noises_distorted": noises_distorted,
            "signals_distorted": signals_distorted,
            "noises_additive_channel_distorted": noises_additive_channel_distorted,
            "signals_additive_channel_distorted": signals_additive_channel_distorted,
        }
        return assets

    def get_approximation_sample_estimation(self, seed, layer_name, num_of_samples_to_approximate, batch_size, channel_count=None, im_size=512, output_block_names=["decode"]):

        np.random.seed(seed)
        block_size_spec = get_random_spec([layer_name], self.params, channel_count=channel_count)
        batch_indices = np.random.choice(len(self.dataset), num_of_samples_to_approximate, replace=False)
        assert num_of_samples_to_approximate % batch_size == 0

        num_of_batches = num_of_samples_to_approximate // batch_size

        batch_noises_distorted = []
        batch_signals_distorted = []
        batch_loss_distorted = []
        batch_mIoU_distorted = []
        batch_loss_baseline = []
        batch_mIoU_baseline = []

        with torch.no_grad():
            for cur_batch_indices in np.array_split(batch_indices,num_of_batches):
                torch.cuda.empty_cache()

                # Baseline
                batch, ground_truth = self.get_samples(cur_batch_indices, im_size)
                resnet_block_name_to_activation_baseline, loss_baseline, mIoU_baseline = \
                    self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                         input_tensor=batch,
                                         output_block_names=self.params.BLOCK_NAMES[:-1],
                                         model=self.model,
                                         ground_truth=ground_truth)

                layer_block = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
                layer_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == layer_block)[0,0]
                input_block_name = self.params.BLOCK_NAMES[layer_block_index]

                if layer_block_index == 0:
                    input_tensor = batch
                else:
                    prev_block_name = self.params.BLOCK_NAMES[layer_block_index - 1]
                    input_tensor = resnet_block_name_to_activation_baseline[prev_block_name]

                # Real
                with model_block_relu_transform(self.model, block_size_spec, self.arch_utils, self.params) as noisy_model:

                    resnet_block_name_to_activation_distorted, loss_distorted, mIoU_distorted = \
                        self.get_activations(input_block_name=input_block_name,
                                                input_tensor=input_tensor,
                                                output_block_names=output_block_names,
                                                model=noisy_model,
                                                ground_truth=ground_truth)

                noises_distorted, signals_distorted = self.get_distortion(
                    resnet_block_name_to_activation_baseline=resnet_block_name_to_activation_baseline,
                    resnet_block_name_to_activation_distorted=resnet_block_name_to_activation_distorted)

                batch_noises_distorted.append(np.array([noises_distorted[block_name] for block_name in output_block_names]).T)
                batch_signals_distorted.append(np.array([signals_distorted[block_name] for block_name in output_block_names]).T)
                batch_loss_distorted.append(loss_distorted)
                batch_mIoU_distorted.append(mIoU_distorted)
                batch_loss_baseline.append(loss_baseline)
                batch_mIoU_baseline.append(mIoU_baseline)

        batch_noises_distorted = np.array(batch_noises_distorted)
        batch_signals_distorted = np.array(batch_signals_distorted)
        s = batch_noises_distorted.shape

        ret = dict()
        ret["batch_noises_distorted"] = batch_noises_distorted.reshape((s[0] * s[1], s[2]))
        ret["batch_signals_distorted"] = batch_signals_distorted.reshape((s[0] * s[1], s[2]))
        ret["batch_loss_distorted"] = np.array(batch_loss_distorted).reshape(-1)
        ret["batch_mIoU_distorted"] = np.array(batch_mIoU_distorted).reshape(-1)
        ret["batch_loss_baseline"] = np.array(batch_loss_baseline).reshape(-1)
        ret["batch_mIoU_baseline"] = np.array(batch_mIoU_baseline).reshape(-1)

        return ret


if __name__ == '__main__':
    # Image size, Batch size, Layer of distortion

    action = "by_layer_additivity_v2"

    if action == "baseline_distortion":
        gpu_id = 1
        dh = DistortionStatistics(gpu_id=gpu_id)

        out_dir = f"/home/yakir/distortion_approximation_v2/baseline_distortion/"
        os.makedirs(out_dir, exist_ok=True)

        for seed in np.arange(gpu_id, 1000000, 2):
            for batch_index in tqdm(range(2048), desc=f"seed={seed}"):
                file_name = os.path.join(out_dir, f"{seed}_{batch_index}.pickle")

                if not os.path.exists(file_name):
                    block_size_spec = get_random_spec(seed, layer_names=dh.params.LAYER_NAMES, params=dh.params)
                    assets = dh.get_batch_distortion(block_size_spec, batch_index, batch_size=8, im_size=512)
                    pickle.dump(obj=assets, file=open(file_name, 'wb'))

    if action == "by_layer_baseline_distortion":
        gpu_id = 1
        dh = DistortionStatistics(gpu_id=gpu_id)
        layer_names = ['layer1_0_0', 'layer4_2_1', 'layer5_0_1', 'decode_3', 'layer6_0_1', 'layer7_0_1', 'layer4_1_0', 'layer3_0_1', 'layer4_3_1', 'layer6_2_0', 'layer4_0_1', 'layer5_1_1', 'layer6_1_1', 'layer2_1_1', 'layer2_0_1', 'layer4_1_1', 'layer3_1_1', 'layer5_0_0', 'layer3_2_1', 'layer5_2_1']

        out_dir = f"/home/yakir/distortion_approximation_v2/by_layer_baseline_distortion/"
        os.makedirs(out_dir, exist_ok=True)

        for seed in np.arange(gpu_id, 1000000, 2):
            for batch_index in tqdm(range(128), desc=f"seed={seed}"):
                block_size_spec = get_random_spec(seed, layer_names=dh.params.LAYER_NAMES, params=dh.params)
                for layer_name in layer_names:
                    file_name = os.path.join(out_dir, f"{layer_name}_{seed}_{batch_index}.pickle")

                    if not os.path.exists(file_name):
                        assets = dh.get_batch_distortion({layer_name:block_size_spec[layer_name]}, batch_index, batch_size=8, im_size=512)
                        pickle.dump(obj=assets, file=open(file_name, 'wb'))

    if action == "by_layer_additivity":
        gpu_id = 1
        dh = DistortionStatistics(gpu_id=gpu_id)

        out_dir = f"/home/yakir/distortion_approximation_v2/by_layer_additivity/"
        os.makedirs(out_dir, exist_ok=True)

        for seed in np.arange(gpu_id, 1000000, 2):
            for batch_index in tqdm(range(4), desc=f"seed={seed}"):
                block_size_spec = get_random_spec(seed, layer_names=dh.params.LAYER_NAMES, params=dh.params)
                for layer_name in dh.params.LAYER_NAMES[:28]:
                    file_name = os.path.join(out_dir, f"{layer_name}_{seed}_{batch_index}.pickle")
                    cur_block_size_spec = {layer_name: block_size_spec[layer_name]}

                    block_name = dh.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
                    block_index = dh.params.BLOCK_NAMES_TO_BLOCK_INDEX[block_name]
                    block_indices = [block_index, block_index + 1, block_index + 2, block_index + 3]
                    block_names = [dh.params.BLOCK_NAMES[index] for index in block_indices]

                    if not os.path.exists(file_name):
                        assets = dh.get_batch_distortion(cur_block_size_spec, batch_index, batch_size=8, im_size=512)
                        additive_assets = []
                        additive_block_size_spec = {layer_name: np.zeros_like(block_size_spec[layer_name])}
                        for channel_index, channel_value in enumerate(block_size_spec[layer_name]):
                            additive_block_size_spec[layer_name][channel_index] = channel_value
                            additive_assets.append(dh.get_batch_distortion(additive_block_size_spec, batch_index, batch_size=8, im_size=512, output_block_names=block_names))
                            additive_block_size_spec[layer_name][channel_index] = 0

                        pickle.dump(obj={"assets":assets, "additive_assets":additive_assets}, file=open(file_name, 'wb'))

    if action == "between_layers_additivity":
        gpu_id = 1
        dh = DistortionStatistics(gpu_id=gpu_id)

        out_dir = f"/home/yakir/distortion_approximation_v2/between_layers_additivity/"
        os.makedirs(out_dir, exist_ok=True)

        for seed in np.arange(gpu_id, 1000000, 2):
            for batch_index in range(1):
                file_name = os.path.join(out_dir, f"{seed}_{batch_index}.pickle")

                block_size_spec = get_random_spec(seed, layer_names=dh.params.LAYER_NAMES, params=dh.params)
                assets = dh.get_batch_distortion(block_size_spec, batch_index, batch_size=8, im_size=512)

                layers_asset = []
                for layer_name in tqdm(dh.params.LAYER_NAMES, desc=f"seed={seed}"):
                    layers_asset.append(dh.get_batch_distortion({layer_name: block_size_spec[layer_name]}, batch_index, batch_size=8, im_size=512))

                pickle.dump(obj={"assets": assets, "layers_asset": layers_asset}, file=open(file_name, 'wb'))

    if action == "by_layer_additivity_v2":
        gpu_id = 1
        dh = DistortionStatistics(gpu_id=gpu_id)

        out_dir = f"/home/yakir/distortion_approximation_v2/by_layer_additivity_v2/"
        os.makedirs(out_dir, exist_ok=True)

        for seed in np.arange(gpu_id, 1000000, 2):
            for batch_index in tqdm(range(2), desc=f"seed={seed}"):
                block_size_spec = get_random_spec_v2(seed, layer_names=dh.params.LAYER_NAMES, params=dh.params)

                for layer_name in dh.params.LAYER_NAMES[:28]:
                    file_name = os.path.join(out_dir, f"{layer_name}_{seed}_{batch_index}.pickle")
                    cur_block_size_spec = {layer_name: block_size_spec[layer_name]}

                    block_name = dh.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
                    block_index = dh.params.BLOCK_NAMES_TO_BLOCK_INDEX[block_name]
                    block_indices = [block_index, block_index + 1, block_index + 2, block_index + 3]
                    block_names = [dh.params.BLOCK_NAMES[index] for index in block_indices]

                    if not os.path.exists(file_name):
                        assets = dh.get_batch_distortion(cur_block_size_spec, batch_index, batch_size=8, im_size=512)
                        additive_assets = []
                        additive_block_size_spec = {layer_name: (np.zeros_like(block_size_spec[layer_name][0]), block_size_spec[layer_name][1])}
                        for channel_index, channel_value in enumerate(block_size_spec[layer_name][0]):
                            additive_block_size_spec[layer_name][0][channel_index] = channel_value
                            additive_assets.append(dh.get_batch_distortion(additive_block_size_spec, batch_index, batch_size=8, im_size=512, output_block_names=block_names))
                            additive_block_size_spec[layer_name][0][channel_index] = 0

                        pickle.dump(obj={"assets":assets, "additive_assets":additive_assets}, file=open(file_name, 'wb'))


    #         def decomposed_channel_distortion(self, seed, batch_index, batch_size, im_size, channel_count):
    #         block_size_spec = get_random_spec(seed, layer_names=self.params.LAYER_NAMES, params=self.params, channel_count=channel_count)
    #
    #         layer_name_to_assets = dict()
    #         for layer_name in self.params.LAYER_NAMES:
    #             additive_assets = []
    #             additive_block_size_spec = {layer_name: np.zeros_like(block_size_spec[layer_name])}
    #             for channel_index, channel_value in enumerate(block_size_spec[layer_name]):
    #                 if channel_value != 0:
    #                     additive_block_size_spec[layer_name][channel_index] = channel_value
    #                     additive_assets.append(self.get_batch_distortion(additive_block_size_spec, batch_index, batch_size, im_size))
    #                     additive_block_size_spec[layer_name][channel_index] = 0
    #
    #             layer_name_to_assets[layer_name] = additive_assets
    #
    #         return layer_name_to_assets






    #
    #
    #
    #
    #
    #
    # assert False
    #
    # if False:
    #     gpu_id = 0
    #     dh = DistortionStatistics(gpu_id=gpu_id)
    #     out_dir = f"/home/yakir/distortion_approximation/decomposed_channel_distortion/"
    #     out_dir_baseline = f"/home/yakir/distortion_approximation/decomposed_channel_distortion_baseline/"
    #     os.makedirs(out_dir, exist_ok=True)
    #     os.makedirs(out_dir_baseline, exist_ok=True)
    #
    #     for seed in tqdm(np.arange(gpu_id, 1000000, 2)):
    #         for batch_index in range(1):
    #
    #             layer_name_to_assets = dh.decomposed_channel_distortion(seed=seed, batch_index=batch_index, batch_size=8, im_size=512, channel_count=16)
    #             file_name = os.path.join(out_dir, f"{seed}_{batch_index}.pickle")
    #             pickle.dump(obj=layer_name_to_assets, file=open(file_name, 'wb'))
    #
    #             layer_name_to_assets = dh.decomposed_channel_distortion_baseline(seed=seed, batch_index=batch_index, batch_size=8, im_size=512, channel_count=16)
    #             file_name = os.path.join(out_dir_baseline, f"{seed}_{batch_index}.pickle")
    #             pickle.dump(obj=layer_name_to_assets, file=open(file_name, 'wb'))
    # else:
    #     gpu_id = 1
    #     dh = DistortionStatistics(gpu_id=gpu_id)
    #     out_dir = f"/home/yakir/distortion_approximation/decomposed_channel_distortion_512/"
    #     out_dir_baseline = f"/home/yakir/distortion_approximation/decomposed_channel_distortion_baseline_512/"
    #     os.makedirs(out_dir, exist_ok=True)
    #     os.makedirs(out_dir_baseline, exist_ok=True)
    #
    #     for seed in tqdm(np.arange(gpu_id, 1000000, 2)):
    #         for batch_index in range(1):
    #             layer_name_to_assets = dh.decomposed_channel_distortion(seed=seed, batch_index=batch_index, batch_size=8, im_size=512, channel_count=16)
    #             file_name = os.path.join(out_dir, f"{seed}_{batch_index}.pickle")
    #             pickle.dump(obj=layer_name_to_assets, file=open(file_name, 'wb'))
    #
    #         # for batch_index in range(16):
    #         #     layer_name_to_assets = dh.decomposed_channel_distortion_baseline(seed=seed, batch_index=batch_index, batch_size=8, im_size=512, channel_count=16)
    #         #     file_name = os.path.join(out_dir_baseline, f"{seed}_{batch_index}.pickle")
    #         #     pickle.dump(obj=layer_name_to_assets, file=open(file_name, 'wb'))
    # if False:
    #     gpu_id = 1
    #     dh = DistortionStatistics(gpu_id=gpu_id)
    #     for layer_name in dh.params.LAYER_NAMES:
    #         out_dir = f"/home/yakir/distortion_approximation/get_approximation_sample_estimation/{layer_name}"
    #         os.makedirs(out_dir, exist_ok=True)
    #         output_block_names = dh.params.BLOCK_NAMES[np.argwhere(dh.params.LAYER_NAME_TO_BLOCK_NAME[layer_name] == np.array(dh.params.BLOCK_NAMES))[0,0]:-1]
    #         for seed in tqdm(np.arange(gpu_id, 50, 2)):
    #             for im_size in [512, 128]:
    #                 out_file = os.path.join(out_dir, f"{seed}_{im_size}.pickle")
    #                 if not os.path.exists(out_file):
    #                     out = dh.get_approximation_sample_estimation(
    #                         seed=seed,
    #                         layer_name=layer_name,
    #                         num_of_samples_to_approximate=1024,
    #                         batch_size=8,
    #                         channel_count=None,
    #                         im_size=im_size,
    #                         output_block_names=output_block_names)
    #                     pickle.dump(file=open(out_file,'wb'), obj=out)
    # elif True:
    #     gpu_id = 0
    #     dh = DistortionStatistics(gpu_id=gpu_id)
    #     out_dir = f"/home/yakir/distortion_approximation/get_network_additivity_by_layer_batch_large_debug_2/"
    #     os.makedirs(out_dir, exist_ok=True)
    #
    #     for seed in tqdm(range(100)):
    #         assets = dh.get_network_additivity(seed=seed, batch_size=1, num_batches=1, num_of_splits=len(dh.params.LAYER_NAMES), split_type="by_layer")
    #         file_name = os.path.join(out_dir, f"{seed}.pickle")
    #         pickle.dump(obj=assets, file=open(file_name, 'wb'))
    # elif False:
    #     gpu_id = 1
    #     dh = DistortionStatistics(gpu_id=gpu_id)
    #     out_dir = f"/home/yakir/distortion_approximation/get_network_additivity/"
    #     os.makedirs(out_dir, exist_ok=True)
    #
    #     for seed in tqdm(np.arange(gpu_id, 1000000, 2)):
    #         assets = dh.get_network_additivity(seed=seed, batch_size=8, num_batches=4, num_of_splits=200)
    #         file_name = os.path.join(out_dir, f"{seed}.pickle")
    #         pickle.dump(obj=assets, file=open(file_name, 'wb'))
    # elif True:
    #     gpu_id = 1
    #     dh = DistortionStatistics(gpu_id=gpu_id)
    #     out_dir = f"/home/yakir/distortion_approximation/expected_512/"
    #     os.makedirs(out_dir, exist_ok=True)
    #
    #
    #     for seed in tqdm(range(100)):
    #         assets = dh.get_expected_distortion(seed, batch_index=0, batch_size=1, im_size=512)
    #         file_name = os.path.join(out_dir, f"{seed}.pickle")
    #         pickle.dump(obj=assets, file=open(file_name, 'wb'))
    # else:
    #     gpu_id = 0
    #     dh = DistortionStatistics(gpu_id=gpu_id)
    #     out_dir = f"/home/yakir/distortion_approximation/estimation_512/"
    #     os.makedirs(out_dir, exist_ok=True)
    #
    #     for seed in tqdm(range(100)):
    #         assets = dh.get_estimated_distortion(seed, batch_index=0, batch_size=1, im_size=512)
    #         file_name = os.path.join(out_dir, f"{seed}.pickle")
    #         pickle.dump(obj=assets, file=open(file_name, 'wb'))
