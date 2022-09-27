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

@contextlib.contextmanager
def model_block_relu_transform(model, relu_spec, arch_utils, params):

    layer_name_to_orig_layer = {}
    for layer_name, block_size_indices in relu_spec.items():
        orig_layer = arch_utils.get_layer(model, layer_name)
        layer_name_to_orig_layer[layer_name] = orig_layer

        arch_utils.set_bReLU_layers(model, {layer_name: (block_size_indices, params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])})

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

def get_random_spec(layer_names, params, channel_count=None):

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

    def get_mIoU(self, out, ground_truth):
        seg_logit = resize(
            input=out,
            size=(ground_truth.shape[2], ground_truth.shape[3]),
            mode='bilinear',
            align_corners=self.model.decode_head.align_corners)
        output = F.softmax(seg_logit, dim=1)

        seg_pred = output.argmax(dim=1)

        # results = [intersect_and_union(
        #     seg_pred.cpu().numpy(),
        #     ground_truth.cpu().numpy(),
        #     len(self.dataset.CLASSES),
        #     self.dataset.ignore_index,
        #     label_map=dict(),
        #     reduce_zero_label=self.dataset.reduce_zero_label)]
        results = [intersect_and_union(
            seg_pred.cpu().numpy(),
            ground_truth[:,0].cpu().numpy(),
            len(self.dataset.CLASSES),
            self.dataset.ignore_index,
            label_map=dict(),
            reduce_zero_label=False)]
        assert self.ds_name == "ade_20k"

        mIoU = self.dataset.evaluate(results, logger='silent', **{'metric': ['mIoU']})['mIoU']

        return mIoU

    def get_loss(self, out, ground_truth):
        loss_ce = self.model.decode_head.losses(out, ground_truth)['loss_ce'].cpu().numpy()
        return loss_ce

    def get_samples(self, batch_indices, im_size=512):
        batch = torch.stack([center_crop(self.dataset[sample_id]['img'].data, im_size) for sample_id in batch_indices]).to(self.device)
        ground_truth = torch.stack([center_crop(self.dataset[sample_id]['gt_semantic_seg'].data, im_size) for sample_id in batch_indices]).to(self.device)

        return batch, ground_truth

    # def get_activations(self, batch, ground_truth):
    #
    #     num_blocks = len(self.params.BLOCK_NAMES) - 1
    #
    #     with torch.no_grad():
    #
    #         activations = [batch]
    #         for block_index in range(num_blocks):
    #             activations.append(self.arch_utils.run_model_block(self.model, activations[block_index], self.params.BLOCK_NAMES[block_index]))
    #         resnet_block_name_to_activation = dict(zip(self.params.BLOCK_NAMES, activations))
    #
    #         mIoU = self.get_mIoU(activations[-1], ground_truth)
    #         loss = self.get_loss(activations[-1], ground_truth)
    #
    #         return resnet_block_name_to_activation, loss, mIoU

    def get_activations_v2(self, input_block_name, input_tensor, output_block_names, model, ground_truth=None):
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
            mIoU = self.get_mIoU(activation, ground_truth)
            loss = self.get_loss(activation, ground_truth)
        else:
            mIoU = None
            loss = None

        return resnet_block_name_to_activation, loss, mIoU

    def get_distortion_v2(self, resnet_block_name_to_activation_baseline, resnet_block_name_to_activation_distorted):

        noises = {}
        signals = {}

        for k in resnet_block_name_to_activation_distorted.keys():
            distorted = resnet_block_name_to_activation_distorted[k]
            baseline = resnet_block_name_to_activation_baseline[k]
            noises[k] = float(((distorted - baseline) ** 2).mean())
            signals[k] = float((baseline ** 2).mean())

        return noises, signals

    #
    #
    # def get_distortion(self, resnet_block_name_to_activation, ground_truth, block_size_spec, input_block_name, output_block_names):
    #
    #     input_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == input_block_name)[0, 0]
    #     output_block_indices = [np.argwhere(np.array(self.params.BLOCK_NAMES) == output_block_name)[0, 0] for output_block_name in output_block_names]
    #
    #     input_tensor = resnet_block_name_to_activation[input_block_name]
    #     next_tensors = [resnet_block_name_to_activation[output_block_name] for output_block_name in output_block_names]
    #
    #     with model_block_relu_transform(self.model, block_size_spec, self.arch_utils, self.params) as noisy_model:
    #
    #         cur_out = input_tensor
    #         activations = []
    #         for block_index in range(input_block_index, max(output_block_indices)):
    #             cur_out = self.arch_utils.run_model_block(noisy_model, cur_out, self.params.BLOCK_NAMES[block_index])
    #             activations.append(cur_out)
    #
    #     if output_block_names[-1] is None:
    #         mIoU = self.get_mIoU(activations[-1], ground_truth)
    #         loss = self.get_loss(activations[-1], ground_truth)
    #     else:
    #         mIoU = None
    #         loss = None
    #
    #     noises = []
    #     signals = []
    #
    #     for out, next_tensor in zip(activations, next_tensors):
    #         noise = float(((out - next_tensor) ** 2).mean())
    #         signal = float((next_tensor ** 2).mean())
    #         noises.append(noise)
    #         signals.append(signal)
    #
    #     return noises, signals, loss, mIoU

    def get_additive_sample_estimation(self, seed, layer_name, num_of_samples_to_approximate, batch_size, channel_count=None):

        assert batch_size == num_of_samples_to_approximate == 1

        np.random.seed(seed)
        block_size_spec = get_random_spec([layer_name], self.params, channel_count=channel_count)
        batch_indices = np.random.choice(len(self.dataset), num_of_samples_to_approximate, replace=False)



        with torch.no_grad():

            torch.cuda.empty_cache()

            # Baseline
            batch, ground_truth = self.get_samples(batch_indices)
            resnet_block_name_to_activation_baseline, loss_baseline, mIoU_baseline = \
                self.get_activations_v2(input_block_name=self.params.BLOCK_NAMES[0],
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
                    self.get_activations_v2(input_block_name=input_block_name,
                                            input_tensor=input_tensor,
                                            output_block_names=['decode'],
                                            model=noisy_model,
                                            ground_truth=ground_truth)

            noises_distorted, signals_distorted = self.get_distortion_v2(
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

                block_size_indices_channel = np.zeros(shape=self.params.LAYER_NAME_TO_CHANNELS[layer_name], dtype=np.int32)
                block_size_indices_channel[channel] = block_size_spec[layer_name][channel]
                if block_size_indices_channel[channel] == 0:
                    continue
                with model_block_relu_transform(self.model,
                                                {layer_name:block_size_indices_channel},
                                                self.arch_utils, self.params) as noisy_model:

                    resnet_block_name_to_activation_distorted, loss_additive_channel_distorted[channel], \
                    mIoU_additive_channel_distorted[channel] = \
                        self.get_activations_v2(input_block_name=input_block_name,
                                                input_tensor=input_tensor,
                                                output_block_names=['decode'],
                                                model=noisy_model,
                                                ground_truth=ground_truth)

                noises, signals = self.get_distortion_v2(
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
                    self.get_activations_v2(input_block_name=self.params.BLOCK_NAMES[0],
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
                        self.get_activations_v2(input_block_name=input_block_name,
                                                input_tensor=input_tensor,
                                                output_block_names=output_block_names,
                                                model=noisy_model,
                                                ground_truth=ground_truth)

                noises_distorted, signals_distorted = self.get_distortion_v2(
                    resnet_block_name_to_activation_baseline=resnet_block_name_to_activation_baseline,
                    resnet_block_name_to_activation_distorted=resnet_block_name_to_activation_distorted)

                batch_noises_distorted.append([noises_distorted[block_name] for block_name in output_block_names])
                batch_signals_distorted.append([signals_distorted[block_name] for block_name in output_block_names])
                batch_loss_distorted.append(loss_distorted)
                batch_mIoU_distorted.append(mIoU_distorted)
                batch_loss_baseline.append(loss_baseline)
                batch_mIoU_baseline.append(mIoU_baseline)

        return np.array(batch_noises_distorted), np.array(batch_signals_distorted), np.array(batch_loss_distorted), \
               np.array(batch_mIoU_distorted), np.array(batch_loss_baseline), np.array(batch_mIoU_baseline)


if __name__ == '__main__':
    # Image size, Batch size, Layer of distortion
    gpu_id = 1
    dh = DistortionStatistics(gpu_id=gpu_id)
    layer_name = 'layer5_0_1'
    out_file = f"/home/yakir/distortion_approximation/get_approximation_sample_estimation/{layer_name}"
    os.makedirs(out_file, exist_ok=True)
    output_block_names = dh.params.BLOCK_NAMES[np.argwhere(dh.params.LAYER_NAME_TO_BLOCK_NAME[layer_name] == np.array(dh.params.BLOCK_NAMES))[0,0]:-1]
    for seed in tqdm(np.arange(gpu_id, 1000000, 2)):
        for im_size in [512, 128]:
            batch_noises_distorted, batch_signals_distorted, batch_loss_distorted, batch_mIoU_distorted, \
            batch_loss_baseline, batch_mIoU_baseline = dh.get_approximation_sample_estimation(
                seed=seed,
                layer_name=layer_name,
                num_of_samples_to_approximate=1024,
                batch_size=8,
                channel_count=None,
                im_size=im_size,
                output_block_names=output_block_names)

            np.save(file=os.path.join(out_file, f"batch_noises_distorted_{seed}_{im_size}.npy"), arr=batch_noises_distorted)
            np.save(file=os.path.join(out_file, f"batch_signals_distorted_{seed}_{im_size}.npy"), arr=batch_signals_distorted)
            np.save(file=os.path.join(out_file, f"batch_loss_distorted_{seed}_{im_size}.npy"), arr=batch_loss_distorted)
            np.save(file=os.path.join(out_file, f"batch_mIoU_distorted_{seed}_{im_size}.npy"), arr=batch_mIoU_distorted)
            np.save(file=os.path.join(out_file, f"batch_loss_baseline_{seed}_{im_size}.npy"), arr=batch_loss_baseline)
            np.save(file=os.path.join(out_file, f"batch_mIoU_baseline_{seed}_{im_size}.npy"), arr=batch_mIoU_baseline)

    assert False
    for seed in tqdm(np.arange(gpu_id, 1000000, 2)):
        layer_name_to_assets = dict()
        for layer_name in dh.params.LAYER_NAMES[::4]:
            layer_name_to_assets[layer_name] = dh.get_additive_sample_estimation(seed=seed,
                                                                        layer_name=layer_name,
                                                                        num_of_samples_to_approximate=1,
                                                                        batch_size=1,
                                                                        channel_count=None)
        pickle.dump(obj=layer_name_to_assets, file=open(f"/home/yakir/distortion_approximation/get_additive_sample_estimation/{seed}.pickle", 'wb'))
