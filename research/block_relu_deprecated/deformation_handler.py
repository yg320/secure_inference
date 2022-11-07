# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import glob
import pickle
import shutil

from research.block_relu.consts import TARGET_REDUCTIONS
from research.block_relu.utils import get_model, get_data, center_crop, ArchUtilsFactory
from research.block_relu.params import ParamsFactory

from research.pipeline.backbones.secure_resnet import MyResNet # TODO: find better way to init

from mmseg.ops import resize
import torch.nn.functional as F
from mmseg.core import intersect_and_union

# TODO: add typing
# TODO: pandas
# TODO: docstring
# TODO: multiprocess
# TODO: in mobilnet some layers can be fused
# TODO: Ideally, channel and layer differentiation won't be part of the api. Mainly, the best thing to do is to divide the network log2(num_of_channels) and run with it. Keeping in mind that the proxy layer is the same. and to obviously acount for the different resolutions.
# TODO: block_size_spec should be the same as reduction_spec

# HOUR = 3600
# am7_1 = 1661659248.5575001
# am7_2 = am7_1 + 24 * HOUR
# am7_3 = am7_1 + 48 * HOUR
#
# pm9_1 = am7_1 + 14 * HOUR
# pm9_2 = pm9_1 + 24 * HOUR
#
#
# def is_night(cur_time):
#     is_1_night = (pm9_1 <= cur_time <= am7_2)
#     is_2_night = (pm9_2 <= cur_time <= am7_3)
#     return is_1_night or is_2_night

class DeformationHandler:
    def __init__(self, gpu_id, hierarchy_level, is_extraction, param_json_file, batch_size=8):

        self.gpu_id = gpu_id
        self.hierarchy_level = hierarchy_level
        self.device = f"cuda:{gpu_id}"

        self.params = ParamsFactory()(param_json_file)
        # self.deformation_base_path = os.path.join("/home/yakir/Data2/assets_v3/additive_deformation_estimation_new", self.params.DATASET, self.params.BACKBONE)
        self.deformation_base_path = os.path.join("/home/yakir/Data2/assets_v3/deformations", self.params.DATASET, self.params.BACKBONE)

        # import time
        # assert time.time() <= 1663565856.948121 + 3600 + 3600 + 3600 + 3600
        self.arch_utils = ArchUtilsFactory()(self.params.BACKBONE)

        self.batch_size = batch_size
        self.im_size = 512

        if is_extraction:
            self.model = get_model(
                config=self.params.CONFIG,
                gpu_id=self.gpu_id,
                checkpoint_path=self.params.CHECKPOINT
            )
            self.dataset = get_data(self.params.DATASET)


    def _get_deformation(self, resnet_block_name_to_activation, ground_truth, loss_ce, block_size_spec, input_block_name, output_block_name):

        input_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == input_block_name)[0, 0]
        output_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == output_block_name)[0, 0]

        input_tensor = resnet_block_name_to_activation[input_block_name]
        next_tensor = resnet_block_name_to_activation[output_block_name]

        layer_name_to_orig_layer = {}
        for layer_name, block_sizes in block_size_spec.items():
            orig_layer = self.arch_utils.get_layer(self.model, layer_name)
            layer_name_to_orig_layer[layer_name] = orig_layer

            self.arch_utils.set_bReLU_layers(self.model, {layer_name: block_sizes})
        out = input_tensor
        for block_index in range(input_block_index, output_block_index):
            out = self.arch_utils.run_model_block(self.model, out, self.params.BLOCK_NAMES[block_index])

        if output_block_name is None:
            loss_deform = self.model.decode_head.losses(out, ground_truth)['loss_ce']
            #
            # seg_logit = resize(
            #     input=out,
            #     size=(512, 512),
            #     mode='bilinear',
            #     align_corners=self.model.decode_head.align_corners)
            # output = F.softmax(seg_logit, dim=1)
            #
            # seg_pred = output.argmax(dim=1)
            # # seg_map = self.dataset[0]['gt_semantic_seg'].data
            #
            # results = [intersect_and_union(
            #     seg_pred.cpu().numpy(),
            #     ground_truth[0].cpu().numpy(),
            #     len(self.dataset.CLASSES),
            #     self.dataset.ignore_index,
            #     label_map=dict(),
            #     reduce_zero_label=self.dataset.reduce_zero_label)]
            #
            # mIoU = self.dataset.evaluate(results, logger = 'silent', **{'metric': ['mIoU']})['mIoU']
            #

        else:
            loss_deform = None
            # mIoU = None

        for layer_name_, orig_layer in layer_name_to_orig_layer.items():
            self.arch_utils.set_layers(self.model, {layer_name_: orig_layer})

        noise = float(((out - next_tensor) ** 2).mean())
        signal = float((next_tensor ** 2).mean())

        return noise, signal, loss_deform

        # return noise, signal, float(loss_deform.cpu()), mIoU

    def _get_files_and_matrices(self, batch_index, output_path, layer_name, h, w):

        noise_f_name = os.path.join(output_path,
                                    f"noise_{layer_name}_batch_{batch_index}_{self.batch_size}.npy")
        signal_f_name = os.path.join(output_path,
                                     f"signal_{layer_name}_batch_{batch_index}_{self.batch_size}.npy")
        loss_deform_f_name = os.path.join(output_path,
                                          f"loss_deform_{layer_name}_batch_{batch_index}_{self.batch_size}.npy")
        loss_baseline_f_name = os.path.join(output_path,
                                            f"loss_baseline_{layer_name}_batch_{batch_index}_{self.batch_size}.npy")

        return noise_f_name, signal_f_name, loss_deform_f_name, loss_baseline_f_name, \
               np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))

    def _get_redundancy_arr(self, reduction_to_block_sizes, num_groups, group_size):
        redundancy_arr = np.zeros((TARGET_REDUCTIONS.shape[0], num_groups))

        for reduction_index in range(1, TARGET_REDUCTIONS.shape[0]):
            for group_id in range(num_groups):

                start = group_id * group_size
                end = (group_id + 1) * group_size
                curr = reduction_to_block_sizes[reduction_index, start: end]
                prev = reduction_to_block_sizes[reduction_index - 1, start: end]

                if np.all(curr == prev):
                    redundancy_arr[reduction_index, group_id] = redundancy_arr[reduction_index - 1, group_id]
                else:
                    redundancy_arr[reduction_index, group_id] = reduction_index

        return redundancy_arr

    def get_activations(self, batch_index):

        num_blocks = len(self.params.BLOCK_NAMES) - 1

        with torch.no_grad():
            batch_indices = range(batch_index * self.batch_size, (batch_index + 1) * self.batch_size)
            batch = torch.stack(
                [center_crop(self.dataset[sample_id]['img'].data, self.im_size) for sample_id in batch_indices]).to(
                self.device)
            ground_truth = torch.stack(
                [center_crop(self.dataset[sample_id]['gt_semantic_seg'].data, self.im_size) for sample_id in
                 batch_indices]).to(self.device)
            activations = [batch]
            for block_index in range(num_blocks):
                activations.append(self.arch_utils.run_model_block(self.model, activations[block_index], self.params.BLOCK_NAMES[block_index]))
            loss_ce = self.model.decode_head.losses(activations[-1], ground_truth)['loss_ce'].cpu()
            resnet_block_name_to_activation = dict(zip(self.params.BLOCK_NAMES, activations))

            return resnet_block_name_to_activation, ground_truth, loss_ce

    def extract_deformation_by_blocks(self, batch_index):

        resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations(batch_index)

        output_path = os.path.join(self.deformation_base_path, "block")
        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():

            for layer_index, layer_name in enumerate(self.params.LAYER_NAMES):
                torch.cuda.empty_cache()

                layer_block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]
                layer_num_channels = self.params.LAYER_NAME_TO_CHANNELS[layer_name]

                assert layer_block_sizes[0][0] == 1 and layer_block_sizes[0][1] == 1

                cur_block_name = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
                next_block_name = self.params.IN_LAYER_PROXY_SPEC[layer_name]

                noise_f_name, signal_f_name, loss_deform_f_name, loss_baseline_f_name, noise, signal, loss_deform = \
                    self._get_files_and_matrices(batch_index, output_path, layer_name, len(layer_block_sizes), layer_num_channels)
                if os.path.exists(noise_f_name) and os.path.exists(signal_f_name):
                    continue

                for channel in tqdm(range(layer_num_channels), desc=f"Batch={batch_index} Layer={layer_index}"):
                    for block_size_index in range(len(layer_block_sizes)):
                        if block_size_index == 0:
                            continue

                        block_size_indices = np.zeros(shape=layer_num_channels, dtype=np.int32)
                        block_size_indices[channel] = block_size_index
                        block_size_spec = {layer_name: block_size_indices}
                        block_size_spec = {
                            layer_name: np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])[index] for
                            layer_name, index in block_size_spec.items()}

                        noise_val, signal_val, loss_deform_val = \
                            self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                                  ground_truth=ground_truth,
                                                  loss_ce=loss_ce,
                                                  block_size_spec=block_size_spec,
                                                  input_block_name=cur_block_name,
                                                  output_block_name=next_block_name)

                        noise[block_size_index, channel] = noise_val
                        signal[block_size_index, channel] = signal_val
                        loss_deform[block_size_index, channel] = loss_deform_val

                np.save(file=noise_f_name, arr=noise)
                np.save(file=signal_f_name, arr=signal)
                np.save(file=loss_deform_f_name, arr=loss_deform)
                np.save(file=loss_baseline_f_name, arr=np.array([loss_ce]))

    def collect_deformation_by_blocks(self):

        input_dir = os.path.join(self.deformation_base_path, "block")
        out_dir_format = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "channels", "{}", f"collect_0")

        for layer_index, layer_name in tqdm(enumerate(self.params.LAYER_NAMES)):
            out_dir = out_dir_format.format(layer_name)
            os.makedirs(out_dir, exist_ok=True)

            reduction_to_block_sizes_f_name = os.path.join(out_dir, f"reduction_to_block_sizes.npy")
            redundancy_arr_f_name = os.path.join(out_dir, f"redundancy_arr.npy")

            if os.path.exists(reduction_to_block_sizes_f_name) and os.path.exists(redundancy_arr_f_name):
                continue

            num_groups = self.params.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[layer_name][0]
            files = glob.glob(os.path.join(input_dir, f"signal_{layer_name}_batch_*.npy"))

            signal = np.stack([np.load(f) for f in files])
            noise = np.stack([np.load(f.replace("signal", "noise")) for f in files])
            noise = noise.mean(axis=0)
            signal = signal.mean(axis=0)
            deformation = noise / signal
            deformation[0] = 0
            target_deformation = np.unique(deformation)

            assert not np.any(np.isnan(deformation))

            block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]
            activation_reduction = np.array([1 / x[0] / x[1] for x in block_sizes])
            deformation_and_channel_to_block_size = []
            deformation_and_channel_to_reduction = []

            broadcast_block_size = np.repeat(np.array([x[0] * x[1] for x in block_sizes])[:, np.newaxis],
                                             deformation.shape[1], axis=1)
            for cur_target_deformation_index, cur_target_deformation in enumerate(target_deformation):
                valid_block_sizes = deformation <= cur_target_deformation
                block_sizes_with_zero_on_non_valid_blocks = broadcast_block_size * valid_block_sizes

                # TODO: we disregard the case of two block sizes with similar size (e.g. [2,1],[1,2] that are
                #  below deformation threshold. We would like to pick the optimal one
                cur_block_sizes = np.argmax(block_sizes_with_zero_on_non_valid_blocks, axis=0)
                cur_reduction = activation_reduction[cur_block_sizes]

                deformation_and_channel_to_block_size.append(cur_block_sizes)
                deformation_and_channel_to_reduction.append(cur_reduction)

            deformation_and_channel_to_reduction = np.array(deformation_and_channel_to_reduction)
            deformation_and_channel_to_block_size = np.array(deformation_and_channel_to_block_size)

            channels = deformation_and_channel_to_reduction.shape[1]
            group_size = channels // num_groups

            # deformation_and_channel_to_reduction[deformation_index, channel_index] gives us the reduction due to
            # using the appropriate block sizes. multi_channel_reduction[deformation_index, channel_group_index] gives
            # us the reduction of the group due to using the relevant block_sizes
            # multi_channel_reduction.shape = deformation x num_of_channels_in_a_group

            # To summarize, if we set deformation = target_deformation[deformation_index] and we try to stretch every
            # channel's block size to be the largest one, as long as the incurred deformation is not exceeded, and then
            # we collect the derived block sizes along a group of channel (channel_group_index), then
            # multi_channel_reduction[deformation_index, channel_group_index] = the resulted group reduction
            multi_channel_reduction = deformation_and_channel_to_reduction.reshape(
                (target_deformation.shape[0], num_groups, group_size)).mean(axis=-1)

            # Here we extract the final block sizes.
            reduction_to_block_sizes = np.zeros((TARGET_REDUCTIONS.shape[0], channels))
            for target_reduction_index, target_reduction in enumerate(TARGET_REDUCTIONS):

                # indices are the deformation index that needs to be used in order to get the proper reduction
                indices = np.argmin(np.abs(target_reduction - multi_channel_reduction), axis=0)

                # say:
                # channels = 64
                # group_size = 2
                # indices = [8392, 3526,  241, 5043, 3278, 9433, 1236, 3941, 1862, 2546,  217, 3106, 2049,  577, 3887, 2386, ...]
                #
                # then multi_channel_reduction[8392, 0] is approximately target_reduction. And this is the deformation that is due to
                # And so we wish channel-0 to use: deformation_and_channel_to_block_size[8392, 0].
                # And channel-1 to use: deformation_and_channel_to_block_size[8392, 1].
                indices = np.repeat(indices, group_size)
                reduction_to_block_sizes[target_reduction_index] = \
                    deformation_and_channel_to_block_size[indices, np.arange(0, channels)]

            redundancy_arr = self._get_redundancy_arr(reduction_to_block_sizes, num_groups, group_size)

            np.save(file=redundancy_arr_f_name, arr=redundancy_arr)
            np.save(file=reduction_to_block_sizes_f_name, arr=reduction_to_block_sizes)

    def extract_deformation_by_channels(self, batch_index):

        resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations(batch_index)

        input_path_format = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "channels", "{}", f"collect_{self.hierarchy_level}")
        output_path_format = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "channels", "{}", f"extract_{self.hierarchy_level}")


        with torch.no_grad():

            for layer_index, layer_name in enumerate(self.params.LAYER_NAMES):
                output_path = output_path_format.format(layer_name)
                input_path = input_path_format.format(layer_name)

                if len(self.params.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[layer_name]) <= self.hierarchy_level:
                    continue

                os.makedirs(output_path, exist_ok=True)

                torch.cuda.empty_cache()

                num_of_groups = self.params.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[layer_name][self.hierarchy_level]

                redundancy_arr_f_name = os.path.join(input_path, f"redundancy_arr.npy")
                redundancy_arr = np.load(redundancy_arr_f_name)

                reduction_to_block_sizes = np.load(os.path.join(input_path, f"reduction_to_block_sizes.npy"))

                cur_block_name = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
                if num_of_groups > 1:
                    next_block_name = self.params.IN_LAYER_PROXY_SPEC[layer_name]
                else:
                    next_block_name = None

                noise_f_name, signal_f_name, loss_deform_f_name, loss_baseline_f_name, noise, signal, loss_deform = \
                    self._get_files_and_matrices(batch_index, output_path, layer_name, TARGET_REDUCTIONS.shape[0], num_of_groups)

                if os.path.exists(noise_f_name) and os.path.exists(signal_f_name) and os.path.exists(loss_deform_f_name):
                    continue
                channels = self.params.LAYER_NAME_TO_CHANNELS[layer_name]
                group_size = channels // num_of_groups
                assert group_size * num_of_groups == channels

                for channel_group in tqdm(range(num_of_groups), desc=f"Batch={batch_index} Layer={layer_index}"):
                    for reduction_index, reduction in enumerate(TARGET_REDUCTIONS):

                        red_arr_val = redundancy_arr[reduction_index, channel_group].astype(np.int32)

                        if redundancy_arr[reduction_index, channel_group] != reduction_index:
                            assert red_arr_val < reduction_index
                            noise_val = noise[red_arr_val, channel_group]
                            signal_val = signal[red_arr_val, channel_group]
                            loss_deform_val = loss_deform[red_arr_val, channel_group]

                        else:

                            block_size_indices = np.zeros(shape=channels, dtype=np.int32)
                            ind_start = channel_group * group_size
                            ind_end = ind_start + group_size
                            block_size_indices[ind_start:ind_end] = \
                                reduction_to_block_sizes[reduction_index, ind_start:ind_end]
                            block_size_spec = {layer_name: block_size_indices}
                            block_size_spec = {
                                layer_name: np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])[index] for
                                layer_name, index in block_size_spec.items()}

                            noise_val, signal_val, loss_deform_val = \
                                self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                                      ground_truth=ground_truth,
                                                      loss_ce=loss_ce,
                                                      block_size_spec=block_size_spec,
                                                      input_block_name=cur_block_name,
                                                      output_block_name=next_block_name)

                        noise[reduction_index, channel_group] = noise_val
                        signal[reduction_index, channel_group] = signal_val
                        loss_deform[reduction_index, channel_group] = loss_deform_val

                np.save(file=noise_f_name, arr=noise)
                np.save(file=signal_f_name, arr=signal)
                np.save(file=loss_deform_f_name, arr=loss_deform)
                np.save(file=loss_baseline_f_name, arr=np.array([loss_ce]))

    def collect_deformation_by_channels(self):

        prev_input_path_format = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "channels", "{}", f"collect_{self.hierarchy_level}")
        input_path_format = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "channels", "{}", f"extract_{self.hierarchy_level}")
        output_path_format = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "channels", "{}", f"collect_{self.hierarchy_level + 1}")

        for layer_index, layer_name in tqdm(enumerate(self.params.LAYER_NAMES)):
            prev_input_path = prev_input_path_format.format(layer_name)
            input_path = input_path_format.format(layer_name)
            output_path = output_path_format.format(layer_name)

            if len(self.params.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[layer_name]) <= self.hierarchy_level + 1:
                continue

            os.makedirs(output_path, exist_ok=True)

            num_groups_prev = self.params.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[layer_name][self.hierarchy_level]
            num_groups_curr = self.params.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[layer_name][self.hierarchy_level + 1]
            block_size_index_to_reduction = 1 / np.prod(np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]), axis=1)
            redundancy_arr_path = os.path.join(output_path, f"redundancy_arr.npy")
            reduction_to_block_size_new_path = os.path.join(output_path, f"reduction_to_block_sizes.npy")
            deformation_index_to_reduction_path = os.path.join(output_path, f"deformation_index_to_reduction.npy")
            # if os.path.exists(reduction_to_block_size_new_path) and os.path.exists(redundancy_arr_path):
            #     continue
            reduction_to_block_size = np.load(
                os.path.join(prev_input_path, f"reduction_to_block_sizes.npy"))

            # if False: #num_groups_prev == num_groups_curr:
            #     files = glob.glob(os.path.join(input_path, f"loss_deform_{layer_name}_batch_*.npy"))
            #     deformation = np.stack([np.load(f) for f in files]).mean(axis=0)
            #     assert deformation.max() <= target_deformation[-1], deformation.max()
            #
            #     assert not np.any(np.isnan(deformation))
            # else:
            files = glob.glob(os.path.join(input_path, f"signal_{layer_name}_batch_*.npy"))
            signal = np.stack([np.load(f) for f in files])
            noise = np.stack([np.load(f.replace("signal", "noise")) for f in files])
            noise = noise.mean(axis=0)
            signal = signal.mean(axis=0)
            deformation = noise / signal
            target_deformation = np.unique(deformation)

            assert not np.any(np.isnan(deformation))

            channels = reduction_to_block_size.shape[1]
            group_size_prev = channels // num_groups_prev
            group_size_curr = channels // num_groups_curr

            assert num_groups_prev * group_size_prev == channels
            assert num_groups_curr * group_size_curr == channels

            deformation_index_to_reduction_index = []
            chosen_block_sizes = []
            for cur_target_deformation_index, cur_target_deformation in enumerate(target_deformation):
                valid_reductions = deformation <= cur_target_deformation

                channel_block_reduction_index = np.argmax((TARGET_REDUCTIONS / 2)[::-1][:, np.newaxis] + valid_reductions, axis=0)
                channel_block_reduction_index = np.repeat(channel_block_reduction_index, group_size_prev)

                # We set deformation = target_deformation[cur_target_deformation_index]
                # We find the highest reduction that does not exceed this deformation.
                # channel_block_reduction_index[group_index] is the index of this reduction.

                # We now find the index of the block_sizes that are responsible for this reduction. Which is block_sizes.

                # Meaning that if for example, cur_target_deformation_index = 1000, channels=64 and group_size_prev = 2.
                # Then if we use block_sizes[0] for channel 0 and block_sizes[1] for channel 1. The deformation would not exceed
                # target_deformation[deformation_index]. Moreover, this would be the highest block_sizes that we can use that won't
                # exceed this deformation
                block_sizes = reduction_to_block_size[channel_block_reduction_index, range(channels)]
                chosen_block_sizes.append(block_sizes)
                deformation_index_to_reduction_index.append(channel_block_reduction_index)

            chosen_block_sizes = np.array(chosen_block_sizes)
            deformation_index_to_reduction_index = np.array(deformation_index_to_reduction_index)

            # assert False
            # if num_groups_prev == num_groups_curr:
            deformation_index_to_reduction = block_size_index_to_reduction[chosen_block_sizes.flatten().astype(np.int32)]
            deformation_index_to_reduction = deformation_index_to_reduction.reshape(chosen_block_sizes.shape)
            deformation_index_to_reduction = deformation_index_to_reduction.reshape((target_deformation.shape[0], num_groups_curr, group_size_curr)).mean(axis=-1)
            # else:
            #     deformation_index_to_reduction = TARGET_REDUCTIONS[deformation_index_to_reduction_index.flatten()]
            #     deformation_index_to_reduction = deformation_index_to_reduction.reshape(deformation_index_to_reduction_index.shape)
            #     deformation_index_to_reduction = deformation_index_to_reduction.reshape((target_deformation.shape[0], num_groups_curr, group_size_curr)).mean(axis=-1)

            reduction_to_block_size_new = np.zeros((TARGET_REDUCTIONS.shape[0], channels))
            for target_reduction_index, target_reduction in enumerate(TARGET_REDUCTIONS):
                indices = np.argmin(np.abs(target_reduction - deformation_index_to_reduction), axis=0)
                indices = np.repeat(indices, group_size_curr)
                reduction_to_block_size_new[target_reduction_index] = chosen_block_sizes[indices, np.arange(0, channels)]

            redundancy_arr = self._get_redundancy_arr(reduction_to_block_size_new, num_groups_curr, group_size_curr)

            np.save(file=redundancy_arr_path, arr=redundancy_arr)
            np.save(file=reduction_to_block_size_new_path, arr=reduction_to_block_size_new)
            np.save(file=deformation_index_to_reduction_path, arr=deformation_index_to_reduction)

    def extract_deformation_by_layers(self, batch_index):

        resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations(batch_index)

        input_path_format = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "layers", "{}", f"collect_{self.hierarchy_level}")
        output_path_format = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "layers", "{}", f"extract_{self.hierarchy_level}")
        red_format = os.path.join(input_path_format, "reduction_to_block_sizes.npy")
        # os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():

            for layer_group_index, layer_group in enumerate(self.params.LAYER_HIERARCHY_SPEC[self.hierarchy_level]):
                # assert False, "Fix this assumption"
                # TODO: change this assumption
                layer_name_of_first_layer_in_group = layer_group[0]
                output_path = output_path_format.format(layer_name_of_first_layer_in_group)

                noise_f_name, signal_f_name, loss_deform_f_name, loss_baseline_f_name, noise, signal, loss_deform = \
                    self._get_files_and_matrices(batch_index, output_path, layer_name_of_first_layer_in_group, TARGET_REDUCTIONS.shape[0], 1)

                if os.path.exists(noise_f_name) and os.path.exists(signal_f_name):
                    continue

                reduction_to_block_sizes = \
                    {
                        layer_name: np.load(red_format.format(layer_name)) for layer_name in layer_group
                    }

                torch.cuda.empty_cache()

                cur_block_name = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name_of_first_layer_in_group]
                next_block_name = None

                for reduction_index, reduction in tqdm(enumerate(TARGET_REDUCTIONS), desc=f"Layer-Group = {layer_group_index}"):

                    block_size_spec = {layer_name: reduction_to_block_sizes[layer_name][reduction_index].astype(np.int32) for layer_name in layer_group}
                    block_size_spec = {layer_name: np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])[index] for layer_name, index in block_size_spec.items()}
                    noise_val, signal_val, loss_deform_val = \
                        self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                              ground_truth=ground_truth,
                                              loss_ce=loss_ce,
                                              block_size_spec=block_size_spec,
                                              input_block_name=cur_block_name,
                                              output_block_name=next_block_name)

                    noise[reduction_index, 0] = noise_val
                    signal[reduction_index, 0] = signal_val
                    loss_deform[reduction_index, 0] = loss_deform_val

                os.makedirs(output_path, exist_ok=True)

                np.save(file=noise_f_name, arr=noise)
                np.save(file=signal_f_name, arr=signal)
                np.save(file=loss_deform_f_name, arr=loss_deform)

    def get_block_spec(self):
        output_path = os.path.join(self.deformation_base_path,self.params.HIERARCHY_NAME, "reduction_specs")
        input_path = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "layers", "{}", f"collect_{self.hierarchy_level}")
        os.makedirs(output_path, exist_ok=True)
        reductions = [0.062,0.083,0.111]
        for target_reduction_index, target_reduction in enumerate(reductions):
            red_spec = {}
            for layer_name in self.params.LAYER_NAMES:
                reduction_index = np.argwhere(TARGET_REDUCTIONS == target_reduction)[0, 0]
                block_sizes_to_use = np.load(os.path.join(input_path.format(layer_name), f"reduction_to_block_sizes.npy"))[reduction_index].astype(np.int32)
                red_spec[layer_name] = np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])[block_sizes_to_use]

            relu_tot = sum(self.params.LAYER_NAME_TO_RELU_COUNT.values())
            count = 0
            for layer_name in self.params.LAYER_NAMES:
                count += (1 / red_spec[layer_name][:,0] / red_spec[layer_name][:,1]).mean() * self.params.LAYER_NAME_TO_RELU_COUNT[layer_name]
            count /= relu_tot
            print(target_reduction, count)
            reduction_spec_file = os.path.join(output_path, "layer_reduction_{:.2f}.pickle".format(target_reduction))
            with open(reduction_spec_file, 'wb') as f:
                pickle.dump(obj=red_spec, file=f)

    def collect_deformation_by_layers(self):

        cache_dir = "/home/yakir/Data2/cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)

        for group_index, layers_in_group in tqdm(enumerate(self.params.LAYER_HIERARCHY_SPEC[self.hierarchy_level + 1])):
            deformation_index_to_reductions = []
            relu_counts = []
            # TODO: just to extract the target_deformation...
            if self.hierarchy_level == -1:
                target_deformations = []
                for layer_index, layer_name in tqdm(enumerate(layers_in_group)):

                    ll = len(self.params.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[layer_name]) - 1
                    representative = layer_name
                    input_path = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "channels", layer_name, f"extract_{ll}",f"signal_{representative}_batch_*.npy")
                    files = glob.glob(input_path)
                    # # TODO: it happens many times - put in a function
                    signal = np.stack([np.load(f) for f in files])
                    noise = np.stack([np.load(f.replace("signal", "noise")) for f in files])
                    noise = noise.mean(axis=0)
                    signal = signal.mean(axis=0)
                    deformation = noise / signal
                    assert not np.any(np.isnan(deformation))
                    target_deformations.append(np.unique(deformation))
                target_deformation = np.unique(np.hstack(target_deformations))
            else:
                target_deformation = None

            for layer_index, layer_name in tqdm(enumerate(layers_in_group)):

                block_size_index_to_reduction = 1 / np.prod(np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]), axis=1)

                if self.hierarchy_level == -1:
                    ll = len(self.params.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[layer_name]) - 1
                    reduction_to_block_size = np.load(os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "channels", layer_name, f"collect_{ll}", "reduction_to_block_sizes.npy"))
                    representative = layer_name
                    input_path = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "channels", layer_name, f"extract_{ll}",f"signal_{representative}_batch_*.npy")
                    files = glob.glob(input_path)
                else:
                    prev_input_path = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "layers", layer_name, f"collect_{self.hierarchy_level}", "reduction_to_block_sizes.npy")
                    reduction_to_block_size = np.load(prev_input_path)

                    prev_group_to_representative = {layer: group[0] for group in
                                                    self.params.LAYER_HIERARCHY_SPEC[self.hierarchy_level] for layer in
                                                    group}
                    representative = prev_group_to_representative[layer_name]
                    input_path = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, f"layers", representative, f"extract_{self.hierarchy_level}", f"signal_{representative}_batch_*.npy")
                    files = glob.glob(input_path)
                # # TODO: it happens many times - put in a function

                signal = np.stack([np.load(f) for f in files])
                noise = np.stack([np.load(f.replace("signal", "noise")) for f in files])
                noise = noise.mean(axis=0)
                signal = signal.mean(axis=0)
                deformation = noise / signal

                if target_deformation is None:
                    target_deformation = np.unique(deformation)

                assert not np.any(np.isnan(deformation))

                channels = reduction_to_block_size.shape[1]

                chosen_block_sizes = []
                for cur_target_deformation_index, cur_target_deformation in enumerate(target_deformation):
                    valid_reductions = deformation <= cur_target_deformation

                    # TODO: happens many times..
                    channel_block_reduction_index = np.argmax((TARGET_REDUCTIONS / 2)[::-1][:, np.newaxis] + valid_reductions, axis=0)

                    block_sizes = reduction_to_block_size[channel_block_reduction_index, range(channels)]

                    chosen_block_sizes.append(block_sizes)

                chosen_block_sizes = np.array(chosen_block_sizes)

                deformation_index_to_reduction = block_size_index_to_reduction[chosen_block_sizes.flatten().astype(np.int32)]
                deformation_index_to_reduction = deformation_index_to_reduction.reshape(chosen_block_sizes.shape)
                deformation_index_to_reduction = deformation_index_to_reduction.mean(axis=-1)

                deformation_index_to_reductions.append(deformation_index_to_reduction)
                relu_counts.append(self.params.LAYER_NAME_TO_RELU_COUNT[layer_name])
                np.save(file=os.path.join(cache_dir, f"chosen_block_indices_{layer_name}.npy"), arr=chosen_block_sizes)

            deformation_index_to_reductions = np.array(deformation_index_to_reductions)
            relu_counts = np.array(relu_counts)[..., np.newaxis]

            deformation_index_to_reduction = (deformation_index_to_reductions * relu_counts).sum(axis=0) / sum(
                relu_counts)

            for layer_index, layer_name in enumerate(layers_in_group):
                chosen_block_sizes = np.load(os.path.join(cache_dir, f"chosen_block_indices_{layer_name}.npy"))
                channels = chosen_block_sizes.shape[1]
                reduction_to_block_size_new = np.zeros((TARGET_REDUCTIONS.shape[0], channels))
                for target_reduction_index, target_reduction in enumerate(TARGET_REDUCTIONS):
                    index = np.argmin(np.abs(target_reduction - deformation_index_to_reduction), axis=0)
                    reduction_to_block_size_new[target_reduction_index] = chosen_block_sizes[index]

                output_path = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME, "layers", f"{layer_name}", f"collect_{self.hierarchy_level + 1}")


                os.makedirs(output_path, exist_ok=True)
                reduction_to_block_size_new_path = os.path.join(output_path, f"reduction_to_block_sizes.npy")
                deformation_index_to_reduction_path = os.path.join(output_path, f"deformation_index_to_reduction.npy")

                np.save(file=reduction_to_block_size_new_path, arr=reduction_to_block_size_new)
                np.save(file=deformation_index_to_reduction_path, arr=deformation_index_to_reduction)

        shutil.rmtree(cache_dir)

    def _get_deformation_v2(self, resnet_block_name_to_activation, ground_truth, block_size_spec, input_block_name, output_block_names):

        input_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == input_block_name)[0, 0]
        output_block_indices = [np.argwhere(np.array(self.params.BLOCK_NAMES) == output_block_name)[0, 0] for output_block_name in output_block_names]

        input_tensor = resnet_block_name_to_activation[input_block_name]
        next_tensors = [resnet_block_name_to_activation[output_block_name] for output_block_name in output_block_names]

        layer_name_to_orig_layer = {}
        for layer_name, block_size_indices in block_size_spec.items():
            orig_layer = self.arch_utils.get_layer(self.model, layer_name)
            layer_name_to_orig_layer[layer_name] = orig_layer

            self.arch_utils.set_bReLU_layers(self.model, {layer_name: (block_size_indices,
                                                                       self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])})

        cur_out = input_tensor
        outs = []

        for block_index in range(input_block_index, max(output_block_indices)):
            cur_out = self.arch_utils.run_model_block(self.model, cur_out, self.params.BLOCK_NAMES[block_index])
            outs.append(cur_out)

        if output_block_names[-1] is None:
            loss_deform = self.model.decode_head.losses(cur_out, ground_truth)['loss_ce']
        else:
            loss_deform = None

        for layer_name_, orig_layer in layer_name_to_orig_layer.items():
            self.arch_utils.set_layers(self.model, {layer_name_: orig_layer})


        seg_logit = resize(
            input=outs[-1],
            size=(512, 512),
            mode='bilinear',
            align_corners=self.model.decode_head.align_corners)
        output = F.softmax(seg_logit, dim=1)

        seg_pred = output.argmax(dim=1)
        # seg_map = self.dataset[0]['gt_semantic_seg'].data
        try:
            results = [intersect_and_union(
                seg_pred.cpu().numpy(),
                ground_truth[0].cpu().numpy(),
                len(self.dataset.CLASSES),
                self.dataset.ignore_index,
                label_map=dict(),
                reduce_zero_label=self.dataset.reduce_zero_label)]
        except IndexError:
            print('fldskja')
        metric = self.dataset.evaluate(results, **{'metric': ['mIoU']})['mIoU']

        noises = []
        signals = []

        for out, next_tensor in zip(outs, next_tensors):
            noise = float(((out - next_tensor) ** 2).mean())
            signal = float((next_tensor ** 2).mean())
            noises.append(noise)
            signals.append(signal)

        return noises, signals, loss_deform.cpu().numpy(), metric



    def get_random_spec(self):
        channel_ord_to_layer_name = np.hstack([self.params.LAYER_NAME_TO_CHANNELS[layer_name] * [layer_name] for layer_name in self.params.LAYER_NAMES])
        channel_ord_to_channel_index = np.hstack([np.arange(self.params.LAYER_NAME_TO_CHANNELS[layer_name]) for layer_name in self.params.LAYER_NAMES])
        num_channels = len(channel_ord_to_layer_name)

        layers_to_noise = ["stem_2",
                           "stem_5",
                           "stem_8",
                           "layer1_0_1",
                           "layer1_0_2",
                           "layer1_0_3",
                           "layer1_1_1",
                           "layer1_1_2",
                           "layer1_1_3",
                           "layer1_2_1",
                           "layer1_2_2",
                           "layer1_2_3",
                           "layer2_0_1",
                           "layer2_0_2",
                           "layer2_0_3",
                           "layer2_1_1",
                           "layer2_1_2",
                           "layer2_1_3",
                           "layer2_2_1",
                           "layer2_2_2",
                           "layer2_2_3",
                           "layer2_3_1",
                           "layer2_3_2",
                           "layer2_3_3", ]

        num_channel_to_add_noise_to = num_channels #// 2  # np.random.randint(low=0, high=num_channels)
        channel_sample = np.random.choice(num_channels, size=num_channel_to_add_noise_to, replace=False)
        layer_and_channels = [(channel_ord_to_layer_name[x], channel_ord_to_channel_index[x]) for x in channel_sample]
        # block_size_pseudo_indices = np.random.choice(5, size=num_channel_to_add_noise_to, replace=True)
        block_size_indices = [np.random.randint(1, len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])) for
                              layer_name, _ in layer_and_channels]
        block_size_spec = {
            layer_name: np.zeros(shape=(self.params.LAYER_NAME_TO_CHANNELS[layer_name],), dtype=np.int32) for
            layer_name in self.params.LAYER_NAMES if layer_name in layers_to_noise}

        for index, (layer_name, channel) in zip(block_size_indices, layer_and_channels):
            if layer_name in layers_to_noise:
                block_size_spec[layer_name][channel] = index

        return block_size_spec
    def get_distortion_and_miou_stats(self):

        self.batch_size = 1
        os.makedirs(self.deformation_base_path, exist_ok=True)
        with torch.no_grad():

            torch.cuda.empty_cache()
            results_noise = []
            results_orig = []
            while True:
                for batch_index in tqdm(range(len(self.dataset))):
                    resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations(batch_index)

                    block_size_spec = self.get_random_spec()

                    try:
                        results_noise.append(
                            self._get_deformation_v2(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                                     ground_truth=ground_truth,
                                                     block_size_spec=block_size_spec,
                                                     input_block_name="stem",
                                                     output_block_names=[None]))

                        results_orig.append(
                            self._get_deformation_v2(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                                     ground_truth=ground_truth,
                                                     block_size_spec={},
                                                     input_block_name="stem",
                                                     output_block_names=[None]))
                    except:
                        pass
                        print('LALA')
                    #
                    if batch_index % 10 == 0:
                        pickle.dump(obj=results_noise, file=open(os.path.join(self.deformation_base_path, "results_noise.pickle"), 'wb'))
                        pickle.dump(obj=results_orig, file=open(os.path.join(self.deformation_base_path, "results_orig.pickle"), 'wb'))

    def get_per_layer_distortion_and_miou_stats(self):
        from collections import defaultdict
        self.batch_size = 1
        our_dir = self.deformation_base_path
        os.makedirs(our_dir, exist_ok=True)
        layer_name = 'layer2_0_1'
        layer_name = 'layer1_1_2'
        layer_name = 'layer3_1_1'
        d = defaultdict(list)
        for batch_index, batch_id in enumerate(range(5000)):

            try:
                resnet_block_name_to_activation, ground_truth, loss_orig = self.get_activations(batch_id)
            except ValueError:
                continue


            block_size_spec = {layer_name: np.random.randint(low=0,
                                                             high=len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]),
                                                             size=self.params.LAYER_NAME_TO_CHANNELS[layer_name])}
            ratio = np.mean([1/x[0]/x[1] for x in np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])[block_size_spec[layer_name]]])
            with torch.no_grad():

                noises, signals, loss, miou = self._get_deformation_v2(
                    resnet_block_name_to_activation=resnet_block_name_to_activation,
                    ground_truth=ground_truth,
                    block_size_spec=block_size_spec,
                    input_block_name="stem",
                    output_block_names=self.params.BLOCK_NAMES[1:])

                noises_baseline, signals_baseline, loss_baseline, miou_baseline = self._get_deformation_v2(
                    resnet_block_name_to_activation=resnet_block_name_to_activation,
                    ground_truth=ground_truth,
                    block_size_spec={},
                    input_block_name="stem",
                    output_block_names=self.params.BLOCK_NAMES[1:])

                # assert loss_orig == loss_baseline
                d["noises"].append(noises)
                d["signals"].append(signals)
                d["loss"].append(loss)
                d["miou"].append(miou)
                d["noises_baseline"].append(noises_baseline)
                d["signals_baseline"].append(signals_baseline)
                d["loss_baseline"].append(loss_baseline)
                d["miou_baseline"].append(miou_baseline)
                d["ratio"].append(ratio)

            if batch_index % 10 == 9:
                pickle.dump(obj=d, file=open(os.path.join(self.deformation_base_path, f"data_{layer_name}.pickle"), 'wb'))






    def foo(self, batch_index):
        resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations(batch_index)

        output_path = os.path.join(self.deformation_base_path, "block")
        os.makedirs(output_path, exist_ok=True)

        layer_name = "layer2_2_2"

        with torch.no_grad():

            torch.cuda.empty_cache()

            layer_block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]
            layer_num_channels = self.params.LAYER_NAME_TO_CHANNELS[layer_name]

            assert layer_block_sizes[0][0] == 1 and layer_block_sizes[0][1] == 1

            cur_block_name = self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name]
            next_block_name = self.params.IN_LAYER_PROXY_SPEC[layer_name]


            block_index_to_use = [0, 78]
            noise = np.zeros(shape=[len(block_index_to_use)] * 20)
            block_sizes = [(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19)
                           for i0 in  range(2)
                           for i1 in  range(2)
                           for i2 in  range(2)
                           for i3 in  range(2)
                           for i4 in  range(2)
                           for i5 in  range(2)
                           for i6 in  range(2)
                           for i7 in  range(2)
                           for i8 in  range(2)
                           for i9 in  range(2)
                           for i10 in range(2)
                           for i11 in range(2)
                           for i12 in range(2)
                           for i13 in range(2)
                           for i14 in range(2)
                           for i15 in range(2)
                           for i16 in range(2)
                           for i17 in range(2)
                           for i18 in range(2)
                           for i19 in range(2)
                           ]
            counter = 0
            for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19  in tqdm(block_sizes):
                counter += 1
                block_size_indices = np.zeros(shape=layer_num_channels, dtype=np.int32)
                block_size_indices[0] = block_index_to_use[i0]
                block_size_indices[1] = block_index_to_use[i1]
                block_size_indices[2] = block_index_to_use[i2]
                block_size_indices[3] = block_index_to_use[i3]
                block_size_indices[4] = block_index_to_use[i4]
                block_size_indices[5] = block_index_to_use[i5]
                block_size_indices[6] = block_index_to_use[i6]
                block_size_indices[7] = block_index_to_use[i7]
                block_size_indices[8] = block_index_to_use[i8]
                block_size_indices[9] = block_index_to_use[i9]
                block_size_indices[10] = block_index_to_use[i10]
                block_size_indices[11] = block_index_to_use[i11]
                block_size_indices[12] = block_index_to_use[i12]
                block_size_indices[13] = block_index_to_use[i13]
                block_size_indices[14] = block_index_to_use[i14]
                block_size_indices[15] = block_index_to_use[i15]
                block_size_indices[16] = block_index_to_use[i16]
                block_size_indices[17] = block_index_to_use[i17]
                block_size_indices[18] = block_index_to_use[i18]
                block_size_indices[19] = block_index_to_use[i19]

                noise_val, _, _ = \
                    self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                          ground_truth=ground_truth,
                                          loss_ce=loss_ce,
                                          block_size_spec={layer_name: block_size_indices},
                                          input_block_name=cur_block_name,
                                          output_block_name=next_block_name)

                noise[i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19] = noise_val

                if counter %(4**8) == 0:
                    np.save(file=os.path.join(self.deformation_base_path, "tmp_3.npy"), arr=noise)
            np.save(file=os.path.join(self.deformation_base_path, "tmp_3.npy"), arr=noise)

    def foo_2(self, batch_index):
        resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations(batch_index)

        output_path = os.path.join(self.deformation_base_path, "block")
        os.makedirs(output_path, exist_ok=True)

        noise_vector_single = []
        noise_vector_agg = []
        block_size_spec = {}

        layer_names = [
            "layer2_0_1",
            "layer2_0_2",
            "layer2_0_3",
            "layer2_1_1",
            "layer2_1_2",
            "layer2_1_3",
            "layer2_2_1",
            "layer2_2_2",
            "layer2_2_3",
            "layer2_3_1",
            "layer2_3_2",
            "layer2_3_3",
        ]
        init_block = "layer2_0"
        with torch.no_grad():

            torch.cuda.empty_cache()
            for layer_name in layer_names:
                layer_block_sizes = self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]
                layer_num_channels = self.params.LAYER_NAME_TO_CHANNELS[layer_name]
                block_to_use = np.argwhere(np.all(np.array(layer_block_sizes) == [2, 2], axis=1))[0,0]
                agg_block_size_indices = np.zeros(shape=layer_num_channels, dtype=np.int32)

                for channel in tqdm(range(layer_num_channels)):

                    block_size_indices = np.zeros(shape=layer_num_channels, dtype=np.int32)
                    block_size_indices[channel] = block_to_use
                    agg_block_size_indices[channel] = block_to_use

                    noise_val, _, _ = \
                        self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                              ground_truth=ground_truth,
                                              loss_ce=loss_ce,
                                              block_size_spec={layer_name: block_size_indices},
                                              input_block_name=init_block,
                                              output_block_name=None)

                    noise_vector_single.append(noise_val)

                    block_size_spec[layer_name] = agg_block_size_indices

                    noise_val, _, _ = \
                        self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                              ground_truth=ground_truth,
                                              loss_ce=loss_ce,
                                              block_size_spec=block_size_spec,
                                              input_block_name=init_block,
                                              output_block_name=None)

                    noise_vector_agg.append(noise_val)


                np.save(file=os.path.join(self.deformation_base_path, "noise_vector_single.npy"), arr=noise_vector_single)
                np.save(file=os.path.join(self.deformation_base_path, "noise_vector_agg.npy"), arr=noise_vector_agg)

    def foo_3(self):
        self.batch_size = 1
        os.makedirs(self.deformation_base_path, exist_ok=True)

        channel_ord_to_layer_name = np.hstack([self.params.LAYER_NAME_TO_CHANNELS[layer_name] * [layer_name] for layer_name in self.params.LAYER_NAMES])
        channel_ord_to_channel_index = np.hstack([np.arange(self.params.LAYER_NAME_TO_CHANNELS[layer_name]) for layer_name in self.params.LAYER_NAMES])

        num_channels = len(channel_ord_to_layer_name)
        noises_real = []
        noises_agg = []
        for batch_index in np.arange(self.gpu_id, 2000, 2):
            print(batch_index)
            resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations(batch_index)

            num_channel_to_add_noise_to = num_channels // 10 #np.random.randint(low=0, high=num_channels)

            channel_sample = np.random.choice(num_channels, size=num_channel_to_add_noise_to, replace=False)
            layer_and_channels = [(channel_ord_to_layer_name[x], channel_ord_to_channel_index[x]) for x in channel_sample]
            # block_size_pseudo_indices = np.random.choice(5, size=num_channel_to_add_noise_to, replace=True)
            block_size_indices = [np.random.randint(1, len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])) for layer_name, _ in layer_and_channels]
            block_size_spec = {layer_name: np.zeros(shape=(self.params.LAYER_NAME_TO_CHANNELS[layer_name],), dtype=np.int32) for layer_name in self.params.LAYER_NAMES}

            for index, (layer_name, channel) in zip(block_size_indices, layer_and_channels):
                block_size_spec[layer_name][channel] = index

            with torch.no_grad():
                noise_real, _, _ = \
                    self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                          ground_truth=ground_truth,
                                          loss_ce=None,
                                          block_size_spec=block_size_spec,
                                          input_block_name="stem",
                                          output_block_name=None)

                noise_agg = 0
                for index, (layer_name, channel) in tqdm(list(zip(block_size_indices, layer_and_channels))):
                    block_size_indices = np.zeros(shape=self.params.LAYER_NAME_TO_CHANNELS[layer_name], dtype=np.int32)
                    block_size_indices[channel] = index

                    noise, _, _ = \
                        self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                              ground_truth=ground_truth,
                                              loss_ce=None,
                                              block_size_spec={layer_name: block_size_indices},
                                              input_block_name=self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name],
                                              output_block_name=None)

                    noise_agg += noise
            noises_real.append(noise_real)
            noises_agg.append(noise_agg)
            np.save(file=os.path.join(self.deformation_base_path, f"noise_real_{self.gpu_id}.npy"), arr=noises_real)
            np.save(file=os.path.join(self.deformation_base_path, f"noise_estimated_{self.gpu_id}.npy"), arr=noises_agg)

        print('YEY!!')

    def pre_layer_distortion_to_additive_estimated_distortion(self):
        self.batch_size = 1
        use_last_layer = True
        if use_last_layer:
            s = 20
            out_dir = os.path.join(self.deformation_base_path, "use_last_layer")
        else:
            s = -1
            out_dir = self.deformation_base_path
        print(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # if os.path.exists(os.path.join(our_dir, f"noise_real_mat_{self.gpu_id}.npy")):
        #     noise_real_mat = np.load(file=os.path.join(our_dir, f"noise_real_mat_{self.gpu_id}.npy"))
        #     noise_estimated_mat = np.load(file=os.path.join(our_dir, f"noise_estimated_mat_{self.gpu_id}.npy"))
        # else:

        from collections import defaultdict
        assets = [defaultdict(list) for _ in range(57)]

        for batch_index, batch_id in enumerate(np.arange(self.gpu_id, 5000, 2)):

            print(batch_index, batch_id)
            resnet_block_name_to_activation, ground_truth, _ = self.get_activations(batch_id)

            if use_last_layer:
                layer_enumerator = enumerate(self.params.LAYER_NAMES[:s])
            else:
                layer_enumerator = tqdm(enumerate(self.params.LAYER_NAMES), desc=f"{batch_index} ({batch_id})")

            for layer_index, layer_name in layer_enumerator:

                output_block_name = None if use_last_layer else self.params.IN_LAYER_PROXY_SPEC[layer_name]
                block_size_spec = {layer_name: np.random.randint(low=0,
                                                                 high=len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]),
                                                                 size=self.params.LAYER_NAME_TO_CHANNELS[layer_name])}

                with torch.no_grad():

                    noise_clean, signal_clean, loss_clean, mIoU_clean = \
                        self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                              ground_truth=ground_truth,
                                              loss_ce=None,
                                              block_size_spec={},
                                              input_block_name=self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name],
                                              output_block_name=output_block_name)

                    noise_real, signal_real, loss_real, mIoU_real = \
                        self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                              ground_truth=ground_truth,
                                              loss_ce=None,
                                              block_size_spec=block_size_spec,
                                              input_block_name=self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name],
                                              output_block_name=output_block_name)

                    if use_last_layer:
                        channel_enumerator = tqdm(range(self.params.LAYER_NAME_TO_CHANNELS[layer_name]), desc=f"{batch_index} ({batch_id}) - layer={layer_index}")
                    else:
                        channel_enumerator = range(self.params.LAYER_NAME_TO_CHANNELS[layer_name])

                    noise_channels, signal_channels, loss_channels, miou_channels = [],[],[],[]
                    for channel in channel_enumerator:
                        block_size_indices_channel = np.zeros(shape=self.params.LAYER_NAME_TO_CHANNELS[layer_name], dtype=np.int32)
                        block_size_indices_channel[channel] = block_size_spec[layer_name][channel]

                        noise_channel, signal_channel, loss_channel, miou_channel = \
                            self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                                  ground_truth=ground_truth,
                                                  loss_ce=None,
                                                  block_size_spec={layer_name: block_size_indices_channel},
                                                  input_block_name=self.params.LAYER_NAME_TO_BLOCK_NAME[layer_name],
                                                  output_block_name=output_block_name)

                        noise_channels.append(noise_channel)
                        signal_channels.append(signal_channel)
                        loss_channels.append(loss_channel)
                        miou_channels.append(miou_channel)

                assets[layer_index]["noise_channels"].append(noise_channels)
                assets[layer_index]["signal_channels"].append(signal_channels)
                assets[layer_index]["loss_channels"].append(loss_channels)
                assets[layer_index]["miou_channels"].append(miou_channels)
                assets[layer_index]["noise_real"].append(noise_real)
                assets[layer_index]["signal_real"].append(signal_real)
                assets[layer_index]["loss_real"].append(loss_real)
                assets[layer_index]["mIoU_real"].append(mIoU_real)
                assets[layer_index]["noise_clean"].append(noise_clean)
                assets[layer_index]["signal_clean"].append(signal_clean)
                assets[layer_index]["loss_clean"].append(loss_clean)
                assets[layer_index]["mIoU_clean"].append(mIoU_clean)

            pickle.dump(obj=assets, file=open(os.path.join(out_dir, f"assets_{self.gpu_id}.pickle"), 'wb'))





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_index', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    # parser.add_argument('--batch_index', type=str, default=0)
    # parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--hierarchy_level', type=int, default=4)
    parser.add_argument('--hierarchy_type', type=str, default="layers")
    parser.add_argument('--operation', type=str, default="get_reduction_spec")
    # parser.add_argument('--operation', type=str, default="get_distortion_and_miou_stats")
    # parser.add_argument('--param_json_file', type=str)
    parser.add_argument('--param_json_file', type=str, default="/home/yakir/PycharmProjects/secure_inference/research/block_relu/distortion_handler_configs/resnet_COCO_164K_8_hierarchies.json")
    args = parser.parse_args()
    # dh = DeformationHandler(gpu_id=args.gpu_id,
    #                         hierarchy_level=args.hierarchy_level,
    #                         is_extraction=True,
    #                         param_json_file=args.param_json_file)
    # dh.get_distortion_and_miou_stats()

    dh = DeformationHandler(gpu_id=args.gpu_id,
                            hierarchy_level=args.hierarchy_level,
                            is_extraction=args.operation == "extract",
                            param_json_file=args.param_json_file)
    # dh = DeformationHandler(gpu_id=args.gpu_id,
    #                         hierarchy_level=args.hierarchy_level,
    #                         is_extraction=args.operation == "extract",
    #                         param_json_file="/home/yakir/PycharmProjects/secure_inference/research/block_relu/distortion_handler_configs/resnet_COCO_164K_8_hierarchies.json")
    # dh = DeformationHandler(gpu_id=args.gpu_id,
    #                         hierarchy_level=None,
    #                         is_extraction=args.operation == "extract",
    #                         param_json_file=args.param_json_file)

    # dh.pre_layer_distortion_to_additive_estimated_distortion()
    # assert False
    # dh.pre_layer_distortion_to_additive_estimated_distortion()
    # assert False
    # dh.collect_deformation_by_layers()
    # dh.get_block_spec()
    # dh.get_block_spec_v2()
    # assert False
    if args.operation == "extract":
        indices = [int(x) for x in args.batch_index.split(",")]
        for batch_index in indices:
            getattr(dh, f"{args.operation}_deformation_by_{args.hierarchy_type}")(batch_index)
    elif args.operation == "collect":
        getattr(dh, f"{args.operation}_deformation_by_{args.hierarchy_type}")()
    elif args.operation == "get_reduction_spec":
        import time
        # time.sleep(160*60)
        dh.get_block_spec()
    elif args.operation == "get_distortion_and_miou_stats":
        assert False
        dh = DeformationHandler(gpu_id=args.gpu_id,
                                hierarchy_level=args.hierarchy_level,
                                is_extraction=True,
                                param_json_file=args.param_json_file)
        dh.get_distortion_and_miou_stats()