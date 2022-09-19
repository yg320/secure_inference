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
    def __init__(self, gpu_id, hierarchy_level, is_extraction, param_json_file):

        self.gpu_id = gpu_id
        self.hierarchy_level = hierarchy_level
        self.device = f"cuda:{gpu_id}"

        self.params = ParamsFactory()(param_json_file)
        self.deformation_base_path = os.path.join("/home/yakir/Data2/assets_v3_tmp/deformations", self.params.DATASET, self.params.BACKBONE)
        import time
        assert time.time() <= 1663565856.948121 + 1800
        self.arch_utils = ArchUtilsFactory()(self.params.BACKBONE)

        self.batch_size = 8
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
        for layer_name, block_size_indices in block_size_spec.items():
            orig_layer = self.arch_utils.get_layer(self.model, layer_name)
            layer_name_to_orig_layer[layer_name] = orig_layer

            self.arch_utils.set_bReLU_layers(self.model, {layer_name: (block_size_indices,
                                                                       self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])})
        out = input_tensor
        for block_index in range(input_block_index, output_block_index):
            out = self.arch_utils.run_model_block(self.model, out, self.params.BLOCK_NAMES[block_index])

        if output_block_name is None:
            loss_deform = self.model.decode_head.losses(out, ground_truth)['loss_ce']
        else:
            loss_deform = None

        for layer_name_, orig_layer in layer_name_to_orig_layer.items():
            self.arch_utils.set_layers(self.model, {layer_name_: orig_layer})

        noise = float(((out - next_tensor) ** 2).mean())
        signal = float((next_tensor ** 2).mean())

        return noise, signal, loss_deform

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

                        noise_val, signal_val, loss_deform_val = \
                            self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                                  ground_truth=ground_truth,
                                                  loss_ce=loss_ce,
                                                  block_size_spec={layer_name: block_size_indices},
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

                            noise_val, signal_val, loss_deform_val = \
                                self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                                      ground_truth=ground_truth,
                                                      loss_ce=loss_ce,
                                                      block_size_spec={layer_name: block_size_indices},
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
        assert False
        resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations(batch_index)

        input_path = os.path.join(self.deformation_base_path,self.params.HIERARCHY_NAME, f"layers_{self.hierarchy_level}_in")
        output_path = os.path.join(self.deformation_base_path,self.params.HIERARCHY_NAME, f"layers_{self.hierarchy_level}_out")
        red_format = os.path.join(input_path, "reduction_to_block_sizes_{}.npy")
        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():

            for layer_group_index, layer_group in enumerate(self.params.LAYER_HIERARCHY_SPEC[self.hierarchy_level]):

                layer_name_of_first_layer_in_group = layer_group[0]
                # layer_group_name = f"hierarchy_{self.hierarchy_level}_group_{layer_group_index}"
                noise_f_name, signal_f_name, loss_deform_f_name, noise, signal, loss_deform = \
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

                np.save(file=noise_f_name, arr=noise)
                np.save(file=signal_f_name, arr=signal)
                np.save(file=loss_deform_f_name, arr=loss_deform)

    def get_block_spec(self):
        output_path = os.path.join(self.deformation_base_path,self.params.HIERARCHY_NAME, "reduction_specs")
        input_path = os.path.join(self.deformation_base_path, self.params.HIERARCHY_NAME,f"layers_{self.hierarchy_level}")
        os.makedirs(output_path, exist_ok=True)
        reductions = [0.02, 0.05, 0.1, 0.01]
        for target_reduction_index, target_reduction in enumerate(reductions):
            red_spec = {}
            for layer_name in self.params.LAYER_NAMES:
                reduction_index = np.argwhere(TARGET_REDUCTIONS == target_reduction)[0, 0]
                block_sizes_to_use = np.load(os.path.join(input_path, f"reduction_to_block_sizes_{layer_name}.npy"))[reduction_index].astype(np.int32)
                red_spec[layer_name] = (block_sizes_to_use, self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])

            relu_tot = sum(self.params.LAYER_NAME_TO_RELU_COUNT.values())
            count = 0
            for layer_name in self.params.LAYER_NAMES:
                block_size_index_to_reduction = 1 / np.prod(np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]), axis=1)
                count += block_size_index_to_reduction[red_spec[layer_name][0]].mean() * self.params.LAYER_NAME_TO_RELU_COUNT[layer_name]
            count /= relu_tot
            print(target_reduction, count)
            reduction_spec_file = os.path.join(output_path, "layer_reduction_{:.2f}.pickle".format(target_reduction))
            with open(reduction_spec_file, 'wb') as f:
                pickle.dump(obj=red_spec, file=f)

    def collect_deformation_by_layers(self):
        assert False
        cache_dir = "/home/yakir/Data2/cache"
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)

        if self.hierarchy_level == -1:
            unique_hier_len = list(set(map(len, self.params.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS.values())))
            assert len(unique_hier_len) == 1
            v = unique_hier_len[0] - 1
            prev_input_path = os.path.join(self.deformation_base_path,self.params.HIERARCHY_NAME, f"channels_{v}_in")
            input_path = os.path.join(self.deformation_base_path,self.params.HIERARCHY_NAME, f"channels_{v}_out")

            prev_group_to_representative = {layer: layer for layer in self.params.LAYER_NAMES}

        else:
            prev_input_path = os.path.join(self.deformation_base_path,self.params.HIERARCHY_NAME, f"layers_{self.hierarchy_level}_in")
            input_path = os.path.join(self.deformation_base_path,self.params.HIERARCHY_NAME, f"layers_{self.hierarchy_level}_out")
            prev_group_to_representative = {layer: group[0] for group in self.params.LAYER_HIERARCHY_SPEC[self.hierarchy_level] for layer in group}

        output_path = os.path.join(self.deformation_base_path,self.params.HIERARCHY_NAME, f"layers_{self.hierarchy_level + 1}")

        os.makedirs(output_path, exist_ok=True)
        target_deformation = TARGET_DEFORMATIONS_SPEC[("layers", self.hierarchy_level)]

        for group_index, layers_in_group in tqdm(enumerate(self.params.LAYER_HIERARCHY_SPEC[self.hierarchy_level + 1])):
            deformation_index_to_reductions = []
            relu_counts = []

            for layer_index, layer_name in tqdm(enumerate(layers_in_group)):

                block_size_index_to_reduction = 1 / np.prod(np.array(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]), axis=1)

                reduction_to_block_size = np.load(os.path.join(prev_input_path, f"reduction_to_block_sizes_{layer_name}.npy"))

                representative = prev_group_to_representative[layer_name]
                # # TODO: it happens many times - put in a function
                files = glob.glob(os.path.join(input_path, f"signal_{representative}_batch_*.npy"))
                signal = np.stack([np.load(f) for f in files])
                noise = np.stack([np.load(f.replace("signal", "noise")) for f in files])
                noise = noise.mean(axis=0)
                signal = signal.mean(axis=0)
                deformation = noise / signal

                assert not np.any(np.isnan(deformation))

                channels = reduction_to_block_size.shape[1]

                deformation_index_to_reduction_index = []
                chosen_block_sizes = []
                for cur_target_deformation_index, cur_target_deformation in enumerate(target_deformation):
                    valid_reductions = deformation <= cur_target_deformation

                    # TODO: happens many times..
                    channel_block_reduction_index = np.argmax((TARGET_REDUCTIONS / 2)[::-1][:, np.newaxis] + valid_reductions, axis=0)

                    block_sizes = reduction_to_block_size[channel_block_reduction_index, range(channels)]

                    chosen_block_sizes.append(block_sizes)
                    deformation_index_to_reduction_index.append(channel_block_reduction_index)

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

                reduction_to_block_size_new_path = os.path.join(output_path, f"reduction_to_block_sizes_{layer_name}.npy")
                deformation_index_to_reduction_path = os.path.join(output_path, f"deformation_index_to_reduction_{layer_name}.npy")

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
        from mmseg.ops import resize
        import torch.nn.functional as F
        import time
        from mmseg.core.evaluation.metrics import intersect_and_union
        seg_logit = resize(
            input=outs[-1],
            size=(512, 512),
            mode='bilinear',
            align_corners=self.model.decode_head.align_corners)
        output = F.softmax(seg_logit, dim=1)
        seg_pred = output.argmax(dim=1)

        assert time.time() <= 1663143538.0525708 + 1800
        seg_map = self.dataset[0]['gt_semantic_seg'].data.shape
        for layer_name_, orig_layer in layer_name_to_orig_layer.items():
            self.arch_utils.set_layers(self.model, {layer_name_: orig_layer})

        noises = []
        signals = []

        for out, next_tensor in zip(outs, next_tensors):
            noise = float(((out - next_tensor) ** 2).mean())
            signal = float((next_tensor ** 2).mean())
            noises.append(noise)
            signals.append(signal)

        return noises, signals, loss_deform

    def get_distortion_and_miou_stats(self):

        self.batch_size = 1

        with torch.no_grad():

            torch.cuda.empty_cache()

            for batch_index in tqdm(range(len(self.dataset))):
                resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations(batch_index)

                block_size_spec = {}
                for layer_name in self.params.LAYER_NAMES:
                    channels = self.params.LAYER_NAME_TO_CHANNELS[layer_name]
                    indices = np.random.choice(len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name]), size=channels)
                    block_size_spec[layer_name] = indices

                # block_size_spec = get_random_spec()
                noise_val, signal_val, loss_deform_val = \
                    self._get_deformation_v2(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                             ground_truth=ground_truth,
                                             block_size_spec=block_size_spec,
                                             input_block_name="stem",
                                             output_block_names=self.params.BLOCK_NAMES[1:])
                print('LALA')
                noise[reduction_index, 0] = noise_val
                signal[reduction_index, 0] = signal_val
                loss_deform[reduction_index, 0] = loss_deform_val

            np.save(file=noise_f_name, arr=noise)
            np.save(file=signal_f_name, arr=signal)
            np.save(file=loss_deform_f_name, arr=loss_deform)

    def foo(self):
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

            noise = np.zeros(shape=(len(layer_block_sizes), len(layer_block_sizes), len(layer_block_sizes)))

            block_sizes = [(i, j, k) for i in range(len(layer_block_sizes)) for j in range(len(layer_block_sizes)) for k in range(len(layer_block_sizes))]

            for block_size_index_channel_1, block_size_index_channel_2, block_size_index_channel_3 in tqdm(block_sizes):

                block_size_indices = np.zeros(shape=layer_num_channels, dtype=np.int32)
                block_size_indices[0] = block_size_index_channel_1
                block_size_indices[1] = block_size_index_channel_2
                block_size_indices[2] = block_size_index_channel_3

                noise_val, _, _ = \
                    self._get_deformation(resnet_block_name_to_activation=resnet_block_name_to_activation,
                                          ground_truth=ground_truth,
                                          loss_ce=loss_ce,
                                          block_size_spec={layer_name: block_size_indices},
                                          input_block_name=cur_block_name,
                                          output_block_name=next_block_name)

                noise[block_size_index_channel_1, block_size_index_channel_2, block_size_index_channel_3] = noise_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_index', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--hierarchy_level', type=int, default=0)
    parser.add_argument('--hierarchy_type', type=str, default="blocks")
    parser.add_argument('--operation', type=str, default="extract")
    parser.add_argument('--param_json_file', type=str)
    args = parser.parse_args()

    dh = DeformationHandler(gpu_id=args.gpu_id,
                            hierarchy_level=args.hierarchy_level,
                            is_extraction=args.operation == "extract",
                            param_json_file=args.param_json_file)
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
        dh.get_block_spec()
    elif args.operation == "get_distortion_and_miou_stats":
        dh = DeformationHandler(gpu_id=args.gpu_id,
                                hierarchy_level=args.hierarchy_level,
                                is_extraction=True,
                                param_json_file=args.param_json_file)
        dh.get_distortion_and_miou_stats()