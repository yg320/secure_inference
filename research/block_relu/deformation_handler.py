import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import glob

from research.block_relu.consts import LAYER_NAME_TO_BLOCK_SIZES, LAYER_NAMES, LAYER_NAME_TO_CHANNELS, \
    LAYER_NAME_TO_BLOCK_NAME, BLOCK_NAMES, IN_LAYER_PROXY_SPEC, TARGET_REDUCTIONS, HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS, \
    LAYER_HIERARCHY_SPEC, TARGET_DEFORMATIONS_SPEC
from research.block_relu.utils import get_model, get_data, run_model_block, get_layer, set_bReLU_layers, set_layers, \
    center_crop

# TODO: add typing
class DeformationHandler:
    def __init__(self, batch_index, gpu_id, hierarchy_level, is_extraction):
        self.batch_index = batch_index
        self.gpu_id = gpu_id
        self.hierarchy_level = hierarchy_level
        self.device = f"cuda:{gpu_id}"

        self.deformation_base_path = "/home/yakir/Data2/assets_v2/deformations"
        self.config = "/home/yakir/PycharmProjects/mmsegmentation/configs/secure_semantic_segmentation/baseline_40k_finetune_tmp.py"
        self.checkpoint = "/home/yakir/PycharmProjects/mmsegmentation/work_dirs/baseline_40k/latest.pth"

        self.batch_size = 8
        self.num_blocks = 18
        self.im_size = 512

        if is_extraction:
            self.model = get_model(
                config=self.config,
                gpu_id=self.gpu_id,
                checkpoint_path=self.checkpoint
            )
            self.dataset = get_data(self.config)

            self.resnet_block_name_to_activation, self.ground_truth, self.loss_ce = self.get_activations()

    def _get_deformation(self, block_size_spec, input_block_name, output_block_name):

        input_block_index = np.argwhere(np.array(BLOCK_NAMES) == input_block_name)[0, 0]
        output_block_index = np.argwhere(np.array(BLOCK_NAMES) == output_block_name)[0, 0]

        input_tensor = self.resnet_block_name_to_activation[input_block_name]
        next_tensor = self.resnet_block_name_to_activation[output_block_name]

        layer_name_to_orig_layer = {}
        for layer_name, block_size_indices in block_size_spec.items():
            orig_layer = get_layer(self.model, layer_name)
            layer_name_to_orig_layer[layer_name] = orig_layer

            set_bReLU_layers(self.model, {layer_name: (block_size_indices,
                                                       LAYER_NAME_TO_BLOCK_SIZES[layer_name])})
        out = input_tensor
        for block_index in range(input_block_index, output_block_index):
            out = run_model_block(self.model, out, BLOCK_NAMES[block_index])

        if output_block_name is None:
            cur_loss_ce = self.model.decode_head.losses(out, self.ground_truth)['loss_ce']
            loss_deform = (cur_loss_ce - self.loss_ce) / self.loss_ce
        else:
            loss_deform = None

        for layer_name_, orig_layer in layer_name_to_orig_layer.items():
            set_layers(self.model, {layer_name_: orig_layer})

        noise = float(((out - next_tensor) ** 2).mean())
        signal = float((next_tensor ** 2).mean())

        return noise, signal, loss_deform

    def _get_files_and_matrices(self, output_path, layer_name, h, w):

        noise_f_name = os.path.join(output_path,
                                    f"noise_{layer_name}_batch_{self.batch_index}_{self.batch_size}.npy")
        signal_f_name = os.path.join(output_path,
                                     f"signal_{layer_name}_batch_{self.batch_index}_{self.batch_size}.npy")
        loss_deform_f_name = os.path.join(output_path,
                                          f"loss_deform_{layer_name}_batch_{self.batch_index}_{self.batch_size}.npy")

        return noise_f_name, signal_f_name, loss_deform_f_name, np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))

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

    def get_activations(self):
        with torch.no_grad():
            batch_indices = range(self.batch_index * self.batch_size, (self.batch_index + 1) * self.batch_size)
            batch = torch.stack(
                [center_crop(self.dataset[sample_id]['img'].data, self.im_size) for sample_id in batch_indices]).to(
                self.device)
            ground_truth = torch.stack(
                [center_crop(self.dataset[sample_id]['gt_semantic_seg'].data, self.im_size) for sample_id in
                 batch_indices]).to(self.device)
            activations = [batch]
            for block_index in range(self.num_blocks):
                activations.append(run_model_block(self.model, activations[block_index], BLOCK_NAMES[block_index]))
            loss_ce = self.model.decode_head.losses(activations[-1], ground_truth)['loss_ce']
            resnet_block_name_to_activation = dict(zip(BLOCK_NAMES, activations))

            return resnet_block_name_to_activation, ground_truth, loss_ce

    def extract_deformation_by_blocks(self):
        output_path = os.path.join(self.deformation_base_path, "block")
        os.makedirs(output_path, exist_ok=True)
        with torch.no_grad():

            for layer_index, layer_name in enumerate(LAYER_NAMES):
                torch.cuda.empty_cache()

                layer_block_sizes = LAYER_NAME_TO_BLOCK_SIZES[layer_name]
                layer_num_channels = LAYER_NAME_TO_CHANNELS[layer_name]

                assert layer_block_sizes[0][0] == 1 and layer_block_sizes[0][1] == 1

                cur_block_name = LAYER_NAME_TO_BLOCK_NAME[layer_name]
                next_block_name = IN_LAYER_PROXY_SPEC[layer_name]

                noise_f_name, signal_f_name, loss_deform_f_name, noise, signal, loss_deform = \
                    self._get_files_and_matrices(output_path, layer_name, len(layer_block_sizes), layer_num_channels)

                for channel in tqdm(range(layer_num_channels), desc=f"Batch={self.batch_index} Layer={layer_index}"):
                    for block_size_index in range(len(layer_block_sizes)):
                        if block_size_index == 0:
                            continue

                        block_size_indices = np.zeros(shape=layer_num_channels, dtype=np.int32)
                        block_size_indices[channel] = block_size_index

                        noise_val, signal_val, loss_deform_val = \
                            self._get_deformation(block_size_spec={layer_name: block_size_indices},
                                                  input_block_name=cur_block_name,
                                                  output_block_name=next_block_name)

                        noise[block_size_index, channel] = noise_val
                        signal[block_size_index, channel] = signal_val
                        loss_deform[block_size_index, channel] = loss_deform_val

                np.save(file=noise_f_name, arr=noise)
                np.save(file=signal_f_name, arr=signal)
                np.save(file=loss_deform_f_name, arr=loss_deform)

    def collect_deformation_by_blocks(self):
        target_deformation = TARGET_DEFORMATIONS_SPEC["block"]
        num_groups = HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[0]
        input_dir = os.path.join(self.deformation_base_path, "block")
        out_dir = os.path.join(self.deformation_base_path, f"channels_0_in")

        os.makedirs(out_dir)

        for layer_index, layer_name in tqdm(enumerate(LAYER_NAMES)):

            files = glob.glob(os.path.join(input_dir, f"signal_{layer_name}_batch_*.npy"))

            signal = np.stack([np.load(f) for f in files])
            noise = np.stack([np.load(f.replace("signal", "noise")) for f in files])
            noise = noise.mean(axis=0)
            signal = signal.mean(axis=0)
            deformation = noise / signal
            deformation[0] = 0
            assert not np.any(np.isnan(deformation))
            assert deformation.max() <= target_deformation[-1]
            block_sizes = LAYER_NAME_TO_BLOCK_SIZES[layer_name]
            activation_reduction = np.array([1 / x[0] / x[1] for x in block_sizes])
            deformation_and_channel_to_block_size = []
            deformation_and_channel_to_reduction = []

            broadcast_block_size = np.repeat(np.array([x[0] * x[1] for x in block_sizes])[:, np.newaxis],
                                             deformation.shape[1], axis=1)
            for cur_target_deformation_index, cur_target_deformation in enumerate(target_deformation):
                valid_block_sizes = deformation <= cur_target_deformation
                block_sizes_with_zero_on_non_valid_blocks = broadcast_block_size * valid_block_sizes

                cur_block_sizes = np.argmax(block_sizes_with_zero_on_non_valid_blocks, axis=0)
                cur_reduction = activation_reduction[cur_block_sizes]

                deformation_and_channel_to_block_size.append(cur_block_sizes)
                deformation_and_channel_to_reduction.append(cur_reduction)

            deformation_and_channel_to_reduction = np.array(deformation_and_channel_to_reduction)
            deformation_and_channel_to_block_size = np.array(deformation_and_channel_to_block_size)

            channels = deformation_and_channel_to_reduction.shape[1]
            group_size = channels // num_groups
            multi_channel_reduction = deformation_and_channel_to_reduction.reshape(
                (target_deformation.shape[0], num_groups, group_size)).mean(axis=-1)

            # multi_channel_reduction.shape = deformation x num_of_channels_in_a_group
            reduction_to_block_sizes = np.zeros((TARGET_REDUCTIONS.shape[0], channels))
            for target_reduction_index, target_reduction in enumerate(TARGET_REDUCTIONS):
                indices = np.repeat(np.argmin(np.abs(target_reduction - multi_channel_reduction), axis=0), group_size)
                reduction_to_block_sizes[target_reduction_index] = \
                    deformation_and_channel_to_block_size[indices, np.arange(0, channels)]

            redundancy_arr = self._get_redundancy_arr(reduction_to_block_sizes, num_groups, group_size)

            np.save(file=os.path.join(out_dir, f"redundancy_arr_{layer_name}.npy"), arr=redundancy_arr)
            np.save(file=os.path.join(out_dir, f"reduction_to_block_sizes_{layer_name}.npy"),
                    arr=reduction_to_block_sizes)

    def extract_deformation_by_channels(self):

        num_of_groups = HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[self.hierarchy_level]
        input_path = os.path.join(self.deformation_base_path, f"channels_{self.hierarchy_level}_in")
        output_path = os.path.join(self.deformation_base_path, f"channels_{self.hierarchy_level}_out")

        os.makedirs(output_path, exist_ok=True)

        with torch.no_grad():

            for layer_index, layer_name in enumerate(LAYER_NAMES):
                torch.cuda.empty_cache()

                redundancy_arr_f_name = os.path.join(input_path, f"redundancy_arr_{layer_name}.npy")
                redundancy_arr = np.load(redundancy_arr_f_name)

                reduction_to_block_sizes = np.load(os.path.join(input_path, f"reduction_to_block_sizes_{layer_name}.npy"))

                cur_block_name = LAYER_NAME_TO_BLOCK_NAME[layer_name]
                next_block_name = IN_LAYER_PROXY_SPEC[layer_name]

                noise_f_name, signal_f_name, loss_deform_f_name, noise, signal, loss_deform = \
                    self._get_files_and_matrices(output_path, layer_name, TARGET_REDUCTIONS.shape[0], num_of_groups)

                channels = LAYER_NAME_TO_CHANNELS[layer_name]
                group_size = channels // num_of_groups
                assert group_size * num_of_groups == channels

                for channel_group in tqdm(range(num_of_groups), desc=f"Batch={self.batch_index} Layer={layer_index}"):
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
                                self._get_deformation(block_size_spec={layer_name: block_size_indices},
                                                      input_block_name=cur_block_name,
                                                      output_block_name=next_block_name)

                        noise[reduction_index, channel_group] = noise_val
                        signal[reduction_index, channel_group] = signal_val
                        loss_deform[reduction_index, channel_group] = loss_deform_val

                np.save(file=noise_f_name, arr=noise)
                np.save(file=signal_f_name, arr=signal)
                np.save(file=loss_deform_f_name, arr=loss_deform)

    def collect_deformation_by_channels(self):
        prev_input_path = os.path.join(self.deformation_base_path, f"channels_{self.hierarchy_level}_in")
        input_path = os.path.join(self.deformation_base_path, f"channels_{self.hierarchy_level}_out")
        output_path = os.path.join(self.deformation_base_path, f"channels_{self.hierarchy_level + 1}_in")

        os.makedirs(output_path, exist_ok=True)
        target_deformation = TARGET_DEFORMATIONS_SPEC[("channels", self.hierarchy_level)]

        num_groups_prev = HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[self.hierarchy_level]
        num_groups_curr = HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS[self.hierarchy_level + 1]

        for layer_index, layer_name in tqdm(enumerate(LAYER_NAMES)):

            files = glob.glob(os.path.join(input_path, f"signal_{layer_name}_batch_*.npy"))
            reduction_to_block_size = np.load(os.path.join(prev_input_path, f"reduction_to_block_sizes_{layer_name}.npy"))
            signal = np.stack([np.load(f) for f in files])
            noise = np.stack([np.load(f.replace("signal", "noise")) for f in files])
            noise = noise.mean(axis=0)
            signal = signal.mean(axis=0)
            deformation = noise / signal
            assert deformation.max() <= target_deformation[-1], deformation.max()
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
                block_sizes = reduction_to_block_size[channel_block_reduction_index, range(channels)]
                chosen_block_sizes.append(block_sizes)
                deformation_index_to_reduction_index.append(channel_block_reduction_index)

            chosen_block_sizes = np.array(chosen_block_sizes)
            deformation_index_to_reduction_index = np.array(deformation_index_to_reduction_index)

            deformation_index_to_reduction = TARGET_REDUCTIONS[deformation_index_to_reduction_index.flatten()]
            deformation_index_to_reduction = deformation_index_to_reduction.reshape(deformation_index_to_reduction_index.shape)
            deformation_index_to_reduction = deformation_index_to_reduction.reshape((target_deformation.shape[0], num_groups_curr, group_size_curr)).mean(axis=-1)

            reduction_to_block_size_new = np.zeros((TARGET_REDUCTIONS.shape[0], channels))
            for target_reduction_index, target_reduction in enumerate(TARGET_REDUCTIONS):
                indices = np.repeat(np.argmin(np.abs(target_reduction - deformation_index_to_reduction), axis=0), group_size_curr)
                reduction_to_block_size_new[target_reduction_index] = chosen_block_sizes[indices, np.arange(0, channels)]

            redundancy_arr = self._get_redundancy_arr(reduction_to_block_size_new, num_groups_curr, group_size_curr)

            np.save(file=os.path.join(output_path, f"redundancy_arr_{layer_name}.npy"), arr=redundancy_arr)
            np.save(file=os.path.join(output_path, f"reduction_to_block_sizes_{layer_name}.npy"), arr=reduction_to_block_size_new)

    def extract_deformation_by_layers_group(self):
        assert False, "Use Loss"
        input_path = os.path.join(self.deformation_base_path, f"layers_{self.hierarchy_level}_in")
        output_path = os.path.join(self.deformation_base_path, f"layers_{self.hierarchy_level}_out")

        with torch.no_grad():

            for layer_group in LAYER_HIERARCHY_SPEC[self.hierarchy_level]:
                torch.cuda.empty_cache()
                _, layer_name_of_first_layer_in_group = layer_group[0]
                cur_block_name = LAYER_NAME_TO_BLOCK_NAME[layer_name_of_first_layer_in_group]
                next_block_name = None

                cur_block_index = np.argwhere(np.array(BLOCK_NAMES) == cur_block_name)[0, 0]
                next_block_index = np.argwhere(np.array(BLOCK_NAMES) == next_block_name)[0, 0]

                cur_tensor = resnet_block_name_to_activation[cur_block_name]
                next_tensor = resnet_block_name_to_activation[next_block_name]

                noise_f_name = os.path.join(output_path,
                                            f"noise_{layer_name_of_first_layer_in_group}_batch_{self.batch_index}_{self.batch_size}.npy")
                signal_f_name = os.path.join(output_path,
                                             f"signal_{layer_name_of_first_layer_in_group}_batch_{self.batch_index}_{self.batch_size}.npy")
                loss_deform_f_name = os.path.join(output_path,
                                                  f"loss_deform_{layer_name_of_first_layer_in_group}_batch_{self.batch_index}_{self.batch_size}.npy")

                noise = np.zeros((TARGET_REDUCTIONS.shape[0],))
                signal = np.zeros((TARGET_REDUCTIONS.shape[0],))
                loss_deform = np.zeros((TARGET_REDUCTIONS.shape[0],))

                for reduction_index, reduction in enumerate(TARGET_REDUCTIONS):

                    layer_name_to_orig_layer = {}
                    for layer_index, layer_name in layer_group:
                        out = cur_tensor
                        redundancy_arr_f_name = os.path.join(input_path, f"redundancy_arr_{layer_name}.npy")
                        reduction_to_block_sizes = np.load(redundancy_arr_f_name)
                        layer_block_sizes = LAYER_NAME_TO_BLOCK_SIZES[layer_name]

                        orig_layer = get_layer(self.model, layer_name)
                        layer_name_to_orig_layer[layer_name] = orig_layer
                        set_bReLU_layers(self.model, {layer_name: (reduction_to_block_sizes, layer_block_sizes)})

                    for block_index in range(cur_block_index, next_block_index):
                        out = run_model_block(self.model, out, BLOCK_NAMES[block_index])

                    if next_block_name is None:
                        cur_loss_ce = self.model.decode_head.losses(out, ground_truth)['loss_ce']
                        loss_deform[reduction_index] = (cur_loss_ce - loss_ce) / loss_ce

                    for layer_name_, orig_layer in layer_name_to_orig_layer.items():
                        set_layers(self.model, {layer_name_: orig_layer})

                    noise[reduction_index] = float(((out - next_tensor) ** 2).mean())
                    signal[reduction_index] = float((next_tensor ** 2).mean())

                np.save(file=noise_f_name, arr=noise)
                np.save(file=signal_f_name, arr=signal)
                if next_block_name is None:
                    np.save(file=loss_deform_f_name, arr=loss_deform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_index', type=int, default=None)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--hierarchy_level', type=int, default=0)
    parser.add_argument('--hierarchy_type', type=str, default="blocks")
    parser.add_argument('--operation', type=str, default="extract")
    args = parser.parse_args()

    dh = DeformationHandler(batch_index=args.batch_index,
                            gpu_id=args.gpu_id,
                            hierarchy_level=args.hierarchy_level,
                            is_extraction=args.operation == "extract")

    getattr(dh, f"{args.operation}_deformation_by_{args.hierarchy_type}")()
