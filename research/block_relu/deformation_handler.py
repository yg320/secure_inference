import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import glob

from research.block_relu.consts import LAYER_NAME_TO_BLOCK_SIZES, LAYER_NAMES, LAYER_NAME_TO_CHANNELS, \
    LAYER_NAME_TO_BLOCK_NAME, BLOCK_NAMES, IN_LAYER_PROXY_SPEC, TARGET_REDUCTIONS, NUM_OF_IN_LAYER_GROUPS, \
    LAYER_HIERARCHY_SPEC, TARGET_DEFORMATIONS_SPEC
from research.block_relu.utils import get_model, get_data, run_model_block, get_layer, set_bReLU_layers, set_layers, \
    center_crop


class DeformationHandler:
    def __init__(self, batch_index, batch_size, gpu_id, hierarchy_level):
        self.batch_index = batch_index
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.hierarchy_level = hierarchy_level
        self.device = f"cuda:{gpu_id}"

        self.deformation_base_path = "/home/yakir/Data2/assets_v2/deformations"
        self.config = "/home/yakir/PycharmProjects/mmsegmentation/configs/secure_semantic_segmentation/baseline_40k_finetune_tmp.py"
        self.checkpoint = "/home/yakir/PycharmProjects/mmsegmentation/work_dirs/baseline_40k/latest.pth"

        self.num_blocks = 18
        self.im_size = 512

        self.model = get_model(
            config=self.config,
            gpu_id=self.gpu_id,
            checkpoint_path=self.checkpoint
        )
        self.dataset = get_data(self.config)

    def _get_files(self, output_path, layer_name):

        noise_f_name = os.path.join(output_path,
                                    f"noise_{layer_name}_batch_{self.batch_index}_{self.batch_size}.npy")
        signal_f_name = os.path.join(output_path,
                                     f"signal_{layer_name}_batch_{self.batch_index}_{self.batch_size}.npy")
        loss_deform_f_name = os.path.join(output_path,
                                          f"loss_deform_{layer_name}_batch_{self.batch_index}_{self.batch_size}.npy")

        return noise_f_name, signal_f_name, loss_deform_f_name

    def get_activations(self):
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

    def extract_deformation_by_block(self):
        output_path = os.path.join(self.deformation_base_path, "block")
        with torch.no_grad():
            resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations()

            for layer_index, layer_name in enumerate(LAYER_NAMES):
                torch.cuda.empty_cache()

                layer_block_sizes = LAYER_NAME_TO_BLOCK_SIZES[layer_name]
                assert layer_block_sizes[0][0] == 1 and layer_block_sizes[0][1] == 1


                cur_block_name = LAYER_NAME_TO_BLOCK_NAME[layer_name]
                next_block_name = IN_LAYER_PROXY_SPEC[layer_name]

                cur_block_index = np.argwhere(np.array(BLOCK_NAMES) == cur_block_name)[0, 0]
                next_block_index = np.argwhere(np.array(BLOCK_NAMES) == next_block_name)[0, 0]

                cur_tensor = resnet_block_name_to_activation[cur_block_name]
                next_tensor = resnet_block_name_to_activation[next_block_name]

                noise_f_name, signal_f_name, loss_deform_f_name = self._get_files(output_path, layer_name)

                channels = LAYER_NAME_TO_CHANNELS[layer_name]
                noise = np.zeros((len(layer_block_sizes), channels))
                signal = np.zeros((len(layer_block_sizes), channels))
                loss_deform = np.zeros((len(layer_block_sizes), channels))

                for channel in tqdm(range(channels), desc=f"Batch={self.batch_index} Layer={layer_index}"):
                    for block_size_index in range(len(layer_block_sizes)):
                        if block_size_index == 0:
                            continue
                        out = cur_tensor
                        block_size_indices = np.zeros(shape=channels, dtype=np.int32)
                        block_size_indices[channel] = block_size_index

                        orig_layer = get_layer(self.model, layer_name)
                        set_bReLU_layers(self.model, {layer_name: (block_size_indices, layer_block_sizes)})

                        for block_index in range(cur_block_index, next_block_index):
                            out = run_model_block(self.model, out, BLOCK_NAMES[block_index])

                        if next_block_name is None:
                            cur_loss_ce = self.model.decode_head.losses(out, ground_truth)['loss_ce']
                            loss_deform[block_size_index, channel] = (cur_loss_ce - loss_ce) / loss_ce

                        set_layers(self.model, {layer_name: orig_layer})

                        noise[block_size_index, channel] = float(((out - next_tensor) ** 2).mean())
                        signal[block_size_index, channel] = float((next_tensor ** 2).mean())

                np.save(file=noise_f_name, arr=noise)
                np.save(file=signal_f_name, arr=signal)
                if next_block_name is None:
                    np.save(file=loss_deform_f_name, arr=loss_deform)

    def collect_deformation_by_block(self):
        target_deformation = TARGET_DEFORMATIONS_SPEC["block"]

        input_dir = os.path.join(self.deformation_base_path, "block")
        out_dir = os.path.join(self.deformation_base_path, f"channels_{self.hierarchy_level}_in")

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
            group_size = channels // NUM_OF_IN_LAYER_GROUPS
            multi_channel_reduction = deformation_and_channel_to_reduction.reshape(
                (target_deformation.shape[0], NUM_OF_IN_LAYER_GROUPS, group_size)).mean(axis=-1)

            reduction_to_block_sizes = np.zeros((TARGET_REDUCTIONS.shape[0], channels))
            for target_reduction_index, target_reduction in enumerate(TARGET_REDUCTIONS):
                indices = np.repeat(np.argmin(np.abs(target_reduction - multi_channel_reduction), axis=0), group_size)
                reduction_to_block_sizes[target_reduction_index] = deformation_and_channel_to_block_size[
                    indices, np.arange(0, channels)]

            redundancy_arr = [0]
            for i in range(1, TARGET_REDUCTIONS.shape[0]):
                if np.all(reduction_to_block_sizes[i] == reduction_to_block_sizes[i - 1]):
                    redundancy_arr.append(redundancy_arr[-1])
                else:
                    redundancy_arr.append(i)

            redundancy_arr = np.array(redundancy_arr)
            np.save(file=os.path.join(out_dir, f"redundancy_arr_{layer_name}.npy"), arr=redundancy_arr)
            np.save(file=os.path.join(out_dir, f"reduction_to_block_sizes_{layer_name}.npy"),
                    arr=reduction_to_block_sizes)

    def extract_deformation_by_channels_group(self):
        num_of_groups = NUM_OF_IN_LAYER_GROUPS[self.hierarchy_level]

        input_path = os.path.join(self.deformation_base_path, f"channels_{self.hierarchy_level}_in")
        output_path = os.path.join(self.deformation_base_path, f"channels_{self.hierarchy_level}_out")

        with torch.no_grad():
            resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations()

            for layer_index, layer_name in enumerate(LAYER_NAMES):
                torch.cuda.empty_cache()

                redundancy_arr_f_name = os.path.join(input_path, f"redundancy_arr_{layer_name}.npy")
                if os.path.exists(redundancy_arr_f_name):
                    redundancy_arr = np.load(redundancy_arr_f_name)
                else:
                    redundancy_arr = None
                reduction_to_block_sizes = np.load(
                    os.path.join(input_path, f"reduction_to_block_sizes_{layer_name}.npy"))

                cur_block_name = LAYER_NAME_TO_BLOCK_NAME[layer_name]
                next_block_name = IN_LAYER_PROXY_SPEC[layer_name]

                cur_block_index = np.argwhere(np.array(BLOCK_NAMES) == cur_block_name)[0, 0]
                next_block_index = np.argwhere(np.array(BLOCK_NAMES) == next_block_name)[0, 0]

                layer_block_sizes = LAYER_NAME_TO_BLOCK_SIZES[layer_name]
                assert layer_block_sizes[0][0] == 1 and layer_block_sizes[0][1] == 1
                cur_tensor = resnet_block_name_to_activation[cur_block_name]
                next_tensor = resnet_block_name_to_activation[next_block_name]

                noise_f_name = os.path.join(output_path,
                                            f"noise_{layer_name}_batch_{self.batch_index}_{self.batch_size}.npy")
                signal_f_name = os.path.join(output_path,
                                             f"signal_{layer_name}_batch_{self.batch_index}_{self.batch_size}.npy")
                loss_deform_f_name = os.path.join(output_path,
                                                  f"loss_deform_{layer_name}_batch_{self.batch_index}_{self.batch_size}.npy")

                channels = LAYER_NAME_TO_CHANNELS[layer_name]
                noise = np.zeros((TARGET_REDUCTIONS.shape[0], num_of_groups))
                signal = np.zeros((TARGET_REDUCTIONS.shape[0], num_of_groups))
                loss_deform = np.zeros((TARGET_REDUCTIONS.shape[0], num_of_groups))

                group_size = channels // num_of_groups
                assert group_size * num_of_groups == channels

                for channel_group in tqdm(range(num_of_groups), desc=f"Batch={self.batch_index} Layer={layer_index}"):
                    for reduction_index, reduction in enumerate(TARGET_REDUCTIONS):
                        if (redundancy_arr is not None) and (redundancy_arr[reduction_index] != reduction_index):
                            assert redundancy_arr[reduction_index] < reduction_index
                            noise[reduction_index, channel_group] = noise[
                                redundancy_arr[reduction_index], channel_group]
                            signal[reduction_index, channel_group] = signal[
                                redundancy_arr[reduction_index], channel_group]
                        else:
                            out = cur_tensor
                            block_size_indices = np.zeros(shape=channels, dtype=np.int32)

                            ind_start = channel_group * group_size
                            ind_end = ind_start + group_size
                            block_size_indices[ind_start:ind_end] = reduction_to_block_sizes[reduction_index,
                                                                    ind_start:ind_end]

                            orig_layer = get_layer(self.model, layer_name)
                            set_bReLU_layers(self.model, {layer_name: (block_size_indices, layer_block_sizes)})

                            for block_index in range(cur_block_index, next_block_index):
                                out = run_model_block(self.model, out, BLOCK_NAMES[block_index])

                            if next_block_name is None:
                                cur_loss_ce = self.model.decode_head.losses(out, ground_truth)['loss_ce']
                                loss_deform[reduction_index, channel_group] = (cur_loss_ce - loss_ce) / loss_ce

                            set_layers(self.model, {layer_name: orig_layer})

                            noise[reduction_index, channel_group] = float(((out - next_tensor) ** 2).mean())
                            signal[reduction_index, channel_group] = float((next_tensor ** 2).mean())

                np.save(file=noise_f_name, arr=noise)
                np.save(file=signal_f_name, arr=signal)
                if next_block_name is None:
                    np.save(file=loss_deform_f_name, arr=loss_deform)

    def collect_deformation_by_channel_group(self):
        pass

    def extract_deformation_by_layers_group(self):
        input_path = os.path.join(self.deformation_base_path, f"layers_{self.hierarchy_level}_in")
        output_path = os.path.join(self.deformation_base_path, f"layers_{self.hierarchy_level}_out")

        with torch.no_grad():
            resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations()

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
    parser.add_argument('--batch_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--hierarchy_level', type=int, default=0)
    parser.add_argument('--hierarchy_type', type=str, default="blocks")
    args = parser.parse_args()

    DeformationHandler(batch_index=args.batch_index,
                      batch_size=args.batch_size,
                      gpu_id=args.gpu_id,
                      hierarchy_level=args.hierarchy_level,
                      hierarchy_type=args.hierarchy_typez)
