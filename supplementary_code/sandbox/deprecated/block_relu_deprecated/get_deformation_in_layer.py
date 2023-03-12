import argparse
import os
from tqdm import tqdm
import torch
import numpy as np

from research.block_relu.consts import LAYER_NAME_TO_BLOCK_SIZES, LAYER_NAMES, LAYER_NAME_TO_CHANNELS, \
    LAYER_NAME_TO_BLOCK_NAME, BLOCK_NAMES, TARGET_REDUCTIONS, NUM_OF_IN_LAYER_GROUPS, HIERARCHY_LAYER_PROXY_SPEC
from research.block_relu.utils import get_model, get_data, run_model_block, get_layer, set_bReLU_layers, set_layers,\
    center_crop


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--hierarchy_index', type=int, default=0)
    args = parser.parse_args()

    hierarchy_index = args.hierarchy_index
    input_dir = f"/home/yakir/Data2/assets_v2/deformation_grouping_in_layer_step_{hierarchy_index}"
    output_path = f"/home/yakir/Data2/assets_v2/in_layer_deformation_step_{hierarchy_index}"

    config = "/home/yakir/PycharmProjects/mmsegmentation/configs/secure_semantic_segmentation/baseline_40k_finetune_tmp.py"
    checkpoint = "/home/yakir/PycharmProjects/mmsegmentation/work_dirs/baseline_40k/latest.pth"
    os.makedirs(output_path, exist_ok=True)

    batch_index = args.batch_index
    batch_size = args.batch_size
    gpu_id = args.gpu_id
    num_blocks = 18
    im_size = 512
    device = f"cuda:{gpu_id}"
    model = get_model(
        config=config,
        gpu_id=gpu_id,
        checkpoint_path=checkpoint
    )

    dataset = get_data(config)
    num_of_groups = NUM_OF_IN_LAYER_GROUPS[hierarchy_index]
    with torch.no_grad():
        batch_indices = range(batch_index * batch_size, (batch_index + 1) * batch_size)
        batch = torch.stack([center_crop(dataset[sample_id]['img'].data, im_size) for sample_id in batch_indices]).to(device)
        batch_gt = torch.stack([center_crop(dataset[sample_id]['gt_semantic_seg'].data, im_size) for sample_id in batch_indices]).to(device)
        activations = [batch]
        for block_index in range(num_blocks):
            activations.append(run_model_block(model, activations[block_index], BLOCK_NAMES[block_index]))
        loss_ce = model.decode_head.losses(activations[-1], batch_gt)['loss_ce']
        resnet_block_name_to_activation = dict(zip(BLOCK_NAMES, activations))

        for layer_index, layer_name in enumerate(LAYER_NAMES):
            redundancy_arr_f_name = os.path.join(input_dir, f"redundancy_arr_{layer_name}.npy")
            if os.path.exists(redundancy_arr_f_name):
                redundancy_arr = np.load(redundancy_arr_f_name)
            else:
                redundancy_arr = None
            reduction_to_block_sizes = np.load(os.path.join(input_dir, f"reduction_to_block_sizes_{layer_name}.npy"))
            torch.cuda.empty_cache()
            cur_block_name = LAYER_NAME_TO_BLOCK_NAME[layer_name]
            next_block_name = HIERARCHY_LAYER_PROXY_SPEC[hierarchy_index][layer_name]

            cur_block_index = np.argwhere(np.array(BLOCK_NAMES) == cur_block_name)[0, 0]
            next_block_index = np.argwhere(np.array(BLOCK_NAMES) == next_block_name)[0, 0]

            layer_block_sizes = LAYER_NAME_TO_BLOCK_SIZES[layer_name]
            assert layer_block_sizes[0][0] == 1 and layer_block_sizes[0][1] == 1
            cur_tensor = resnet_block_name_to_activation[cur_block_name]
            next_tensor = resnet_block_name_to_activation[next_block_name]
            noise_f_name = os.path.join(output_path, f"noise_{layer_name}_batch_{batch_index}_{batch_size}.npy")
            signal_f_name = os.path.join(output_path, f"signal_{layer_name}_batch_{batch_index}_{batch_size}.npy")
            loss_deform_f_name = os.path.join(output_path, f"loss_deform_{layer_name}_batch_{batch_index}_{batch_size}.npy")
            if os.path.exists(noise_f_name):
                continue
            channels = LAYER_NAME_TO_CHANNELS[layer_name]
            noise = np.zeros((TARGET_REDUCTIONS.shape[0], num_of_groups))
            signal = np.zeros((TARGET_REDUCTIONS.shape[0], num_of_groups))
            loss_deform = np.zeros((TARGET_REDUCTIONS.shape[0], num_of_groups))

            group_size = channels // num_of_groups
            assert group_size * num_of_groups == channels

            for channel_group in tqdm(range(num_of_groups), desc=f"Batch={batch_index} Layer={layer_index}"):
                for reduction_index, reduction in enumerate(TARGET_REDUCTIONS):
                    if (redundancy_arr is not None) and (redundancy_arr[reduction_index] != reduction_index):
                        assert redundancy_arr[reduction_index] < reduction_index
                        noise[reduction_index, channel_group] = noise[redundancy_arr[reduction_index], channel_group]
                        signal[reduction_index, channel_group] = signal[redundancy_arr[reduction_index], channel_group]
                    else:
                        out = cur_tensor
                        block_size_indices = np.zeros(shape=channels, dtype=np.int32)

                        ind_start = channel_group * group_size
                        ind_end = ind_start + group_size
                        block_size_indices[ind_start:ind_end] = reduction_to_block_sizes[reduction_index, ind_start:ind_end]

                        orig_layer = get_layer(model, layer_name)
                        set_bReLU_layers(model, {layer_name: (block_size_indices, layer_block_sizes)})

                        for block_index in range(cur_block_index, next_block_index):
                            out = run_model_block(model, out, BLOCK_NAMES[block_index])

                        if next_block_name is None:
                            cur_loss_ce = model.decode_head.losses(out, batch_gt)['loss_ce']
                            loss_deform[reduction_index, channel_group] = cur_loss_ce - loss_ce

                        set_layers(model, {layer_name: orig_layer})

                        noise[reduction_index, channel_group] = float(((out - next_tensor) ** 2).mean())
                        signal[reduction_index, channel_group] = float((next_tensor ** 2).mean())

            np.save(file=noise_f_name, arr=noise)
            np.save(file=signal_f_name, arr=signal)
            if next_block_name is None:
                np.save(file=loss_deform_f_name, arr=loss_deform)


if __name__ == '__main__':
    main()
