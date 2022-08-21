import numpy as np
import os
from tqdm import tqdm

from research.block_relu.consts import NUM_OF_IN_LAYER_GROUPS, TARGET_DEFORMATIONS, TARGET_REDUCTIONS, LAYER_NAMES


input_dir = "/home/yakir/Data2/assets_v2/collected_by_channel_by_resize_factor_deformation"
output_dir = "/home/yakir/Data2/assets_v2/deformation_grouping_in_layer_step_0"

os.makedirs(output_dir, exist_ok=True)
for layer_name in tqdm(LAYER_NAMES):

    reduction = np.load(os.path.join(input_dir, f"deformation_and_channel_to_reduction_{layer_name}.npy"))
    block_size = np.load(os.path.join(input_dir, f"deformation_and_channel_to_block_size_{layer_name}.npy"))

    channels = reduction.shape[1]
    group_size = channels // NUM_OF_IN_LAYER_GROUPS
    multi_channel_reduction = reduction.reshape((TARGET_DEFORMATIONS.shape[0], NUM_OF_IN_LAYER_GROUPS, group_size)).mean(axis=-1)

    reduction_to_block_sizes = np.zeros((TARGET_REDUCTIONS.shape[0], channels))
    for target_reduction_index, target_reduction in enumerate(TARGET_REDUCTIONS):

        indices = np.repeat(np.argmin(np.abs(target_reduction - multi_channel_reduction), axis=0), group_size)
        reduction_to_block_sizes[target_reduction_index] = block_size[indices, np.arange(0, channels)]

    redundancy_arr = [0]
    for i in range(1, TARGET_REDUCTIONS.shape[0]):
        if np.all(reduction_to_block_sizes[i] == reduction_to_block_sizes[i-1]):
            redundancy_arr.append(redundancy_arr[-1])
        else:
            redundancy_arr.append(i)

    redundancy_arr = np.array(redundancy_arr)
    np.save(file=os.path.join(output_dir, f"redundancy_arr_{layer_name}.npy"), arr=redundancy_arr)
    np.save(file=os.path.join(output_dir, f"reduction_to_block_sizes_{layer_name}.npy"), arr=reduction_to_block_sizes)