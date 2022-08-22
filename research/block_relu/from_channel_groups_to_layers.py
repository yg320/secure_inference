import os
import numpy as np
import glob
from tqdm import tqdm
import time
from research.block_relu.consts import LAYER_NAMES, TARGET_DEFORMATIONS, TARGET_REDUCTIONS, LAYER_NAME_TO_BLOCK_SIZES

input_path = "/home/yakir/Data2/assets_v2/in_layer_deformation_step_0/"
output_path = "/home/yakir/Data2/assets_v2/deformation_grouping_in_layer_step_1/"


os.makedirs(output_path, exist_ok=True)
layer_block_sizes = []
layers_reduction_indices = []
layers_reductions = []
layers_deformation_to_reduction = []

for layer_index, layer_name in tqdm(enumerate(LAYER_NAMES)):

    files = glob.glob(os.path.join(input_path, f"signal_{layer_name}_batch_*.npy"))
    reduction_to_block_size = np.load(os.path.join("/home/yakir/Data2/assets_v2/deformation_grouping_in_layer_step_0/", f"reduction_to_block_sizes_{layer_name}.npy"))
    signal = np.stack([np.load(f) for f in files])
    noise = np.stack([np.load(f.replace("signal", "noise")) for f in files])
    noise = noise.mean(axis=0)
    signal = signal.mean(axis=0)
    deformation = noise / signal
    if not np.all(deformation[-1] == 0):
        print(deformation[-1].max())
    assert not np.any(np.isnan(deformation))

    num_groups = deformation.shape[1]
    channels = reduction_to_block_size.shape[1]

    group_size = channels // num_groups
    assert num_groups * group_size == channels
    deformation_index_to_reduction_index = []
    chosen_block_sizes = []
    for cur_target_deformation_index, cur_target_deformation in enumerate(TARGET_DEFORMATIONS):

        valid_block_sizes = deformation <= cur_target_deformation

        chosen_reduction_index = np.argmax((TARGET_REDUCTIONS / 2)[::-1][:,np.newaxis] + valid_block_sizes, axis=0)
        chosen_block_sizes.append(reduction_to_block_size[np.repeat(chosen_reduction_index, group_size), range(channels)])
        deformation_index_to_reduction_index.append(chosen_reduction_index)
    # assert time.time() <= 1661070082.368254 + 3600, "Is Deformation range good enough?"

    deformation_index_to_reduction_index = np.array(deformation_index_to_reduction_index)
    chosen_block_sizes = np.array(chosen_block_sizes)
    deformation_to_reduction = TARGET_REDUCTIONS[deformation_index_to_reduction_index.flatten()].reshape(deformation_index_to_reduction_index.shape).mean(axis=1)

    reduction_to_block_size_new = []
    for reduction in TARGET_REDUCTIONS:
        index = np.argmin(np.abs(deformation_to_reduction - reduction))
        reduction_to_block_size_new.append(chosen_block_sizes[index])
    reduction_to_block_size_new = np.array(reduction_to_block_size_new)

    redundancy_arr = [0]
    for i in range(1, TARGET_REDUCTIONS.shape[0]):
        if np.all(reduction_to_block_size_new[i] == reduction_to_block_size_new[i-1]):
            redundancy_arr.append(redundancy_arr[-1])
        else:
            redundancy_arr.append(i)
    redundancy_arr = np.array(redundancy_arr)

    np.save(file=os.path.join(output_path, f"redundancy_arr_{layer_name}.npy"), arr=redundancy_arr)
    np.save(file=os.path.join(output_path, f"reduction_to_block_sizes_{layer_name}.npy"), arr=reduction_to_block_size_new)

    # print('Hey')
    # block_sizes = np.array(LAYER_NAME_TO_BLOCK_SIZES[layer_name])
    # reductions_spec = 1 / (block_sizes[:, 0] * block_sizes[:, 1])
    # a = reductions_spec[reduction_to_block_size_new.flatten().astype(np.int32)].reshape(reduction_to_block_size_new.shape)
    # b = reductions_spec[reduction_to_block_size.flatten().astype(np.int32)].reshape(reduction_to_block_size_new.shape)
    # plt.plot(TARGET_REDUCTIONS, a.mean(axis=1)) # should be close to identity
