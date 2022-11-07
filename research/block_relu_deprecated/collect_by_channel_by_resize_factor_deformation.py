import os
import numpy as np
import glob
from tqdm import tqdm
from research.block_relu.consts import LAYER_NAMES, LAYER_NAME_TO_BLOCK_SIZES, TARGET_DEFORMATIONS

deformation_path = "/home/yakir/Data2/assets_v2/by_channel_by_block_size_deformation_arrays"
out_dir = "/home/yakir/Data2/assets_v2/collected_by_channel_by_resize_factor_deformation"

os.makedirs(out_dir)
block_relu_index_to_block_sizes = {}
block_relu_index_to_reduction = {}

block_sizes_spec = {}
for layer_index, layer_name in tqdm(enumerate(LAYER_NAMES)):

    files = glob.glob(os.path.join(deformation_path, f"signal_{layer_name}_batch_*.npy"))

    signal = np.stack([np.load(f) for f in files])
    noise = np.stack([np.load(f.replace("signal", "noise")) for f in files])
    noise = noise.mean(axis=0)
    signal = signal.mean(axis=0)
    deformation = noise / signal
    deformation[0] = 0
    assert not np.any(np.isnan(deformation))

    block_sizes = LAYER_NAME_TO_BLOCK_SIZES[layer_name]
    activation_reduction = np.array([1/x[0]/x[1] for x in block_sizes])
    deformation_and_channel_to_block_size = []
    deformation_and_channel_to_reduction = []

    broadcast_block_size = np.repeat(np.array([x[0] * x[1] for x in block_sizes])[:, np.newaxis], deformation.shape[1], axis=1)
    for cur_target_deformation_index, cur_target_deformation in enumerate(TARGET_DEFORMATIONS):

        valid_block_sizes = deformation <= cur_target_deformation
        block_sizes_with_zero_on_non_valid_blocks = broadcast_block_size * valid_block_sizes

        cur_block_sizes = np.argmax(block_sizes_with_zero_on_non_valid_blocks, axis=0)
        cur_reduction = activation_reduction[cur_block_sizes]

        deformation_and_channel_to_block_size.append(cur_block_sizes)
        deformation_and_channel_to_reduction.append(cur_reduction)

    deformation_and_channel_to_reduction = np.array(deformation_and_channel_to_reduction)
    deformation_and_channel_to_block_size = np.array(deformation_and_channel_to_block_size)

    np.save(file=os.path.join(out_dir, f"deformation_and_channel_to_reduction_{layer_name}.npy"), arr=deformation_and_channel_to_reduction)
    np.save(file=os.path.join(out_dir, f"deformation_and_channel_to_block_size_{layer_name}.npy"), arr=deformation_and_channel_to_block_size)
