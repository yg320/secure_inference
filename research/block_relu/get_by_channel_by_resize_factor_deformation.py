# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
import argparse
# import time
import os
from tqdm import tqdm
import torch
import numpy as np
import time
from research.block_relu.consts import LAYER_NAME_TO_BLOCK_SIZES, LAYER_NAMES, LAYER_NAME_TO_CHANNELS, \
    LAYER_NAME_TO_BLOCK_NAME, BLOCK_NAMES, BY_CHANNEL_BY_BLOCK_SIZE_DEFORMATION_PROXY_SPEC
from research.block_relu.utils import get_model, get_data, run_model_block, get_layer, set_bReLU_layers, set_layers


def center_crop(tensor, size):
    h = (tensor.shape[1] - size) // 2
    w = (tensor.shape[2] - size) // 2
    return tensor[:, h:h + size, w:w + size]
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu_id', type=int, default=0)

    deformation_path = "/home/yakir/Data2/assets_v2/by_channel_by_block_size_deformation_arrays"
    config = "/home/yakir/PycharmProjects/mmsegmentation/configs/secure_semantic_segmentation/baseline_40k_finetune_tmp.py"
    checkpoint = "/home/yakir/PycharmProjects/mmsegmentation/work_dirs/baseline_40k/latest.pth"
    os.makedirs(deformation_path, exist_ok=True)

    args = parser.parse_args()
    batch_index = args.batch_index
    batch_size = args.batch_size
    gpu_id = args.gpu_id
    resnet_block_start = 0
    resnet_block_end = 18
    im_size = 512
    device = f"cuda:{gpu_id}"
    model = get_model(
        config=config,
        gpu_id=gpu_id,
        checkpoint_path=checkpoint
    )

    dataset = get_data(config)

    with torch.no_grad():
        batch_indices = range(batch_index * batch_size, (batch_index + 1) * batch_size)
        batch = torch.stack([center_crop(dataset[sample_id]['img'][0], im_size) for sample_id in batch_indices]).to(device)
        activations = [batch]
        for block_index in range(resnet_block_start, resnet_block_end):
            activations.append(run_model_block(model, activations[block_index], BLOCK_NAMES[block_index]))

        resnet_block_name_to_activation = dict(zip(BLOCK_NAMES, activations))
        # expected_time = 0
        for layer_index, layer_name in enumerate(LAYER_NAMES):
            torch.cuda.empty_cache()
            cur_block_name = LAYER_NAME_TO_BLOCK_NAME[layer_name]
            next_block_name = BY_CHANNEL_BY_BLOCK_SIZE_DEFORMATION_PROXY_SPEC[layer_name]

            cur_block_index = np.argwhere(np.array(BLOCK_NAMES) == cur_block_name)[0, 0]
            next_block_index = np.argwhere(np.array(BLOCK_NAMES) == next_block_name)[0, 0]

            layer_block_sizes = LAYER_NAME_TO_BLOCK_SIZES[layer_name]
            cur_tensor = resnet_block_name_to_activation[cur_block_name]
            next_tensor = resnet_block_name_to_activation[next_block_name]
            noise_f_name = os.path.join(deformation_path, f"noise_{layer_name}_batch_{batch_index}_{batch_size}.npy")
            signal_f_name = os.path.join(deformation_path, f"signal_{layer_name}_batch_{batch_index}_{batch_size}.npy")
            if os.path.exists(noise_f_name):
                continue
            channels = LAYER_NAME_TO_CHANNELS[layer_name]
            noise = np.zeros((len(layer_block_sizes), channels))
            signal = np.zeros((len(layer_block_sizes), channels))

            t0 = time.time()
            for channel in tqdm(range(channels), desc=f"Batch={batch_index} Layer={layer_index}"):
                for block_size_index in range(len(layer_block_sizes)):
                    out = cur_tensor
                    block_size_indices = np.zeros(shape=channels, dtype=np.int32)
                    block_size_indices[channel] = block_size_index

                    orig_layer = get_layer(model, layer_name)
                    set_bReLU_layers(model, {layer_name: (block_size_indices, layer_block_sizes)})

                    for block_index in range(cur_block_index, next_block_index):
                        out = run_model_block(model, out, BLOCK_NAMES[block_index])

                    set_layers(model, {layer_name: orig_layer})

                    noise[block_size_index, channel] = float(((out - next_tensor) ** 2).mean())
                    signal[block_size_index, channel] = float((next_tensor ** 2).mean())
            # t1 = time.time() - t0
            # extra = (t1 * channels)
            # expected_time += extra
            # print(layer_index, extra, expected_time)

            np.save(file=noise_f_name, arr=noise)
            np.save(file=signal_f_name, arr=signal)


if __name__ == '__main__':
    main()
