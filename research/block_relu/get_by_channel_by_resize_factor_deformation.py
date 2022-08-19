import argparse
import time
import os
from tqdm import tqdm
import torch
import numpy as np

from research.block_relu.consts import BLOCK_SIZES, LAYER_NAMES, LAYER_NAME_TO_CHANNELS, LAYER_NAME_TO_BLOCK_NAME, BLOCK_NAMES, NETWORK_MULTI_BLOCK_SPEC
from research.block_relu.utils import get_model, get_data, run_model_block, get_layer, set_bReLU_layers, set_layers

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--sample_id_start', type=int, default=0)
    parser.add_argument('--sample_id_end', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)

    deformation_path = "/home/yakir/Data2/assets/by_channel_by_block_size_deformation_arrays_tmp2"

    os.makedirs(deformation_path, exist_ok=True)
    config = "/home/yakir/PycharmProjects/mmsegmentation/configs/secure_semantic_segmentation/baseline_40k_finetune_tmp.py"
    checkpoint = "/home/yakir/PycharmProjects/mmsegmentation/work_dirs/baseline_40k/latest.pth"

    args = parser.parse_args()
    sample_id_start = args.sample_id_start
    sample_id_end = args.sample_id_end
    gpu_id = args.gpu_id
    resnet_block_start = 0
    resnet_block_end = 18
    device = f"cuda:{gpu_id}"
    model = get_model(
        config=config,
        gpu_id=gpu_id,
        checkpoint_path=checkpoint
    )

    dataset = get_data(config)

    with torch.no_grad():
        for sample_id in range(sample_id_start, sample_id_end):
            t0 = time.time()
            sample = dataset[sample_id]
            activations = [torch.unsqueeze(sample['img'][0].to(device), dim=0)]
            for block_index in range(resnet_block_start, resnet_block_end):
                activations.append(run_model_block(model, activations[block_index], BLOCK_NAMES[block_index]))

            resnet_block_name_to_activation = dict(zip(BLOCK_NAMES, activations))

            for layer_name in LAYER_NAMES:
                torch.cuda.empty_cache()
                cur_block_name = LAYER_NAME_TO_BLOCK_NAME[layer_name]
                next_block_name = NETWORK_MULTI_BLOCK_SPEC[layer_name]

                cur_block_index = np.argwhere(np.array(BLOCK_NAMES) == cur_block_name)[0, 0]
                next_block_index = np.argwhere(np.array(BLOCK_NAMES) == next_block_name)[0, 0]

                cur_tensor = resnet_block_name_to_activation[cur_block_name]
                next_tensor = resnet_block_name_to_activation[next_block_name]
                noise_f_name = os.path.join(deformation_path, f"noise_{layer_name}_sample_{sample_id}.npy")
                signal_f_name = os.path.join(deformation_path, f"signal_{layer_name}_sample_{sample_id}.npy")
                if os.path.exists(noise_f_name):
                    continue
                channels = LAYER_NAME_TO_CHANNELS[layer_name]
                noise = np.zeros((len(BLOCK_SIZES), channels))
                signal = np.zeros((len(BLOCK_SIZES), channels))
                for channel in tqdm(range(channels), desc=f"Sample={sample_id} Layer={layer_name}"):
                    for block_size_index in range(len(BLOCK_SIZES)):
                        out = cur_tensor
                        block_size_indices = np.zeros(shape=channels, dtype=np.int32)
                        block_size_indices[channel] = block_size_index

                        orig_layer = get_layer(model, layer_name)
                        set_bReLU_layers(model, {layer_name: block_size_indices})

                        for block_index in range(cur_block_index, next_block_index):
                            out = run_model_block(model, out, BLOCK_NAMES[block_index])

                        set_layers(model, {layer_name: orig_layer})

                        noise[block_size_index, channel] = float(((out - next_tensor) ** 2).mean())
                        signal[block_size_index, channel] = float((next_tensor ** 2).mean())

                np.save(file=noise_f_name, arr=noise)
                np.save(file=signal_f_name, arr=signal)
            print(time.time() - t0)


if __name__ == '__main__':
    main()
