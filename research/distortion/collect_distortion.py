import pickle
import os
import glob
import numpy as np
from tqdm import tqdm

input_path = "/storage/yakir/secure_inference/outputs_v2/distortions/classification/resnet50_8xb32_in1k/iterative_knapsack_0_4x4/"
out_path = "/storage/yakir/secure_inference/outputs_v2/distortions/classification/resnet50_8xb32_in1k/iterative_knapsack_0_4x4_collected/"
layer_names = [
                "stem",
                'layer1_0_1',
                'layer1_0_2',
                'layer1_0_3',
                'layer1_1_1',
                'layer1_1_2',
                'layer1_1_3',
                'layer1_2_1',
                'layer1_2_2',
                'layer1_2_3',
                'layer2_0_1',
                'layer2_0_2',
                'layer2_0_3',
                'layer2_1_1',
                'layer2_1_2',
                'layer2_1_3',
                'layer2_2_1',
                'layer2_2_2',
                'layer2_2_3',
                'layer2_3_1',
                'layer2_3_2',
                'layer2_3_3',
                'layer3_0_1',
                'layer3_0_2',
                'layer3_0_3',
                'layer3_1_1',
                'layer3_1_2',
                'layer3_1_3',
                'layer3_2_1',
                'layer3_2_2',
                'layer3_2_3',
                'layer3_3_1',
                'layer3_3_2',
                'layer3_3_3',
                'layer3_4_1',
                'layer3_4_2',
                'layer3_4_3',
                'layer3_5_1',
                'layer3_5_2',
                'layer3_5_3',
                'layer4_0_1',
                'layer4_0_2',
                'layer4_0_3',
                'layer4_1_1',
                'layer4_1_2',
                'layer4_1_3',
                'layer4_2_1',
                'layer4_2_2',
                'layer4_2_3',

            ]
expected_num_files = 4

input_path = "/storage/yakir/secure_inference/outputs_v2/distortions/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k/"
out_path = "/storage/yakir/secure_inference/outputs_v2/distortions/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_collected"
layer_names = [
                "conv1",
                "layer1_0_0",
                "layer2_0_0",
                "layer2_0_1",
                "layer2_1_0",
                "layer2_1_1",
                "layer3_0_0",
                "layer3_0_1",
                "layer3_1_0",
                "layer3_1_1",
                "layer3_2_0",
                "layer3_2_1",
                "layer4_0_0",
                "layer4_0_1",
                "layer4_1_0",
                "layer4_1_1",
                "layer4_2_0",
                "layer4_2_1",
                "layer4_3_0",
                "layer4_3_1",
                "layer5_0_0",
                "layer5_0_1",
                "layer5_1_0",
                "layer5_1_1",
                "layer5_2_0",
                "layer5_2_1",
                "layer6_0_0",
                "layer6_0_1",
                "layer6_1_0",
                "layer6_1_1",
                "layer6_2_0",
                "layer6_2_1",
                "layer7_0_0",
                "layer7_0_1",
                "decode_0",
                "decode_1",
                "decode_2",
                "decode_3",
                "decode_4",
                "decode_5",
            ]
expected_num_files = 4

os.makedirs(out_path)

for layer_name in tqdm(layer_names):
    glob_pattern = os.path.join(input_path, f"{layer_name}_*.pickle")
    files = glob.glob(glob_pattern)
    assert len(files) == expected_num_files, glob_pattern

    distortion = np.stack([pickle.load(open(f, 'rb'))["Noise"] for f in files])

    distortion = distortion.mean(axis=0).T
    if len(distortion.shape) == 3:
        distortion = distortion.mean(axis=0)
    distortion = distortion[:-1]
    distortion = np.array(distortion).T

    np.save(os.path.join(out_path,  f"{layer_name}.npy"), -distortion)

