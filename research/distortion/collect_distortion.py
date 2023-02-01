import pickle
import os
import glob
import numpy as np
from tqdm import tqdm

input_path = "/storage/yakir/secure_inference/outputs_v2/distortions/classification/resnet50_8xb32_in1k/iterative_knapsack_0_4x4/"
out_path = "/storage/yakir/secure_inference/outputs_v2/distortions/classification/resnet50_8xb32_in1k/iterative_knapsack_0_4x4_collected/"
os.makedirs(out_path)
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

for layer_name in tqdm(layer_names):
    glob_pattern = os.path.join(input_path, f"{layer_name}_*.pickle")
    files = glob.glob(glob_pattern)
    assert len(files) == 4, glob_pattern

    distortion = np.stack([pickle.load(open(f, 'rb'))["Noise"] for f in files])
    distortion = distortion.mean(axis=0).T
    distortion = distortion[:-1]
    distortion = np.array(distortion).T
    #
    # loss = np.stack([pickle.load(open(f, 'rb'))["Distorted Loss"] for f in files])
    # loss = loss.mean(axis=0).T
    # loss = loss[:-1].T
    #
    # np.save(os.path.join(out_path, "loss", f"{layer_name}.npy"), -loss)
    np.save(os.path.join(out_path,  f"{layer_name}.npy"), -distortion)

