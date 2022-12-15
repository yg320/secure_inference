import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import os
import numpy as np
layers = ['conv1', 'layer1_0_0', 'layer2_0_0', 'layer2_0_1', 'layer2_1_0', 'layer2_1_1', 'layer3_0_0', 'layer3_0_1', 'layer3_1_0', 'layer3_1_1', 'layer3_2_0', 'layer3_2_1', 'layer4_0_0', 'layer4_0_1', 'layer4_1_0', 'layer4_1_1', 'layer4_2_0', 'layer4_2_1', 'layer4_3_0', 'layer4_3_1', 'layer5_0_0', 'layer5_0_1', 'layer5_1_0', 'layer5_1_1', 'layer5_2_0', 'layer5_2_1', 'layer6_0_0', 'layer6_0_1', 'layer6_1_0', 'layer6_1_1', 'layer6_2_0', 'layer6_2_1', 'layer7_0_0', 'layer7_0_1', 'decode_0', 'decode_1', 'decode_2', 'decode_3', 'decode_4', 'decode_5']
out_dir = f"/home/yakir/distortion_approximation/decomposed_channel_distortion_512/"
out_dir_baseline = f"/home/yakir/distortion_approximation/decomposed_channel_distortion_baseline_128/"

batch_index = 0
additive_noise = {layer:[] for layer in layers}
noise = {layer:[] for layer in layers}

for seed in range(6):
    file_name = os.path.join(out_dir, f"{seed}_{batch_index}.pickle")
    file_name_baseline = os.path.join(out_dir_baseline, f"{seed}_{batch_index}.pickle")

    content = pickle.load(open(file_name, 'rb'))
    content_baseline = pickle.load(open(file_name_baseline, 'rb'))

    for layer in layers:
        additive_noise[layer].append(np.array([x['noises_distorted'] for x in content[layer]]).sum(axis=0)[:2].mean())
        noise[layer].append(content_baseline[layer]['noises_distorted'].mean())

for layer_index, layer in enumerate(layers):
    plt.subplot(8,5,layer_index+1)
    m = 0.95*min(min(noise[layer]), min(additive_noise[layer]))
    M = 1.05*max(max(noise[layer]), max(additive_noise[layer]))

    plt.plot([m,M], [m,M])
    plt.xlim([m, M])
    plt.ylim([m, M])
    plt.xticks([])
    plt.yticks([])
    plt.scatter(additive_noise[layer], noise[layer], alpha=0.5)