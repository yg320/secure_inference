import collections

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import glob
import collections

counter = 0
files = glob.glob("/home/yakir/distortion_approximation/get_additive_sample_estimation/*.pickle")
layers = ['conv1', 'layer2_1_0', 'layer3_1_0', 'layer4_0_0', 'layer4_2_0', 'layer5_0_0', 'layer5_2_0', 'layer6_1_0', 'layer7_0_0', 'decode_2']

noises_additive_channel_distorted = collections.defaultdict(list)
noises_distorted = collections.defaultdict(list)
loss_distorted = collections.defaultdict(list)
loss_baseline = collections.defaultdict(list)
for f in files:
    content = pickle.load(open(f, 'rb'))
    for layer in layers:
        noises_additive_channel_distorted[layer].append(content[layer]['noises_additive_channel_distorted'].sum())
        noises_distorted[layer].append(content[layer]['noises_distorted'])
        loss_baseline[layer].append(content[layer]['loss_baseline'])
        loss_distorted[layer].append(content[layer]['loss_distorted'])

import numpy as np
loss_distorted_layer4_2_0 = np.array([float(x) for x in loss_distorted['layer4_2_0']])
loss_baseline_layer4_2_0 = np.array([float(x) for x in loss_baseline['layer4_2_0']])
noises_distorted_layer4_2_0 = noises_distorted['layer4_2_0']
np.corrcoef(noises_additive_channel_distorted['layer4_2_0'], noises_distorted_layer4_2_0/loss_baseline_layer4_2_0)

for layer_name in layers:
    counter += 1
    plt.subplot(2,5,counter)
    noises_additive_channel_distorted_cur = noises_additive_channel_distorted[layer_name]
    noises_distorted_cur = noises_distorted[layer_name]
    m = 0.8*min(min(noises_additive_channel_distorted_cur), min(noises_distorted_cur))
    M = max(max(noises_additive_channel_distorted_cur), max(noises_distorted_cur))/0.8
    plt.plot([m,M], [m,M])
    plt.xlim([m,M])
    plt.ylim([m,M])
    plt.scatter(noises_additive_channel_distorted_cur, noises_distorted_cur, color="black", alpha=0.2)
    plt.title(layer_name)