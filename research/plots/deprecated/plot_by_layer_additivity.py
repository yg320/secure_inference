import collections

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import os
import numpy as np
import glob
from tqdm import tqdm

BLOCK_NAMES_TO_BLOCK_INDEX = \
            {
                "conv1":0,
                "layer1_0":1,
                "layer2_0":2,
                "layer2_1":3,
                "layer3_0":4,
                "layer3_1":5,
                "layer3_2":6,
                "layer4_0":7,
                "layer4_1":8,
                "layer4_2":9,
                "layer4_3":10,
                "layer5_0":11,
                "layer5_1":12,
                "layer5_2":13,
                "layer6_0":14,
                "layer6_1":15,
                "layer6_2":16,
                "layer7_0":17,
                "decode":18,
                None:19,
            }

layers = ['conv1', 'layer1_0_0', 'layer2_0_0', 'layer2_0_1', 'layer2_1_0', 'layer2_1_1', 'layer3_0_0', 'layer3_0_1', 'layer3_1_0', 'layer3_1_1', 'layer3_2_0', 'layer3_2_1', 'layer4_0_0', 'layer4_0_1', 'layer4_1_0', 'layer4_1_1', 'layer4_2_0', 'layer4_2_1', 'layer4_3_0', 'layer4_3_1', 'layer5_0_0', 'layer5_0_1', 'layer5_1_0', 'layer5_1_1', 'layer5_2_0', 'layer5_2_1', 'layer6_0_0', 'layer6_0_1', 'layer6_1_0', 'layer6_1_1', 'layer6_2_0', 'layer6_2_1', 'layer7_0_0', 'layer7_0_1']

for block_index in [3]:
    plt.figure()
    plt.suptitle(block_index)
    all_losses = []
    all_snrs = []
    for layer_index, layer in enumerate(layers[:25]):
        losses = []
        noises = []
        signals = []
        losses_distorted = []
        for seed in range(168):
            cur_losses_distorted = []
            cur_noises = []
            cur_signals = []
            cur_losses = []
            for batch_index in range(2):

                f = f"/home/yakir/distortion_approximation_v2/by_layer_additivity_v2/{layer}_{seed}_{batch_index}.pickle"
                content = pickle.load(open(f, 'rb'))

                additive_assets = content['additive_assets']

                key_to_use = sorted(additive_assets[0]["noises_distorted"].keys(), key=lambda x: BLOCK_NAMES_TO_BLOCK_INDEX[x])[block_index]
                N_channels = len(additive_assets)
                cur_noises.append(np.array([x['noises_distorted'][key_to_use] for x in additive_assets]).sum(axis=0))
                cur_signals.append(np.array([x['signals_distorted'][key_to_use] for x in additive_assets]).sum(axis=0))
                cur_losses_distorted.append(np.array(content["assets"]['losses_distorted']))
                cur_losses.append(np.array(content["assets"]['losses_baseline']))

            losses_distorted.append(np.array(cur_losses_distorted).mean())
            losses.append(np.array(cur_losses).mean())
            noises.append(np.array(cur_noises).mean())
            signals.append(np.array(cur_signals).mean())
        losses_ratio = np.array(losses_distorted) / np.array(losses)
        snr = np.log(np.array(signals) / np.array(noises))

        x = str(np.corrcoef(snr, losses_ratio)[0,1])[:5]
        plt.subplot(5,5,layer_index+1)
        plt.scatter(snr, losses_ratio, label=f"{layer}: {x}", alpha=0.3)
        plt.legend()
