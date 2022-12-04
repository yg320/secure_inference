import collections

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import os
import numpy as np
import glob
from tqdm import tqdm

layers = ['layer1_0_0', 'layer2_0_1', 'layer2_1_1', 'layer3_0_1', 'layer3_1_1', 'layer3_2_1', 'layer4_0_1', 'layer4_1_0', 'layer4_1_1', 'layer4_2_1', 'layer4_3_1', 'layer5_0_0', 'layer5_0_1', 'layer5_1_1', 'layer5_2_1', 'layer6_0_1', 'layer6_1_1', 'layer6_2_0', 'layer7_0_1', 'decode_3']
num_batches = 128

stuff = []
for layer in tqdm(layers):
    noises_distorted_means = collections.defaultdict(list)
    noises_distorted_stds = collections.defaultdict(list)
    losses = []

    for seed in range(170):

        files = [f"/home/yakir/distortion_approximation_v2/by_layer_baseline_distortion/{layer}_{seed}_{batch_index}.pickle" for batch_index in range(num_batches)]
        contents = []

        for f in files:
            contents.append(pickle.load(open(f, 'rb')))

        blocks = list(contents[0]['noises_distorted'].keys())
        noises_distorted = {block: np.array([x['noises_distorted'][block] for x in contents])[:num_batches].flatten() for block in blocks}

        distortion_last_layer = np.array([x['noises_distorted']["decode"] for x in contents])[:num_batches].flatten().mean()
        loss = np.array([np.array(x['losses_distorted']) / (np.array(x['losses_baseline'])+1e-10) for x in contents])[:num_batches].flatten().mean()
        losses.append(loss)

        for block in blocks:
            noises_distorted_stds[block].append(noises_distorted[block].std())
            noises_distorted_means[block].append(noises_distorted[block].mean())

    # plt.scatter(noises_distorted_means['layer1_0'], losses)

    stuff.append([np.corrcoef(noises_distorted_means[b], losses)[0,1] for b in blocks])
np.array([x[:7] for x in stuff if len(x) >= 7]).mean(axis=0)
print('jkj')
# layer1_0_0 0.6431078538043705
# layer2_0_1 0.438077039552152
# layer2_1_1 0.6137404429804468
# layer3_0_1 0.7709757010430632
# layer3_1_1 0.8404354010773195
# layer3_2_1 0.5317353960950628
# layer4_0_1 0.7474431477254628
# layer4_1_0 0.9357319801524179
# layer4_1_1 0.7029078681647077
# layer4_2_1 0.605992430781918
# layer4_3_1 0.4201569239333726
# layer5_0_0 0.6796298088054359
# layer5_0_1 0.5913856022793748
# layer5_1_1 0.3355895526198341
# layer5_2_1 0.5734197599112266
# layer6_0_1 0.3816448674061922
# layer6_1_1 0.539071748911032
# layer6_2_0 0.8940915500850319
# layer7_0_1 0.45939993910195354
# decode_3 1.0