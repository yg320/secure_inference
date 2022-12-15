import collections

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import os
import numpy as np
import glob
from tqdm import tqdm

num_batches = 1
# layers = ['layer4_1_1']

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

files = [f"/home/yakir/distortion_approximation_v2/between_layers_additivity/{seed}_0.pickle" for seed in range(36)]
contents = [pickle.load(open(f, 'rb')) for f in files ]

additive_noises = np.array([[x["layers_asset"][i]['noises_distorted']['decode'] for i in range(40)] for x in contents]).sum(axis=1).mean(axis=1)
noises = np.array([x["assets"]['noises_distorted']['decode'] for x in contents]).mean(axis=1)
np.corrcoef(additive_noises, noises)
plt.scatter(additive_noises, noises)
block = list(contents[0]["assets"]['noises_distorted'].keys())[0]
noises_distorted = np.array([x["assets"]['noises_distorted'][block] for x in contents]).mean(axis=1)
additive_noises_distorted = np.array([np.array([x['additive_assets'][i]['noises_distorted'][block] for i in range(len(contents[0]['additive_assets']))]) for x in contents]).sum(axis=1).mean(axis=1)
print(np.corrcoef(noises_distorted, additive_noises_distorted)[0,1])
