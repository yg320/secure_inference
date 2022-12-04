import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import os
import numpy as np
import glob
from tqdm import tqdm

num_batches = 256
for seed in [0,1]:
    files = glob.glob(f"/home/yakir/distortion_approximation_v2/baseline_distortion/{seed}_*.pickle")
    contents = []

    for f in files:
        contents.append(pickle.load(open(f, 'rb')))
    noises_distorted = np.array([x['losses_distorted'] for x in contents])[:num_batches].flatten()


    sample_sizes = [1, 2, 4, 8, 16]#, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]#, 16384]
    stds = []
    for sample_size in tqdm(sample_sizes):
        stds.append(np.std([noises_distorted[np.random.choice(len(noises_distorted), size=sample_size, replace=False)].mean() for _ in range(1000)]))

    plt.plot(sample_sizes, stds, "*-", label=noises_distorted.mean())

plt.legend()
print('hey')