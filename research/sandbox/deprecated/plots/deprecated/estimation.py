import collections

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import glob
import collections

seeds = 75
noises = []
signals_distorted = []
for seed in range(seeds):

    f = f"/home/yakir/distortion_approximation/expected_512/{seed}.pickle"
    content = pickle.load(open(f, 'rb'))
    noises.append(content["noises_distorted"])

estimated_noises = []

signals_additive = []
for seed in range(seeds):
    f = f"/home/yakir/distortion_approximation/estimation_512/{seed}.pickle"
    content = pickle.load(open(f, 'rb'))
    estimated_noises.append(content['noises_additive'])


noises = np.array(noises)[..., 0]
estimated_noises = np.array(estimated_noises)[..., 0]
for i in range(40):
    print(np.corrcoef(estimated_noises.sum(axis=1), noises)[0,1])

plt.scatter(estimated_noises.sum(axis=1), noises)
plt.scatter(noises[:,0], estimated_noises.sum(axis=1)[:,0])
np.corrcoef(estimated_losses_baseline[:,:8].flatten(), losses_baseline.flatten())
