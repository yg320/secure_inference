import matplotlib
matplotlib.use("TkAgg")
import argparse
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import pickle
import glob

from matplotlib import pyplot as plt
import time
channel_ratio = 0.5
seed = 0
noise_files = glob.glob(f"/home/yakir/Data2/random_channel_stats_{channel_ratio}_{seed}_v2/noise_*_0.npy")
signal_files = [f.replace("noise", "signal") for f in noise_files]
loss_files = [f.replace("noise", "loss") for f in noise_files]
relus_files = [f.replace("noise", "relus_count") for f in noise_files]
distorted_loss_files = [f.replace("noise", "distorted_loss") for f in noise_files]

noises = []
signals = []
losses = []
relus = []
distorted_loss = []

for n_f, s_f, l_f, r_f, nd_f in zip(noise_files, signal_files, loss_files, relus_files, distorted_loss_files):
    try:
        noises.append(np.stack([np.load(n_f), np.load(n_f.replace("_0.npy", "_1.npy"))], axis=0).mean(axis=0))
        signals.append(np.stack([np.load(s_f), np.load(s_f.replace("_0.npy", "_1.npy"))], axis=0).mean(axis=0))
        losses.append(np.stack([np.load(l_f), np.load(l_f.replace("_0.npy", "_1.npy"))], axis=0).mean(axis=0))
        relus.append(np.stack([np.load(r_f), np.load(r_f.replace("_0.npy", "_1.npy"))], axis=0).mean(axis=0))
        distorted_loss.append(np.stack([np.load(nd_f), np.load(nd_f.replace("_0.npy", "_1.npy"))], axis=0).mean(axis=0))
    except FileNotFoundError:
        pass
noises = np.array(noises)
signals = np.array(signals)
losses = np.array(losses)
relus = np.array(relus)
distorted_loss = np.array(distorted_loss)

xxx = noises.sum(axis=1)
yyy = distorted_loss.mean(axis=1)
print(np.corrcoef(xxx, losses))
print(np.corrcoef(yyy, losses))
# np.corrcoef(distorted_loss.mean(axis=1), losses)
# plt.scatter(noises.mean(axis=1), losses)
plt.scatter(xxx, losses)
plt.scatter(yyy, losses)
# np.corrcoef(yyy[yyy<0.45210819], losses[yyy<0.45210819])
print('fds')

yyy<0.45257