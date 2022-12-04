import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
d = dict()

# for channel_ratio in tqdm([0.2]):
#     for seed in tqdm(range(1000)):
#         noise_files = glob.glob(f"/home/yakir/Data2/random_channel_stats/baseline/random_channel_stats_{channel_ratio}_{seed}_v2/noise_*_0.npy")
#         # noise_files = glob.glob(f"/home/yakir/Data2/random_channel_stats/small_blocks/{channel_ratio}_{seed}_v2/noise_*_0.npy")
#         signal_files = [f.replace("noise", "signal") for f in noise_files]
#         loss_files = [f.replace("noise", "loss") for f in noise_files]
#         relus_files = [f.replace("noise", "relus_count") for f in noise_files]
#         distorted_loss_files = [f.replace("noise", "distorted_loss") for f in noise_files]
#
#         if len(noise_files) == 0:
#             break
#         noises = []
#         signals = []
#         losses = []
#         relus = []
#         distorted_loss = []
#
#         for n_f, s_f, l_f, r_f, nd_f in zip(noise_files, signal_files, loss_files, relus_files, distorted_loss_files):
#             try:
#                 noises.append(np.stack([np.load(n_f), np.load(n_f.replace("_0.npy", "_1.npy"))], axis=0).mean(axis=0))
#                 signals.append(np.stack([np.load(s_f), np.load(s_f.replace("_0.npy", "_1.npy"))], axis=0).mean(axis=0))
#                 losses.append(np.stack([np.load(l_f), np.load(l_f.replace("_0.npy", "_1.npy"))], axis=0).mean(axis=0))
#                 relus.append(np.stack([np.load(r_f), np.load(r_f.replace("_0.npy", "_1.npy"))], axis=0).mean(axis=0))
#                 distorted_loss.append(np.stack([np.load(nd_f), np.load(nd_f.replace("_0.npy", "_1.npy"))], axis=0).mean(axis=0))
#             except FileNotFoundError:
#                 pass
#         noises = np.array(noises)
#         signals = np.array(signals)
#         losses = np.array(losses)
#         relus = np.array(relus)
#         distorted_loss = np.array(distorted_loss)
#
#         noises = noises.sum(axis=1)
#         distorted_loss = distorted_loss.mean(axis=1)
#         d[(channel_ratio, seed)] = noises, distorted_loss, losses

d = pickle.load(open("/home/yakir/tmp.pickle", 'rb'))

out = []
for channel_ratio in tqdm([0.25]):
# for channel_ratio in tqdm([0.1,0.25,0.33,0.5,0.9,1.0]):
    r = []
    for seed in tqdm(range(5)):
        noises, distorted_loss, losses = d[(channel_ratio, seed)]
        r.append(np.corrcoef(noises, losses)[0,1])
    out.append(np.mean(r))
print('fds')
plt.scatter([0.1,0.25,0.33,0.5,0.9,1.0], out)

noises, distorted_loss, losses = d[(0.33, 3)]
plt.scatter(noises, losses)
np.corrcoef(noises, losses)
#
# yyy<0.45257