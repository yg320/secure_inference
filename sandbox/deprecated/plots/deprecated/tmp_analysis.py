import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np

N = 1100
mIoU_noises_arr = np.stack([np.load("/home/yakir/tmp/0_mIoU_noises_arr.npy")[:N], np.load("/home/yakir/tmp/2_mIoU_noises_arr.npy")[:N]], axis=0).mean(axis=0)
mIoU_baselines_arr = np.stack([np.load("/home/yakir/tmp/0_mIoU_baselines_arr.npy")[:N], np.load("/home/yakir/tmp/2_mIoU_baselines_arr.npy")[:N]], axis=0).mean(axis=0)
losses_baselines_arr = np.stack([np.load("/home/yakir/tmp/0_losses_baselines_arr.npy")[:N], np.load("/home/yakir/tmp/2_losses_baselines_arr.npy")[:N]], axis=0).mean(axis=0)
losses_noises_arr = np.stack([np.load("/home/yakir/tmp/0_losses_noises_arr.npy")[:N], np.load("/home/yakir/tmp/2_losses_noises_arr.npy")[:N]], axis=0).mean(axis=0)
all_signals_arr = np.stack([np.load("/home/yakir/tmp/0_all_signals_arr.npy")[:N], np.load("/home/yakir/tmp/2_all_signals_arr.npy")[:N]], axis=0).mean(axis=0)
all_noises_arr = np.stack([np.load("/home/yakir/tmp/0_all_noises_arr.npy")[:N], np.load("/home/yakir/tmp/2_all_noises_arr.npy")[:N]], axis=0).mean(axis=0)

distortion = all_noises_arr / all_signals_arr

x = []
for i in range(18):
    x.append(np.corrcoef(distortion[:, i], losses_noises_arr / losses_baselines_arr)[0,1])
plt.plot(x)
mat = np.zeros(shape=(18,18))
for i in range(18):
    for j in range(18):
        mat[i,j] = np.corrcoef(distortion[:, i], distortion[:, j])[0,1]
plt.imshow(mat)
plt.plot(x)
np.corrcoef(distortion[:, -1], np.abs(losses_noises_arr / (1e-10+losses_baselines_arr)))[0,1]

plt.scatter(distortion[:, -2], losses_noises_arr / (1e-10+losses_baselines_arr), alpha=0.1)
plt.scatter(distortion[:, -1], losses_noises_arr / (1e-10+losses_baselines_arr), alpha=0.1)
plt.scatter(distortion[:, -1], mIoU_noises_arr / (1e-10+mIoU_baselines_arr), alpha=0.1)
plt.scatter(distortion[:, -3], mIoU_noises_arr / (1e-10+mIoU_baselines_arr), alpha=0.1)
plt.scatter(distortion[:, -5], mIoU_noises_arr / (1e-10+mIoU_baselines_arr), alpha=0.1)
plt.scatter(distortion[:, -7], mIoU_noises_arr / (1e-10+mIoU_baselines_arr), alpha=0.1)

np.corrcoef(distortion[:, -1], mIoU_noises_arr / (1e-10+mIoU_baselines_arr))

spec = {'stem_2': 2, 'stem_5': 2, 'stem_8': 4, 'layer1_0_1': 1,
        'layer1_0_2': 1, 'layer1_0_3': 4, 'layer1_1_1': 1,
        'layer1_1_2': 1,
        'layer1_1_3': 4, 'layer1_2_1': 1, 'layer1_2_2': 1,
        'layer1_2_3': 4,
        'layer2_0_1': 2, 'layer2_0_2': 0.5, 'layer2_0_3': 2,
        'layer2_1_1': 0.5,
        'layer2_1_2': 0.5, 'layer2_1_3': 2, 'layer2_2_1': 0.5,
        'layer2_2_2': 0.5,
        'layer2_2_3': 2, 'layer2_3_1': 0.5, 'layer2_3_2': 0.5,
        'layer2_3_3': 2,
        'layer3_0_1': 1, 'layer3_0_2': 1, 'layer3_0_3': 4,
        'layer3_1_1': 1,
        'layer3_1_2': 1, 'layer3_1_3': 4, 'layer3_2_1': 1,
        'layer3_2_2': 1,
        'layer3_2_3': 4, 'layer3_3_1': 1, 'layer3_3_2': 1,
        'layer3_3_3': 4,
        'layer3_4_1': 1, 'layer3_4_2': 1, 'layer3_4_3': 4,
        'layer3_5_1': 1,
        'layer3_5_2': 1, 'layer3_5_3': 4, 'layer4_0_1': 2,
        'layer4_0_2': 2,
        'layer4_0_3': 8, 'layer4_1_1': 2, 'layer4_1_2': 2,
        'layer4_1_3': 8,
        'layer4_2_1': 2, 'layer4_2_2': 2, 'layer4_2_3': 8,
        'decode_0': 0,
        'decode_1': 2, 'decode_2': 2, 'decode_3': 2,
        'decode_4': 2,
        'decode_5': 2}
[sum([spec[x] for x in l]) for l in c]


