import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

use_last_layer = False
ex = "/use_last_layer" if use_last_layer else ""
s = 19 if use_last_layer else -1
noise_real_mat_0 = np.load(f"/home/yakir/Data2/assets_v3/additive_deformation_estimation/coco_stuff164k/ResNetV1c{ex}/noise_real_mat_0.npy")
noise_estimated_mat_0 = np.load(f"/home/yakir/Data2/assets_v3/additive_deformation_estimation/coco_stuff164k/ResNetV1c{ex}/noise_estimated_mat_0.npy")
noise_real_mat_1 = np.load(f"/home/yakir/Data2/assets_v3/additive_deformation_estimation/coco_stuff164k/ResNetV1c{ex}/noise_real_mat_1.npy")
noise_estimated_mat_1 = np.load(f"/home/yakir/Data2/assets_v3/additive_deformation_estimation/coco_stuff164k/ResNetV1c{ex}/noise_estimated_mat_1.npy")

noise_real_mat = np.zeros(shape=(5000, 57))
noise_estimated_mat = np.zeros(shape=(5000, 57))
noise_real_mat[::2] = noise_real_mat_0
noise_estimated_mat[::2] = noise_estimated_mat_0
noise_real_mat[1::2] = noise_real_mat_1
noise_estimated_mat[1::2] = noise_estimated_mat_1

#.shape
plt.suptitle("Real Deformation vs Estimated Deformation - Last Layer")
counter = 1
for layer in range(57):
    if layer == 51:
        continue
    sample = np.argwhere(noise_estimated_mat[:,s] > 0).max() + 1
    M = 1.05*max(noise_real_mat[:sample, layer].max(), noise_estimated_mat[:sample,layer].max())
    m = 0.95*min(noise_real_mat[:sample, layer].min(), noise_estimated_mat[:sample,layer].min())
    if m == M == 0.0:
        m = -0.01
        M = 0.01
    plt.subplot(7,8, counter)
    counter += 1
    plt.plot([m,M], [m,M], color="black")
    plt.scatter(noise_real_mat[:sample, layer], noise_estimated_mat[:sample, layer], alpha=0.5, s=20, color="black")
    plt.xlim([m,M])
    plt.ylim([m,M])
    plt.xticks([])
    plt.yticks([])
    xxx = str(np.corrcoef(noise_real_mat[:sample, layer], noise_estimated_mat[:sample, layer])[0,1])[:4]
    plt.gca().set_title(f"Layer {layer} (corr-{xxx})", y=1.0, pad=-8, fontsize=8)

    if layer == 24:
        plt.ylabel("Additive Distortion Estimation", fontsize=18)

    if layer == 52:
        plt.xlabel("Real Distortion", fontsize=18)

for i in range(57):
    print(noise_real_mat[0,i])
    print(noise_estimated_mat[0,i])
    print('=====')




















import pickle
import numpy as np
results_noise, results_orig = pickle.load(open("/home/yakir/Data2/assets_v3/additive_deformation_estimation/coco_stuff164k/ResNetV1c/data.pickle", 'rb'))
[x[-1] for x in results_noise]
[x[0][6] for x in results_noise['layer2_2_2']]