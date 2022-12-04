import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import numpy as np

assets_0 = pickle.load(open("/home/yakir/Data2/assets_v3/additive_deformation_estimation_new/coco_stuff164k/ResNetV1c/use_last_layer/assets_0.pickle", 'rb'))
assets_1 = pickle.load(open("/home/yakir/Data2/assets_v3/additive_deformation_estimation_new/coco_stuff164k/ResNetV1c/use_last_layer/assets_1.pickle", 'rb'))

counter = 0

for layer_index in range(20):
    counter += 1
    plt.subplot(4,5,counter)
    mIoU_clean = np.hstack([assets_0[layer_index]['mIoU_clean'], assets_1[layer_index]['mIoU_clean']])
    loss_clean = np.hstack([assets_0[layer_index]['loss_clean'], assets_1[layer_index]['loss_clean']])
    mIoU_real = np.hstack([assets_0[layer_index]['mIoU_real'], assets_1[layer_index]['mIoU_real']])
    loss_real = np.hstack([assets_0[layer_index]['loss_real'], assets_1[layer_index]['loss_real']])
    noise_channels = np.vstack([assets_0[layer_index]['noise_channels'], assets_1[layer_index]['noise_channels']])
    loss_channels = np.vstack([assets_0[layer_index]['loss_channels'], assets_1[layer_index]['loss_channels']])
    noise_real = np.hstack([assets_0[layer_index]['noise_real'], assets_1[layer_index]['noise_real']])

    noise_estimated = noise_channels.sum(axis=1)

    M = 1.05 * max(noise_real.max(), noise_estimated.max())
    m = 0.95 * min(noise_real.min(), noise_estimated.min())

    plt.xlim([m,M])
    plt.ylim([m,M])
    plt.xticks([])
    plt.yticks([])
    xxx = str(np.corrcoef(noise_real, noise_estimated)[0,1])[:4]
    plt.gca().set_title(f"Layer {layer_index} (corr-{xxx})", y=1.0, pad=-8, fontsize=8)

    plt.plot([m, M], [m, M], color="black")
    plt.scatter(noise_real, noise_estimated, alpha=0.5, s=20, color="black")






counter = 0

for layer_index in range(20):
    counter += 1
    plt.subplot(4,5,counter)
    mIoU_clean = np.hstack([assets_0[layer_index]['mIoU_clean'], assets_1[layer_index]['mIoU_clean']])
    loss_clean = np.hstack([assets_0[layer_index]['loss_clean'], assets_1[layer_index]['loss_clean']])
    mIoU_real = np.hstack([assets_0[layer_index]['mIoU_real'], assets_1[layer_index]['mIoU_real']])
    loss_real = np.hstack([assets_0[layer_index]['loss_real'], assets_1[layer_index]['loss_real']])
    noise_channels = np.vstack([assets_0[layer_index]['noise_channels'], assets_1[layer_index]['noise_channels']])
    loss_channels = np.vstack([assets_0[layer_index]['loss_channels'], assets_1[layer_index]['loss_channels']])
    mIoU_channels = np.vstack([assets_0[layer_index]['miou_channels'], assets_1[layer_index]['miou_channels']])
    noise_real = np.hstack([assets_0[layer_index]['noise_real'], assets_1[layer_index]['noise_real']])

    noise_estimated = noise_channels.sum(axis=1)
    mIoU_diff_estimated = mIoU_channels.mean(axis=1) - mIoU_clean
    loss_diff_estimated = loss_channels.sum(axis=1) - loss_channels.shape[1]*loss_clean
    # loss_diff_estimated = noise_real
    loss_diff_real = loss_real - loss_clean
    mIoU_diff_real = mIoU_real - mIoU_clean

    M = 1.05 * max(loss_diff_real.max(), loss_diff_estimated.max())
    m = 0.95 * min(loss_diff_real.min(), loss_diff_estimated.min())

    # plt.xlim([m,M])
    # plt.ylim([m,M])
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_title(f"Layer {layer_index}", y=1.0, pad=-8, fontsize=8)

    plt.plot([m, M], [m, M], color="black")
    plt.scatter(loss_real, loss_clean, alpha=0.5, s=20, color ="black")


layer_index = 11
mIoU_clean = np.hstack([assets_0[layer_index]['mIoU_clean'], assets_1[layer_index]['mIoU_clean']])
loss_clean = np.hstack([assets_0[layer_index]['loss_clean'], assets_1[layer_index]['loss_clean']])
mIoU_real = np.hstack([assets_0[layer_index]['mIoU_real'], assets_1[layer_index]['mIoU_real']])
loss_real = np.hstack([assets_0[layer_index]['loss_real'], assets_1[layer_index]['loss_real']])
noise_channels = np.vstack([assets_0[layer_index]['noise_channels'], assets_1[layer_index]['noise_channels']])
loss_channels = np.vstack([assets_0[layer_index]['loss_channels'], assets_1[layer_index]['loss_channels']])
miou_channels = np.vstack([assets_0[layer_index]['miou_channels'], assets_1[layer_index]['miou_channels']])
noise_real = np.hstack([assets_0[layer_index]['noise_real'], assets_1[layer_index]['noise_real']])
loss_estimated = loss_channels.mean(axis=1)
miou_estimated = miou_channels.mean(axis=1)

plt.scatter(noise_real, mIoU_clean - mIoU_real)
plt.scatter(noise_real, loss_clean - loss_real)

plt.scatter(loss_clean - loss_real, loss_clean - loss_estimated)
plt.scatter(mIoU_clean - mIoU_real, mIoU_clean - miou_estimated)
np.corrcoef(mIoU_clean - mIoU_real, mIoU_clean - miou_estimated)



plt.scatter(loss_real, loss_channels.mean(axis=1))

plt.scatter(noise_real, noise_channels.sum(axis=1))



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
    plt.subplot(4,5, counter)
    counter += 1
    plt.plot([m,M], [m,M], color="black")
    plt.scatter(noise_real_mat[:sample, layer], noise_estimated_mat[:sample, layer], alpha=0.5, s=20, color="black")
    plt.xlim([m,M])
    plt.ylim([m,M])
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_title(f"Layer {layer}", y=1.0, pad=-8, fontsize=8)

    if layer == 24:
        plt.ylabel("Additive Distortion Estimation", fontsize=18)

    if layer == 52:
        plt.xlabel("Real Distortion", fontsize=18)







plt.scatter(loss_clean, loss_real)
len(assets_0)
np.array(assets_0[2]['loss_channels'])[1]