import collections

import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import glob
import collections
import os
import numpy as np

keys = ['batch_noises_distorted', 'batch_signals_distorted', 'batch_loss_distorted', 'batch_mIoU_distorted',
        'batch_loss_baseline', 'batch_mIoU_baseline']

layers = ['conv1', 'layer2_1_0', 'layer3_1_0', 'layer4_0_0', 'layer4_2_0',  'layer5_0_0',  'layer5_2_0',  'layer6_1_0',  'layer7_0_0', 'decode_2']
#
content_agg = {layer:{128: {k: [] for k in keys}, 512: {k: [] for k in keys}} for layer in layers}
samples = 100
for layer_name in layers:
    dir_name = f"/home/yakir/distortion_approximation/get_approximation_sample_estimation/{layer_name}/"

    for seed in range(200):
        if os.path.exists(os.path.join(dir_name, f"{seed}_128.pickle")) and os.path.exists(os.path.join(dir_name, f"{seed}_512.pickle")):
            for im_size in [128, 512]:
                file_name = os.path.join(dir_name, f"{seed}_{im_size}.pickle")

                content = pickle.load(open(file_name, 'rb'))
                for k in keys:
                    content_agg[layer_name][im_size][k].append(content[k])

    for im_size in [128, 512]:
        for k in keys:
            content_agg[layer_name][im_size][k] = np.array(content_agg[layer_name][im_size][k])

plt.subplot(121)
for layer in layers:
    x = np.log(content_agg[layer][128]['batch_signals_distorted'][:,:64].mean(axis=1)[:, -1] / content_agg[layer][128]['batch_noises_distorted'][:,:64].mean(axis=1)[:, -1])
    y = np.log(content_agg[layer][512]['batch_signals_distorted'].mean(axis=1)[:, -1] / content_agg[layer][512]['batch_noises_distorted'].mean(axis=1)[:, -1])
    y = content_agg[layer][512]['batch_loss_distorted'].mean(axis=1) / content_agg[layer][512]['batch_loss_baseline'].mean(axis=1)
    print(x.shape)
    print(y.shape)
    plt.scatter(x, y, label=layer, alpha=0.5)
plt.legend()
plt.subplot(122)
for layer in layers:
    plt.scatter(
        np.log(content_agg[layer][512]['batch_signals_distorted'][:, :].mean(axis=1)[:, -1] / content_agg[layer][512][
                                                                                            'batch_noises_distorted'][
                                                                                        :, :].mean(axis=1)[:, -1]),
        content_agg[layer][512]['batch_loss_distorted'].mean(axis=1) / content_agg[layer][512]['batch_loss_baseline'].mean(axis=1),
                )

all_losses = np.array(all_losses)
all_noises = np.array(all_noises)
all_noises = all_noises[~np.isnan(all_losses)]
all_losses = all_losses[~np.isnan(all_losses)]
plt.scatter(all_losses, all_noises, alpha=0.02)
plt.scatter(content_agg[512]['batch_signals_distorted'].mean(axis=1)[:,-1]/content_agg[512]['batch_noises_distorted'].mean(axis=1)[:,-1],
            content_agg[512]['batch_loss_distorted'].mean(axis=1)/content_agg[512]['batch_loss_baseline'].mean(axis=1))

plt.scatter(
    content_agg[512]['batch_loss_distorted'].mean(axis=1) / content_agg[512]['batch_loss_baseline'].mean(axis=1),
    content_agg[128]['batch_loss_distorted'].mean(axis=1) / content_agg[128]['batch_loss_baseline'].mean(axis=1))

print(np.corrcoef(
    content_agg[512]['batch_loss_distorted'].mean(axis=1)/ content_agg[512]['batch_loss_baseline'].mean(axis=1) ,
    content_agg[128]['batch_loss_distorted'].mean(axis=1)/ content_agg[128]['batch_loss_baseline'].mean(axis=1) ))

print(np.corrcoef(
    content_agg[512]['batch_loss_distorted'].mean(axis=1) / content_agg[512]['batch_loss_baseline'].mean(axis=1),
    content_agg[128]['batch_signals_distorted'].mean(axis=1)[:, -1] / content_agg[128]['batch_noises_distorted'].mean(
        axis=1)[:, -1]))

plt.scatter(
    content_agg[512]['batch_loss_distorted'].mean(axis=1) / content_agg[512]['batch_loss_baseline'].mean(axis=1),
    content_agg[128]['batch_signals_distorted'].mean(axis=1)[:, -1] / content_agg[128]['batch_noises_distorted'].mean(
        axis=1)[:, -1])