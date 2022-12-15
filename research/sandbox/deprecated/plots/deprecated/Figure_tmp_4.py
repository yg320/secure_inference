import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import numpy as np
layer_name = "layer3_1_1"
d = pickle.load(open(f"/home/yakir/Data2/assets_v3/additive_deformation_estimation/coco_stuff164k/ResNetV1c/data_{layer_name}.pickle", 'rb'))
d = {key:np.array(value) for key, value in d.items()}
d.keys()
noise_corr = np.zeros((18,18))
for i in range(18):
    for j in range(18):
        noise_corr[i,j] = np.corrcoef(d['noises'][:,i], d['noises'][:,j])[0,1]


plt.suptitle(layer_name)
plt.subplot(131)
plt.imshow(noise_corr)
plt.colorbar()
plt.subplot(132)
plt.scatter(d['noises'][:,-7], d['miou']/d['miou_baseline'], alpha=0.1)
plt.xlabel("Distortion (Measured in the Last Layer)")
plt.ylabel("mIoU/mIoU_baseline")
plt.subplot(133)
plt.scatter(d['noises'][:,-7], d['loss']/d['loss_baseline'], alpha=0.1)
plt.xlabel("Distortion (Measured in the Last Layer)")
plt.ylabel("loss/loss_baseline")



miou_noise = np.array([x[-1] for x in results_noise])
miou_orig = np.array([y[-1] for y in results_orig])
loss_noise = np.array([x[-2].cpu().numpy() for x in results_noise])
loss_orig = np.array([y[-2].cpu().numpy() for y in results_orig])

distortion_noises = np.array([x[0][-1] for x in results_noise])
distortion_signal = np.array([x[1][-1] for x in results_noise])

miou_noise = miou_noise[miou_orig > 0]
distortion_noises = distortion_noises[miou_orig > 0]
distortion_signal = distortion_signal[miou_orig > 0]
loss_noise = loss_noise[miou_orig > 0]
loss_orig = loss_orig[miou_orig > 0]
miou_orig = miou_orig[miou_orig > 0]

np.corrcoef(distortion_noises, loss_noise)
plt.scatter(distortion_noises, miou_noise)

plt.scatter(loss_noise/loss_orig, distortion_noises/distortion_signal)