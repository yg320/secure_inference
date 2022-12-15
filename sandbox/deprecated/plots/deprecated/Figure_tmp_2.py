import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pickle
import torch
results_noise = pickle.load(open('/home/yakir/Data2/assets_v3_distortion_to_miou/deformations/coco_stuff164k/ResNetV1c/results_noise.pickle', 'rb'))
results_orig = pickle.load(open('/home/yakir/Data2/assets_v3_distortion_to_miou/deformations/coco_stuff164k/ResNetV1c/results_orig.pickle', 'rb'))
# noises, signals, loss_deform, metric


noises_list = [[] for _ in range(18)]
signals_list = [[] for _ in range(18)]
mIoU_orig_list = []
mIoU_deformed_list = []
loss_orig_list = []
loss_deformed_list = []

for (_, _, loss_orig, mIoU_orig), (noises, signals, loss_deformed, mIoU_deformed) in zip(results_orig, results_noise):
    if mIoU_orig != 0:
        mIoU_orig_list.append(mIoU_orig)
        mIoU_deformed_list.append(mIoU_deformed)
        loss_orig_list.append(loss_orig.cpu().numpy())
        loss_deformed_list.append(loss_deformed.cpu().numpy())
        for i in range(18):
            noises_list[i].append(noises[i])
            signals_list[i].append(signals[i])

noises_list = np.array(noises_list)
signals_list = np.array(signals_list)
mIoU_orig_list = np.array(mIoU_orig_list)
mIoU_deformed_list = np.array(mIoU_deformed_list)
loss_orig_list = np.array(loss_orig_list)
loss_deformed_list = np.array(loss_deformed_list)

import sklearn
plt.subplot(121)
plt.scatter(loss_deformed_list,noises_list[-1], alpha=0.1)
plt.subplot(122)
plt.scatter(loss_deformed_list,noises_list[-2], alpha=0.1)
np.corrcoef(noises_list[-2], loss_deformed_list)


noises_deformation = np.array(noises_deformation)
mIoU_deformation = np.array(mIoU_deformation)


plt.scatter([x[3] for x in results_orig], [x[3] for x in results_noise])
plt.scatter(losses_deformation, noises_deformation[-1])
np.corrcoef(noises_deformation[-3], mIoU_deformation_simple)
np.corrcoef(mIoU_orig_list, mIoU_deformation)
plt.plot([0,1], [0,1], lw=5)
np.corrcoef([x[3] for x in results_orig], [x[3] for x in results_noise])
np.corrcoef(noises_deformation[-2], mIoU_deformation)
plt.scatter(noises_deformation[-2], mIoU_deformation)
plt.xlabel("LoSS")
plt.ylabel("MiOU")
print('hey')
import numpy as np
plt.plot([np.corrcoef(losses_deformation, noises_deformation[i])[0,1] for i in range(18)][8:])
for i in range(18):
    print(i,np.corrcoef(losses_deformation, noises_deformation[i])[0,1])

plt.plot(mIoU_deformation)