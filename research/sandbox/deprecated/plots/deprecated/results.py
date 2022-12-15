import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np

exp_names = ["1x1", "3x4", "Algo"]
mIoU = np.array([0.3413, 0.2714, 0.3089])
bandwidth = np.array([11.95, 2.22, 1.93])
mIoU_red = mIoU / mIoU[0]
bandwidth_red = bandwidth / bandwidth[0]
plt.subplot(121)
bar = plt.bar(exp_names, mIoU)
plt.title("mIoU", fontsize=18)
plt.ylabel("mIoU", fontsize=14)

plt.xlabel("Experiment", fontsize=14)
for index, rect in enumerate(bar):
    if index == 0:
        continue
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'mIoU\n relative to baseline\n {100*mIoU_red[index]:.1f} %', ha='center', va='bottom')


plt.subplot(122)
bar = plt.bar(exp_names, bandwidth)
plt.title("Bandwidth (GB)", fontsize=18)

plt.ylabel("Bandwidth (GB)", fontsize=14)
plt.xlabel("Experiment", fontsize=14)
for index, rect in enumerate(bar):
    if index == 0:
        continue
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'Bandwidth \nrelative to baseline\n{100*bandwidth_red[index]:.1f} %', ha='center', va='bottom')
