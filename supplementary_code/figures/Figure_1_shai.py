import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

steps  = np.array(
         [ 7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 57.34, 114.69, 197, 229.38, 557]) / 557
ours   = [44.21, 58.33, 61.87, 68.08, 70.09, 73.35, 73.97, 77.15, 78.00, 78.12, 78.41]
secure = [00.21, 00.33, 00.87, 00.08, 00.09, 73.53, 00.97, 00.15, 00.00, 00.12, 00.41]
ours_m = [59.37, 63.19, 64.45]
theirs = [62.3 , 64.97, 65.36, 68.41, 68.68, 69.5,  72.68, 74.72,  75.5,  76.22, 74.46]
# 0,
# 16.76,
# 16.76,
# 18.49,
plt.figure(figsize=(12, 12))

# plt.subplot(221)

plt.plot(steps, ours, '.-', color="#3399e6", lw=3, markersize=10, label="Classification, ResNet18, CIFAR100 - Ours")

plt.plot(steps[:3], ours_m, '.--', color="#3399e6", lw=3, markersize=10, label="Classification, ResNet18, CIFAR100 - Ours (modified)")


plt.plot(steps,theirs, '.-', color="#69b3a2", lw=3, markersize=10, label="Classification, ResNet18, CIFAR100 - DeepReDuce")

plt.xlabel("Ratio of DReLUs Used", fontsize=13, labelpad=7)
plt.ylabel("Accuracy", fontsize=13, labelpad=7)

plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))
plt.gca().yaxis.set_major_locator(MultipleLocator(5))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.01))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

# plt.xlim([0, 1])
# plt.ylim([65, 80])
plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend()