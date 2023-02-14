import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#CLS
# 1x1 9608704
# 1x2 4834816
# 2x2 2434816
# 2x3 1701120
# 3x3 1189632
# 3x4 874240
# 4x4 644224

# Segmentation
# 1x1 85262848
# 1x2 42631680
# 2x2 21316096
# 2x3 14580224
# 3x3 9973840
# 3x4 7290368
# 4x4 5329408
# 4x5 4330240
# 5x5 3518416
# 5x6 2968880
# 6x6 2505328
# 6x7 2251616
# 7x7 2024560

plt.figure(figsize=(5, 4))
dReLUs_relative_to_baseline_classification = 100*np.array([644224/9608704, 874240/9608704, 1189632/9608704])
dReLUs_relative_to_baseline_segmentation = 100*np.array([2505328/85262848, 3518416/85262848, 5329408/85262848])
performance_relative_to_baseline_classification = 100 - 100*np.array([71.53/76.98, 72.94/76.98, 74.09/76.98])
performance_relative_to_baseline_segmentation = 100 - 100*np.array([33.05/36.5, 34/36.5, 34.73/36.5])

# plt.subplot(211)
plt.plot(dReLUs_relative_to_baseline_classification, performance_relative_to_baseline_classification, '.-', color="#3399e6", lw=3, markersize=15, label="Classification")
plt.plot(dReLUs_relative_to_baseline_segmentation, performance_relative_to_baseline_segmentation, '.-', color="#69b3a2", lw=3, markersize=15, label="Segmentation")
plt.xlabel("Percentage of DReLU Used", fontsize=13, labelpad=7)
plt.ylabel("Relative Performance Decrease (%)", fontsize=13, labelpad=7)
# assert False, "Add percentage in ticks"
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
plt.gca().tick_params(axis='both', which='major', labelsize=11)

plt.xlim([2, 15])
plt.ylim([0, 10])
plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.tight_layout()
plt.legend()
plt.savefig("/home/yakir/Figure_1.png")






# plt.figure(figsize=(10,8))
# plt.scatter([24.39, 19.72, 17.0, 15.24, 14.33], [99, 95, 90, 85, 82], s=100,  color="#69b3a2")
# # plt.scatter([24.39, 19.72, 17.0, 15.24, 14.33], [99, 95, 90, 85, 82], s=100,  color="#69b3a2")
# plt.xlabel("Percentage of Bandwidth Used (Relative to Baseline)", fontsize=14)
# plt.ylabel("Percentage of mIoU (Relative to Baseline)", fontsize=14)
# plt.xticks(np.arange(10,31,1))
# plt.yticks(np.arange(80,101,1))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.xlim([10,30])
# plt.ylim([80,100])
# plt.bar(["{:.0f}".format(x) for x in 100*np.arange(0,1.05,0.05)], conv_cost, width, color="#69b3a2", label='Fused Conv/Norm Layers')
# plt.bar(["{:.0f}".format(x) for x in 100*np.arange(0,1.05,0.05)], mult_cost, width, color="#3399e6", bottom=conv_cost, label='ReLU (after DReLU)')
# plt.bar(["{:.0f}".format(x) for x in 100*np.arange(0,1.05,0.05)], dReLU_cost, width,color="#c74c52",  bottom=conv_cost + mult_cost, label='DReLU')
# plt.legend(loc="upper left")
# plt.ylabel("Bandwidth Percentage", fontsize=14)
# plt.xlabel("Target DReLU Percentage", fontsize=14)
#
# plt.xticks(20*np.arange(0.0, 1.1, 0.1))
# plt.yticks(100*np.arange(0, 1.05, 0.05))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
