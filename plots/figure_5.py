import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# 1x1 9608704
# 1x2 4834816
# 2x2 2434816
# 2x3 1701120
# 3x3 1189632
# 3x4 874240
# 4x4 644224

plt.figure(figsize=(5, 4))
dReLUs_relative_to_baseline = 100*np.array([644224/9608704, 874240/9608704, 1189632/9608704, 1701120/9608704])
performance_relative_to_baseline_classification = 100*np.array([0.934, 0.952, 0.967, 0.99])
performance_relative_to_baseline_segmentation = 100*np.array([0.91, 0.92, 0.93, 0.95])

# plt.subplot(211)
plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_classification, '.-', color="#3399e6", lw=3, markersize=15)
plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_segmentation, '.-', color="#69b3a2", lw=3, markersize=15)
plt.xlabel("Relative DReLU Count (%)", fontsize=13, labelpad=7)
plt.ylabel("Relative Decrease in Performance (%)", fontsize=13, labelpad=7)

plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
plt.gca().tick_params(axis='both', which='major', labelsize=11)

plt.xlim([5, 20])
plt.ylim([90, 100])
plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.tight_layout()
plt.savefig("/home/yakir/figure_5.png")






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
