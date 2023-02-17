import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


plt.figure(figsize=(5, 6.5))
plt.subplot(211)
dReLUs_relative_to_baseline_classification = 100*np.array([644224/9608704, 874240/9608704, 1189632/9608704])
dReLUs_relative_to_baseline_segmentation = 100*np.array([2505328/85262848, 3518416/85262848, 5329408/85262848])
performance_relative_to_baseline_classification = 100 - 100*np.array([71.53/76.98, 72.94/76.98, 74.09/76.98])
performance_relative_to_baseline_segmentation = 100 - 100*np.array([33.05/36.5, 34/36.5, 34.73/36.5])

# plt.subplot(211)
plt.plot(dReLUs_relative_to_baseline_classification, performance_relative_to_baseline_classification, '.-', color="#3399e6", lw=3, markersize=15, label="Classification, ResNet50, ImageNet")
plt.plot(dReLUs_relative_to_baseline_segmentation, performance_relative_to_baseline_segmentation, '.-', color="#69b3a2", lw=3, markersize=15, label="Segmentation, DeepLabV3, MobileNetV2, COCO")
plt.xlabel("Percentage of DReLU Used", fontsize=13, labelpad=7)
plt.ylabel("Performance Decrease (%)", fontsize=13, labelpad=7)
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
plt.legend()

plt.subplot(212)
plt.plot([0,      7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 57.34, 114.69, 197, 229.38, 557],
         [16.76, 44.21, 58.33, 61.87, 68.08, 70.09, 73.35, 73.97, 77.15, 78.00, 78.12, 78.41], '.-', color="#3399e6", lw=3, markersize=10, label="Classification, ResNet18, CIFAR100 - Ours")

plt.plot([0,      7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 57.34, 114.69, 197, 229.38, 557],
         [16.76, 59.37, 63.19, 61.87, 68.08, 70.09, 73.35, 73.97, 77.15, 78.00, 78.12, 78.41], '.--', color="#3399e6", lw=3, markersize=10, label="Classification, ResNet18, CIFAR100 - Ours")


plt.plot([0,     7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 57.34, 114.69, 197,  229.38, 557],
         [18.49, 62.3, 64.97, 65.36, 68.41, 68.68, 69.5,  72.68, 74.72,  75.5,  76.22, 74.46], '.-', color="#69b3a2", lw=3, markersize=10, label="Classification, ResNet18, CIFAR100 - DeepReDuce")

# 63.1900,
plt.xlabel("Number of DReLUs (in K)", fontsize=13, labelpad=7)
plt.ylabel("Accuracy", fontsize=13, labelpad=7)

plt.gca().xaxis.set_major_locator(MultipleLocator(100))
plt.gca().yaxis.set_major_locator(MultipleLocator(5))
plt.gca().xaxis.set_minor_locator(MultipleLocator(20))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

plt.xlim([0, 600])
plt.ylim([58, 80])
plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend()


plt.subplots_adjust(left=0.13, right=0.96, top=0.98, bottom=0.09, hspace=0.4)#, right=0.88, top=0.98, bottom=0.07, hspace=0.35)#, right=0.99, top=0.98, bottom=0.1, hspace=0.02, wspace=0.15)

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
