import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


plt.figure(figsize=(10, 6))
plt.subplot(121)

ticks_font_size = 20
label_font_size = 22
resnet_coco_baseline = 76.55
deeplab = 34.08

dReLUs_relative_to_baseline_classification = 100 * np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18])
dReLUs_relative_to_baseline_segmentation = 100 * np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18])
performance_relative_to_baseline_classification = 100 * np.array([65.59, 70.36, 72.23, 73.30, 74.03, 74.40]) / resnet_coco_baseline
performance_relative_to_baseline_segmentation = 100 * np.array([33.23, 34.73, 35.53, 35.77, 36.01, 36.31]) / deeplab

# plt.subplot(211)
plt.plot(dReLUs_relative_to_baseline_classification, performance_relative_to_baseline_classification, '.-', color="#3399e6", lw=5, markersize=15, label="Classification, ImageNet")
plt.plot(dReLUs_relative_to_baseline_segmentation, performance_relative_to_baseline_segmentation, '.-', color="#69b3a2", lw=5, markersize=15, label="DeepLabV3, ADE_20K")
plt.xlabel("Percentage of DReLU", fontsize=label_font_size, labelpad=10)
plt.ylabel("MMLab's Relative Performance", fontsize=label_font_size, labelpad=7)

# assert False, "Add percentage in ticks"
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.gca().yaxis.set_major_locator(MultipleLocator(5))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().tick_params(axis='both', which='major', labelsize=11)

plt.xlim([2, 20])
plt.ylim([85, 110])
plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend(loc="lower right", prop={'size': 16})
plt.gca().set_yticklabels([None, "85%", "90%", "95%", "100%", "105%", "110%"], fontsize=ticks_font_size)
plt.gca().set_xticklabels([None, "5%", "10%", "15%", None], fontsize=ticks_font_size)

plt.subplot(122)

plt.plot([0,      7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 57.34, 114.69, 197, 229.38, 557],
         [16.76, 59.37, 63.19, 61.87, 68.08, 70.09, 73.35, 73.97, 77.15, 78.00, 78.12, 78.41], '.-', color="#3399e6", lw=5, markersize=10, label="Ours")


plt.plot([0,     7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 57.34, 114.69, 197,  229.38, 557],
         [18.49, 62.3, 64.97, 65.36, 68.41, 68.68, 69.5,  72.68, 74.72,  75.5,  76.22, 74.46], '.-', color="#69b3a2", lw=5, markersize=10, label="DeepReDuce")

# 63.1900,
plt.xlabel("# DReLUs (in K)", fontsize=label_font_size, labelpad=10)
plt.ylabel("Accuracy", fontsize=label_font_size, labelpad=5)

plt.gca().xaxis.set_major_locator(MultipleLocator(200))
plt.gca().yaxis.set_major_locator(MultipleLocator(5))
plt.gca().xaxis.set_minor_locator(MultipleLocator(50))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().set_xticklabels([None, None, "200", "400", "600"], fontsize=ticks_font_size)
plt.gca().set_yticklabels([None, "60", "65", "70", "75", "80"], fontsize=ticks_font_size)
plt.xlim([0, 600])
plt.ylim([58, 80])
plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend(loc="lower right", prop={'size': 16})
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")

plt.subplots_adjust(left=0.14, right=0.9, top=0.95, bottom=0.15, wspace=0.1)#, right=0.88, top=0.98, bottom=0.07, hspace=0.35)#, right=0.99, top=0.98, bottom=0.1, hspace=0.02, wspace=0.15)

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
