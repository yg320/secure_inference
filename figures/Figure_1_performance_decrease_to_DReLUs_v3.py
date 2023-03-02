import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


plt.figure(figsize=(10, 6))
# plt.subplot(121)
green = "#56ae57"
red = "#db5856"
purple = "tab:purple"
blue = "#3399e6"
ticks_font_size = 20
label_font_size = 24

ImageNet_baseline = 76.55
ADE20K_baseline = 34.08
VOC2012_baseline = 77.68
CIFAR100_baseline = 78.27
lw = 3
dReLUs_relative_to_baseline = 100 * np.array([0.03, 0.06, 0.09, 0.12, 0.15, 0.18])
performance_relative_to_baseline_ImageNet = 100 * np.array([65.59, 70.36, 72.23, 73.30, 74.03, 74.40]) / ImageNet_baseline
performance_relative_to_baseline_ADE20K = 100 * np.array([33.23, 34.73, 35.53, 35.77, 36.01, 36.31]) / ADE20K_baseline
performance_relative_to_baseline_VOC = 100 * np.array([72.23, 74.27, 75.89, 76.26, 76.86, 77.48]) / VOC2012_baseline
# performance_relative_to_baseline_CIFAR100 = 100 * np.array([62.90, 70.90, 74.62, 74.53, 76.18, 76.86]) / CIFAR100_baseline
performance_relative_to_baseline_CIFAR100 = 100 * np.array([65.63, 70.90, 74.62, 74.53, 76.18, 76.86]) / CIFAR100_baseline
# plt.subplot(211)

# plt.plot(100*np.array([0,      7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 57.34, 114.69, 197, 229.38, 557])/557,
#          100*np.array([16.76, 59.37, 63.19, 61.87, 68.08, 70.09, 73.35, 73.97, 77.15, 78.00, 78.12, 78.41])/78, '.-', color="tab:red", lw=7, markersize=25, label="ResNet18, COCO100 - Ours")

# plt.plot(100*np.array([0,      7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 114.69, 197])/557,
#          100*np.array([16.76, 59.29, 63.05, 64.16, 67.58, 68.99, 72.44, 76.60, 77.42])/77.75, '.-', color="tab:red", lw=lw, markersize=25, label="ResNet18, COCO100 - Ours")

# plt.plot(100*np.array([14.33, 16.71, 24.57, 28.67, 33.42, 49.15, 50.13, 57.34, 114.69, 197])/557,
#          100*np.array([61.11, 62.90, 68.39, 69.73, 70.90, 74.73, 74.62, 74.52, 76.94, 77.84])/78.27, '.-', color="tab:red", lw=lw, markersize=25, label="ResNet18, COCO100 - Ours")
#

plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_ADE20K, '.-', color=green, lw=lw, markersize=10, label="ADE20K - Seg. - Ours")
plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_VOC, '.-', color=purple, lw=lw, markersize=10, label=  "VOC2012 - Seg. - Ours")
plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_ImageNet, '.-', color=blue, lw=lw, markersize=10, label="ImageNet - Cls. - Ours")
plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_CIFAR100, '.-', color=red, lw=lw, markersize=10, label="CIFAR100 - Cls. - Ours")

plt.plot(100*np.array([7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 57.34, 114.69, 197,  229.38, 557])/557,
         100*np.array([62.3, 64.97, 65.36, 68.41, 68.68, 69.5,  72.68, 74.72,  75.5,  76.22, 74.46])/CIFAR100_baseline, '.:', color="#d3494e", lw=lw, markersize=10, label="CIFAR100 - Cls. - DeepReDuce")

plt.plot(100*np.array([7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 57.34, 114.69, 197,  229.38, 557])/557,
         100*np.array([62.3, 64.97, 65.36, 68.41, 68.68, 69.5,  72.68, 74.72,  75.5,  76.22, 74.46])/CIFAR100_baseline, '.:', color="#d3494e", lw=lw, markersize=25)

plt.plot(dReLUs_relative_to_baseline[1:2], performance_relative_to_baseline_ImageNet[1:2], '.-', color=blue, lw=lw, markersize=20)
plt.plot(dReLUs_relative_to_baseline[1:2], performance_relative_to_baseline_ADE20K[1:2], '.-', color=green, lw=lw, markersize=20)
plt.plot(dReLUs_relative_to_baseline[1:2], performance_relative_to_baseline_VOC[1:2], '.-', color=purple, lw=lw, markersize=20)
plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_CIFAR100, '.-', color=red, lw=lw, markersize=20)

plt.xlabel("Percentage of DReLU Used", fontsize=label_font_size, labelpad=20)
plt.ylabel("Perf. Relative to Baseline", fontsize=label_font_size, labelpad=20)

# assert False, "Add percentage in ticks"
plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.gca().yaxis.set_major_locator(MultipleLocator(5))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().tick_params(axis='both', which='major', labelsize=11)

plt.xlim([2.5, 18.5])
plt.ylim([80, 108])
plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.9)
plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.5)
plt.legend(loc="lower right", prop={'size': 16})
# plt.savefig("/home/yakir/Figure_1.png")

plt.gca().set_yticklabels([None,   "80%", "85%", "90%", "95%", "100%", "105%", "110%"], fontsize=ticks_font_size)
plt.gca().set_xticklabels([None, "5%", "10%", "15%", "20%",  "25%", "30%","35%",None], fontsize=ticks_font_size)
plt.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.17, wspace=0.1)
[i.set_linewidth(2.) for i in plt.gca().spines.values()]

plt.savefig("/home/yakir/Figure_1.png")
#
#
# # 63.1900,
# plt.xlabel("# DReLUs (in K)", fontsize=label_font_size, labelpad=10)
# plt.ylabel("Accuracy", fontsize=label_font_size, labelpad=5)
#
# plt.gca().xaxis.set_major_locator(MultipleLocator(200))
# plt.gca().yaxis.set_major_locator(MultipleLocator(5))
# plt.gca().xaxis.set_minor_locator(MultipleLocator(50))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
# plt.gca().set_xticklabels([None, None, "200", "400", "600"], fontsize=ticks_font_size)
# plt.gca().set_yticklabels([None, "60", "65", "70", "75", "80"], fontsize=ticks_font_size)
# plt.xlim([0, 600])
# plt.ylim([58, 80])
# plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
# plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.legend(loc="lower right", prop={'size': 16})
# plt.gca().yaxis.tick_right()
# plt.gca().yaxis.set_label_position("right")
#
# plt.subplots_adjust(left=0.14, right=0.9, top=0.95, bottom=0.15, wspace=0.1)#, right=0.88, top=0.98, bottom=0.07, hspace=0.35)#, right=0.99, top=0.98, bottom=0.1, hspace=0.02, wspace=0.15)







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
