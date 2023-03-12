import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# https://xkcd.com/color/rgb/


segmentation_mmlab_result = 34.08
classification_mmlab_result = 76.55

segmentation_mmlab_runtime = 1154.7 / 3
classification_mmlab_runtime = 84.37 / 3

classification_3x4_24_bits_results = 73.28
classification_4x4_24_bits_results = 71.44
classification_3x4_64_bits_results = 74.36
classification_4x4_64_bits_results = 71.88

classification_2x3_24_bits_runtime = 16.4944 / 3
classification_3x3_24_bits_runtime = 15.5404 / 3
classification_3x4_24_bits_runtime = 14.5939 / 3
classification_4x4_24_bits_runtime = 14.1082 / 3
classification_1x1_64_bits_runtime = 79.8271 / 3
classification_2x3_64_bits_runtime = 22.1703 / 3
classification_3x3_64_bits_runtime = 18.6956 / 3
classification_3x4_64_bits_runtime = 17.4263 / 3
classification_4x4_64_bits_runtime = 15.6674 / 3

segmentation_4x5_24_bits_runtime = 98.89 / 3
segmentation_6x6_24_bits_runtime = 91.75 / 3
segmentation_4x5_32_bits_runtime = 101.6523 / 3
segmentation_5x5_32_bits_runtime = 98.8634 / 3
segmentation_1x1_64_bits_runtime = 1157.7 / 3
segmentation_4x5_64_bits_runtime = 120.96 / 3
segmentation_5x5_64_bits_runtime = 109.42 / 3
segmentation_6x6_64_bits_runtime = 106.21 / 3

############################################################################################################


segmentation_6x6_64_bits_results_not_secure = 33.05
segmentation_4x4_64_bits_results_not_secure = 34.73
segmentation_1x1_64_bits_results_not_secure = 36.5
classification_3x3_to_3x4_performance_ratio_epoch_41 = 73.85 / 72.7
seg_24_to_32 = segmentation_4x5_24_bits_runtime / segmentation_4x5_32_bits_runtime
seg_4x4_to_4x5 = 1.1
runtime_1x1_24_64 = 0.7
segmentation_5x5_64_bits_results_not_secure_mock = 34.02
classification_1x1_64_bits_results_mock = 76.83

############################################################################################################


segmentation_5x5_24_bits_runtime = seg_24_to_32 * segmentation_5x5_32_bits_runtime
segmentation_4x4_64_bits_runtime = seg_4x4_to_4x5 * segmentation_4x5_64_bits_runtime
segmentation_3x3_64_bits_runtime = 47.4
classification_1x1_24_bits_runtime = runtime_1x1_24_64 * classification_1x1_64_bits_runtime
segmentation_3x3_24_bits_runtime = 41.2
segmentation_4x4_24_bits_runtime = 105 / 3
segmentation_1x1_24_bits_runtime = 0.7 * 1154.7 / 3

segmentation_1x1_64_bits_results = segmentation_1x1_64_bits_results_not_secure
segmentation_3x3_64_bits_results = 35.2
segmentation_4x4_64_bits_results = segmentation_4x4_64_bits_results_not_secure
segmentation_5x5_64_bits_results = segmentation_5x5_64_bits_results_not_secure_mock
segmentation_6x6_64_bits_results = segmentation_6x6_64_bits_results_not_secure
classification_3x3_64_bits_results = classification_3x3_to_3x4_performance_ratio_epoch_41 * classification_3x4_64_bits_results
classification_1x1_64_bits_results = classification_1x1_64_bits_results_mock
classification_3x3_24_bits_results = classification_3x3_64_bits_results
classification_2x3_64_bits_results = 76.1
classification_2x3_24_bits_results = 75.9
classification_1x1_24_bits_results = classification_1x1_64_bits_results

segmentation_results_64_bits = 100 * np.array(
    [segmentation_6x6_64_bits_results, segmentation_5x5_64_bits_results, segmentation_4x4_64_bits_results,
     segmentation_3x3_64_bits_results, segmentation_1x1_64_bits_results]) / segmentation_mmlab_result
segmentation_results_24_bits = segmentation_results_64_bits
segmentation_runtime_lan_64_bits = segmentation_mmlab_runtime / np.array(
    [segmentation_6x6_64_bits_runtime, segmentation_5x5_64_bits_runtime, segmentation_4x4_64_bits_runtime,
     segmentation_3x3_64_bits_runtime, segmentation_1x1_64_bits_runtime])
segmentation_runtime_lan_24_bits = segmentation_mmlab_runtime / np.array(
    [segmentation_6x6_24_bits_runtime, segmentation_5x5_24_bits_runtime, segmentation_4x4_24_bits_runtime,
     segmentation_3x3_24_bits_runtime, segmentation_1x1_24_bits_runtime])
segmentation_runtime_wan_64_bits = segmentation_mmlab_runtime / np.array(
    [segmentation_6x6_64_bits_runtime, segmentation_5x5_64_bits_runtime, segmentation_4x4_64_bits_runtime,
     segmentation_3x3_64_bits_runtime, segmentation_1x1_64_bits_runtime])
segmentation_runtime_wan_24_bits = segmentation_mmlab_runtime / np.array(
    [segmentation_6x6_24_bits_runtime, segmentation_5x5_24_bits_runtime, segmentation_4x4_24_bits_runtime,
     segmentation_3x3_24_bits_runtime, segmentation_1x1_24_bits_runtime])

# segmentation_runtime_wan = 66.6/np.array([7.47, 9.14, 66.6])
classification_results_64_bits = 100 * np.array(
    [classification_4x4_64_bits_results, classification_3x4_64_bits_results, classification_3x3_64_bits_results,
     classification_2x3_64_bits_results, classification_1x1_64_bits_results]) / classification_mmlab_result
classification_results_24_bits = 100 * np.array(
    [classification_4x4_24_bits_results, classification_3x4_24_bits_results, classification_3x3_24_bits_results,
     classification_2x3_24_bits_results, classification_1x1_24_bits_results]) / classification_mmlab_result
classification_runtime_lan_64_bits = classification_mmlab_runtime / np.array(
    [classification_4x4_64_bits_runtime, classification_3x4_64_bits_runtime, classification_3x3_64_bits_runtime,
     classification_2x3_64_bits_runtime, classification_1x1_64_bits_runtime])
classification_runtime_lan_24_bits = classification_mmlab_runtime / np.array(
    [classification_4x4_24_bits_runtime, classification_3x4_24_bits_runtime, classification_3x3_24_bits_runtime,
     classification_2x3_24_bits_runtime, classification_1x1_24_bits_runtime])
classification_runtime_wan_64_bits = classification_mmlab_runtime / np.array(
    [classification_4x4_64_bits_runtime, classification_3x4_64_bits_runtime, classification_3x3_64_bits_runtime,
     classification_2x3_64_bits_runtime, classification_1x1_64_bits_runtime])
classification_runtime_wan_24_bits = classification_mmlab_runtime / np.array(
    [classification_4x4_24_bits_runtime, classification_3x4_24_bits_runtime, classification_3x3_24_bits_runtime,
     classification_2x3_24_bits_runtime, classification_1x1_24_bits_runtime])

def foo(x):
    if x is None:
        return None
    if x < 10:
        return str(round(x, 1))
    else:
        return str(round(x))

segmentation_time = [384.90000000000003, 128.3, 76.98, 54.98571428571429, 42.76666666666667, 34.99090909090909]

colors = ["#3399e6", "#69b3a2"]
ticks_font_size = 16
axis_label_font_size = 20
plot_lw = 5
plot_markersize = 15

segmentation_runtime = [31, 32, 35, 41, 269]
classification_runtime = [4.7,  4.8,  5.2,  5.5, 18.6]

plt.figure(figsize=(8, 6))
plt.plot(classification_results_24_bits, classification_runtime_lan_24_bits, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize, label="Accuracy")
plt.plot(segmentation_results_24_bits, segmentation_runtime_lan_24_bits, ".-", color=colors[1], lw=plot_lw, markersize=plot_markersize, label="mIoU")

ax = plt.gca()

for i in range(len(classification_runtime)):
    if i != 2:
        ax.annotate(str(classification_runtime[i]) + "s", (classification_results_24_bits[i]+0.25, classification_runtime_lan_24_bits[i]+0.25), fontsize=14, weight='bold')


for i in range(len(segmentation_runtime)):
    if i != 0:
        ax.annotate(str(segmentation_runtime[i]) + "s", (segmentation_results_24_bits[i]+0.25, segmentation_runtime_lan_24_bits[i]+0.25), fontsize=14, weight='bold')


ax.set_ylabel('Factor in Runtime', fontsize=axis_label_font_size, labelpad=7)
[i.set_linewidth(2) for i in ax.spines.values()]
ax.set_ylim([1, 13.0])
ax.set_xlim([93, 109])
ax.set_yticklabels([None, "x2", "x4", "x6", "x8", "x10", "x12"], fontsize=ticks_font_size)
ax.set_xticklabels([None] + [str(x) + "%" for x in np.arange(94, 109, 2)], fontsize=ticks_font_size)
ax.set_xlabel('Performance Relative to Baseline', fontsize=axis_label_font_size, labelpad=22)
ax.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5, lw=2, axis='x')
plt.legend(prop={'size': 16})

plt.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.16)

plt.savefig("/home/yakir/Figure_6.png")
