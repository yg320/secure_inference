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
colors = ["#3399e6", "#69b3a2"]
ticks_font_size = 12
axis_label_font_size = 14
plot_lw = 3
plot_markersize = 10
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
# fig.subplots_adjust()

ax1 = axes[0]
[i.set_linewidth(1.5) for i in ax1.spines.values()]

ax1.set_ylabel('Factor in Runtime', fontsize=axis_label_font_size, labelpad=7)
ax1.plot(classification_results_24_bits, classification_runtime_lan_24_bits, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)
# ax1.invert_yaxis()
ax1.set_ylim([0, 7.0])
ax1.set_yticklabels([None, "x1", "x2", "x3", "x4", "x5", "x6"], fontsize=ticks_font_size)

ax1.set_xlim([93, 101])
ax1.set_xticks(np.arange(93, 101, 1))
ax1.set_xticklabels([str(x) + "%" for x in np.arange(93, 101, 1)], fontsize=ticks_font_size)

ax1.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5, lw=2, axis='x')

ax1.set_xlabel('Accuracy Relative to Baseline', fontsize=axis_label_font_size, labelpad=7)
ax2 = ax1.twinx()
ax2.plot(classification_results_24_bits, classification_runtime_lan_24_bits, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)
ax2.set_ylim([0, 7.0])
ax2.set_ylabel("Time (seconds)", fontsize=axis_label_font_size, labelpad=7)
ax2.set_yticklabels(map(foo, [None] + [classification_mmlab_runtime/x for x in range(1,7)]), fontsize=ticks_font_size)


ax1 = axes[1]
[i.set_linewidth(1.5) for i in ax1.spines.values()]

ax1.set_ylabel('Factor in Runtime', fontsize=axis_label_font_size, labelpad=-0.1)

ax1.plot(segmentation_results_24_bits, segmentation_runtime_lan_24_bits, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)
ax1.set_ylim([0, 13])
ax1.set_yticklabels([None, "x1", "x3", "x5", "x7", "x9", "x11"], fontsize=ticks_font_size)


ax1.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5, lw=2, axis='x')

ax1.set_xticks(np.arange(96, 108, 2), fontsize=ticks_font_size)
ax1.set_xticklabels([str(x) + "%" for x in np.arange(96, 108, 2)], fontsize=ticks_font_size)

ax1.set_xlim([96, 108])

ax1.set_xlabel('mIoU Relative to Baseline', fontsize=axis_label_font_size, labelpad=7)

ax2 = ax1.twinx()
ax2.plot(segmentation_results_24_bits, segmentation_runtime_lan_24_bits, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)
ax2.set_ylim([0, 13])
ax2.set_ylabel("Time (seconds)", fontsize=axis_label_font_size, labelpad=7)
ax2.set_yticklabels(map(foo, [None] + [segmentation_mmlab_runtime/x for x in [1, 3, 5, 7, 9, 11]]), fontsize=ticks_font_size)



# plt.tight_layout()
plt.subplots_adjust(left=0.11, right=0.88, top=0.98, bottom=0.07, hspace=0.35)#, right=0.99, top=0.98, bottom=0.1, hspace=0.02, wspace=0.15)
#
plt.savefig("/home/yakir/Figure_6.png")
