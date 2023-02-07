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
classification_1x1_24_bits_runtime = runtime_1x1_24_64 * classification_1x1_64_bits_runtime
segmentation_4x4_24_bits_runtime = 105/3
segmentation_1x1_24_bits_runtime = 0.7*1154.7/3

segmentation_1x1_64_bits_results = segmentation_1x1_64_bits_results_not_secure
segmentation_4x4_64_bits_results = segmentation_4x4_64_bits_results_not_secure
segmentation_5x5_64_bits_results = segmentation_5x5_64_bits_results_not_secure_mock
segmentation_6x6_64_bits_results = segmentation_6x6_64_bits_results_not_secure
classification_3x3_64_bits_results = classification_3x3_to_3x4_performance_ratio_epoch_41 * classification_3x4_64_bits_results
classification_1x1_64_bits_results = classification_1x1_64_bits_results_mock
classification_3x3_24_bits_results = classification_3x3_64_bits_results
classification_2x3_64_bits_results = 76.1
classification_2x3_24_bits_results = 75.9
classification_1x1_24_bits_results = classification_1x1_64_bits_results

segmentation_results_64_bits = 100 * np.array([segmentation_6x6_64_bits_results, segmentation_5x5_64_bits_results, segmentation_4x4_64_bits_results, segmentation_3x3_64_bits_results, segmentation_1x1_64_bits_results]) / segmentation_mmlab_result
segmentation_results_24_bits =  segmentation_results_64_bits
segmentation_runtime_lan_64_bits = segmentation_mmlab_runtime / np.array([segmentation_6x6_64_bits_runtime, segmentation_5x5_64_bits_runtime, segmentation_4x4_64_bits_runtime, segmentation_3x3_64_bits_runtime, segmentation_1x1_64_bits_runtime])
segmentation_runtime_lan_24_bits = segmentation_mmlab_runtime / np.array([segmentation_6x6_24_bits_runtime, segmentation_5x5_24_bits_runtime, segmentation_4x4_24_bits_runtime, segmentation_3x3_24_bits_runtime, segmentation_1x1_24_bits_runtime])

# segmentation_runtime_wan = 66.6/np.array([7.47, 9.14, 66.6])
classification_results_64_bits = 100 * np.array([classification_4x4_64_bits_results, classification_3x4_64_bits_results, classification_3x3_64_bits_results, classification_2x3_64_bits_results, classification_1x1_64_bits_results]) / classification_mmlab_result
classification_results_24_bits = 100 * np.array([classification_4x4_24_bits_results, classification_3x4_24_bits_results, classification_3x3_24_bits_results, classification_2x3_24_bits_results, classification_1x1_24_bits_results]) / classification_mmlab_result
classification_runtime_lan_64_bits = classification_mmlab_runtime / np.array([classification_4x4_64_bits_runtime, classification_3x4_64_bits_runtime, classification_3x3_64_bits_runtime, classification_2x3_64_bits_runtime, classification_1x1_64_bits_runtime])
classification_runtime_lan_24_bits = classification_mmlab_runtime / np.array([classification_4x4_24_bits_runtime, classification_3x4_24_bits_runtime, classification_3x3_24_bits_runtime, classification_2x3_24_bits_runtime, classification_1x1_24_bits_runtime])

colors = ["#3399e6", "#69b3a2"]
ticks_font_size = 12
axis_label_font_size = 12
plot_lw = 3
plot_markersize = 10
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5), constrained_layout=True)
# fig.subplots_adjust()
if True:
    if True:
        ax1 = axes[0][0]
        [i.set_linewidth(1.5) for i in ax1.spines.values()]
        ax1.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)

        ax1.set_ylabel('LAN (s)', fontsize=axis_label_font_size, labelpad=7)
        ax1.plot(classification_results_64_bits, classification_runtime_lan_64_bits, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)
        ax1.plot(classification_results_24_bits, classification_runtime_lan_24_bits, ".-", color=colors[1], lw=plot_lw, markersize=plot_markersize)

        ax1.set_ylim([0.0, 7])
        ax1.set_yticklabels(["", "28.1", "14.0", "9.3", "7.0", "5.6", "4.6"], fontsize=ticks_font_size)

        ax1.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
        ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5, lw=2, axis='x')

        ax1.set_xticks(np.arange(93,101,1))

        ax1.set_xlim([93, 101])
        for i in range(1, 7):
            ax1.text(100.75, i+0.2, f'x{i}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10, color='black',  fontweight='bold')

    if False:
        ax1 = axes[1][0]
        [i.set_linewidth(1.5) for i in ax1.spines.values()]
        ax1.set_ylabel('WAN (s)', fontsize=axis_label_font_size, labelpad=7)
        ax1.plot(segmentation_results_64_bits, segmentation_runtime_lan_64_bits, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)
        ax1.plot(segmentation_results_24_bits, segmentation_runtime_lan_24_bits, ".-", color=colors[1], lw=plot_lw, markersize=plot_markersize)

        ax1.set_ylim([0.0, 7])
        ax1.set_yticklabels(["", "421", "210", "140", "105", "84", "70"],fontsize=ticks_font_size)

        ax1.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
        ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5, lw=2, axis='x')

        ax1.set_xticklabels(labels = [None, 94, 95, 96, 97, 98, 99, 100],fontsize=ticks_font_size)

        ax1.set_xticks(np.arange(93,101,1))

        ax1.set_xlim([93, 101])
        for i in range(1, 7):

            ax1.text(100.75, i+0.2, f'x{i}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10, color='black',  fontweight='bold')

        ax1.set_xlabel('Accuracy Relative to Baseline', fontsize=axis_label_font_size, labelpad=7)


if True:
    if True:
        ax1 = axes[0][1]
        [i.set_linewidth(1.5) for i in ax1.spines.values()]
        ax1.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

        ax1.plot(segmentation_results_64_bits, segmentation_runtime_lan_64_bits, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)
        ax1.plot(segmentation_results_24_bits, segmentation_runtime_lan_24_bits, ".-", color=colors[1], lw=plot_lw, markersize=plot_markersize)


        ax1.set_ylim([0.0, 13])
        ax1.set_yticks([1,3,5,7,9, 11])
        ax1.set_yticklabels(["1", "3", "5", "7", "9", "11"], fontsize=ticks_font_size)

        ax1.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
        ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5, lw=2, axis='x')

        ax1.set_xticks(np.arange(96, 108, 2))

        ax1.set_xlim([96, 108])
        for i in [1,3,5,7,9, 11]:
            ax1.text(107.5, i + 0.35, f'x{i}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10, color='black', fontweight='bold')

    if False:
        ax1 = axes[1][1]
        [i.set_linewidth(1.5) for i in ax1.spines.values()]

        ax1.plot(segmentation_results, segmentation_runtime_wan, ".-", color=colors[0], lw=plot_lw,
                 markersize=plot_markersize)

        ax1.set_ylim([0.0, 13])
        ax1.set_yticks([1,3,5,7,9, 11])
        ax1.set_yticklabels(["1", "3", "5", "7", "9", "11"], fontsize=ticks_font_size)

        ax1.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
        ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5, lw=2, axis='x')

        ax1.set_xticks(np.arange(96, 108, 2))

        ax1.set_xlim([96, 108])
        for i in [1,3,5,7,9, 11]:
            ax1.text(107.5, i + 0.35, f'x{i}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10, color='black', fontweight='bold')
        ax1.set_xlabel('mIoU Relative to Baseline', fontsize=axis_label_font_size, labelpad=7)
        ax1.set_xticklabels(labels = [96, 98, 100, 102, 104, 106],fontsize=ticks_font_size)

plt.subplots_adjust(left=0.075, right=0.99, top=0.98, bottom=0.1, hspace=0.02, wspace=0.15)

plt.savefig("/home/yakir/Figure_4.png")
