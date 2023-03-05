import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

green = "#56ae57"
red = "#db5856"
purple = "tab:purple"
blue = "#3399e6"

results_cifar_baseline = 78.27
results_imnet_baseline = 76.55
results_ade20_baseline = 34.08
results_voc12_baseline = 77.68

runtime_cifar_baseline = 1357235088 * (4/11) / 1000000000
runtime_imnet_baseline = 17478334016 * (4/11) / 1000000000
runtime_ade20_baseline = 64005736576 / 1000000000
runtime_voc12_baseline = 106205953664 / 1000000000

runtime_cifar_16 = np.array([566001920, 578446528, 589733760, 601267480, 612290712, 623566328]) * (4/11) / 1000000000
runtime_cifar_64 = np.array([579236240, 604916752, 629439888, 654208720, 678467856, 702979376]) * (4/11)  / 1000000000
runtime_imnet_16 = np.array([2341120540, 2544648634, 2741103970, 2934551686, 3126430912, 3318121840]) * (4/11)  / 1000000000
runtime_imnet_64 = np.array([2569422460, 3001253266, 3426011314, 3847761742, 4267943680, 4687915408]) * (4/11)  / 1000000000
runtime_ade20_32 = np.array([6341935552, 7542829312, 8724150656, 9879281600, 11035320000, 12158112960]) / 1000000000
runtime_ade20_64 = np.array([7004879488, 8858244736, 10692626048, 12500557952, 14310337152, 16085475456]) / 1000000000
runtime_voc12_32 = np.array([12305108608, 14874560576, 17276690368, 19699197824, 22078597056, 24376760064]) / 1000000000
runtime_voc12_64 = np.array([13765095040, 17783405696, 21574989952, 25407395456, 29186553984, 32841516672]) / 1000000000

perf_imnet = 100 * np.array([65.59, 70.36, 72.23, 73.30, 74.03, 74.40]) / results_imnet_baseline
perf_ade20 = 100 * np.array([33.23, 34.73, 35.53, 35.77, 36.01, 36.31]) / results_ade20_baseline
perf_voc12 = 100 * np.array([72.23, 74.27, 75.89, 76.26, 76.86, 77.48]) / results_voc12_baseline
perf_cifar = 100 * np.array([65.63, 70.90, 74.62, 74.53, 76.18, 76.86]) / results_cifar_baseline

boost_imnet = runtime_imnet_baseline / runtime_imnet_16
boost_cifar = runtime_cifar_baseline / runtime_cifar_16
boost_ade20 = runtime_ade20_baseline / runtime_ade20_32
boost_voc12 = runtime_voc12_baseline / runtime_voc12_32

ticks_font_size = 16
axis_label_font_size = 20
plot_lw = 5
plot_markersize = 15

plt.figure(figsize=(8, 6))
plt.plot(perf_ade20, boost_ade20, ".-", color=green, lw=plot_lw, markersize=plot_markersize, label="ADE20K - Seg.")
plt.plot(perf_voc12, boost_voc12, ".-", color=purple, lw=plot_lw, markersize=plot_markersize, label="VOC12 - Seg.")
plt.plot(perf_imnet, boost_imnet, ".-", color=blue, lw=plot_lw, markersize=plot_markersize, label="ImageNet - Cls.")
plt.plot(perf_cifar, boost_cifar, ".-", color=red, lw=plot_lw, markersize=plot_markersize, label="CIFAR100 - Cls.")

ax = plt.gca()

# for i in range(len(classification_runtime)):
#     if i != 2:
#         ax.annotate(str(classification_runtime[i]) + "s", (classification_results_24_bits[i]+0.25, classification_runtime_lan_24_bits[i]+0.25), fontsize=14, weight='bold')
#
#
# for i in range(len(segmentation_runtime)):
#     if i != 0:
#         ax.annotate(str(segmentation_runtime[i]) + "s", (segmentation_results_24_bits[i]+0.25, segmentation_runtime_lan_24_bits[i]+0.25), fontsize=14, weight='bold')
#

ax.set_ylabel('Factor in Bandwidth', fontsize=axis_label_font_size, labelpad=7)
[i.set_linewidth(2) for i in ax.spines.values()]
ax.set_ylim([1, 11.0])
ax.set_xlim([90, 108])
ax.set_yticklabels([None, "x2", "x4", "x6", "x8", "x10"], fontsize=ticks_font_size)
ax.set_xticklabels([None] + [str(x) + "%" for x in np.arange(92, 107, 2)], fontsize=ticks_font_size)
ax.set_xlabel('Performance Relative to Baseline', fontsize=axis_label_font_size, labelpad=22)
plt.minorticks_on()
ax.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
ax.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.5, lw=1, axis='y')
ax.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.5, lw=1, axis='x')
ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.8, lw=2, axis='x')
plt.legend(prop={'size': 16}, loc='lower right')

plt.subplots_adjust(left=0.12, right=0.96, top=0.97, bottom=0.16)

plt.savefig("/home/yakir/Figure_bandwidth.png")
