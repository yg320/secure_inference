import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
green = "#56ae57"
red = "#db5856"
purple = "tab:purple"
blue = "#3399e6"

results_cifar_baseline = 78.27
results_imnet_baseline = 76.55
results_ade20_baseline = 34.08
results_voc12_baseline = 77.68

runtime_cifar_baseline = 2.8574
runtime_imnet_baseline = 31.6239
runtime_ade20_baseline = 379.4955
runtime_voc12_baseline = 671.46

runtime_cifar_16 = np.array([0.771,	1.5084,	1.572,	1.645,	1.479,	1.613])
runtime_cifar_64 = np.array([0.783,	1.5551,	1.610,	1.712,	1.577,	1.7443])

runtime_imnet_16 = np.array([5.528,	5.8332,	6.098,	6.206,	6.533,	6.680])
runtime_imnet_64 = np.array([5.896,	6.476,	7.189,	7.874,	8.735,	9.313])

runtime_ade20_20 = np.array([32.78,	35.997,	38.29,	41.05,	44.32,	46.78])
runtime_ade20_64 = np.array([37.48,	45.142,	54.27,	61.89,	73.04,	81.84])

runtime_voc12_20 = np.array([84.64,	92.536,	100.4,	106.1,	111.8,	117.2])
runtime_voc12_64 = np.array([96.65,	117.16,	135.2,	155.6,	178.5,	200.3])

perf_imnet = 100 * np.array([65.59, 70.36, 72.23, 73.30, 74.03, 74.40]) / results_imnet_baseline
perf_ade20 = 100 * np.array([33.23, 34.73, 35.53, 35.77, 36.01, 36.31]) / results_ade20_baseline
perf_voc12 = 100 * np.array([72.23, 74.27, 75.89, 76.26, 76.86, 77.48]) / results_voc12_baseline
perf_cifar = 100 * np.array([65.52, 71.16, 74.85, 74.64, 76.19, 76.88]) / results_cifar_baseline

runtime_boost_imnet = runtime_imnet_baseline / runtime_imnet_16
runtime_boost_cifar = runtime_cifar_baseline / runtime_cifar_16
runtime_boost_ade20 = runtime_ade20_baseline / runtime_ade20_20
runtime_boost_voc12 = runtime_voc12_baseline / runtime_voc12_20

runtime_boost_imnet_64 = runtime_imnet_baseline / runtime_imnet_64
runtime_boost_cifar_64 = runtime_cifar_baseline / runtime_cifar_64
runtime_boost_ade20_64 = runtime_ade20_baseline / runtime_ade20_64
runtime_boost_voc12_64 = runtime_voc12_baseline / runtime_voc12_64

bandwidth_cifar_baseline = 493540032 / 1000000000
bandwidth_imnet_baseline = 6355757824 / 1000000000
bandwidth_ade20_baseline = 64005736576 / 1000000000
bandwidth_voc12_baseline = 106205953664 / 1000000000

bandwidth_cifar_16 = np.array([52001120,	210344192,	214448640,	218642720,	222651168,	226751392]) / 1000000000
bandwidth_cifar_64 = np.array([56813600,	219969728,	228887232,	237894080,	246715584,	255628864]) / 1000000000

bandwidth_imnet_16 = np.array([851316560,	925326776,	996765080,	1067109704,	1136883968,	1206581792])   / 1000000000
bandwidth_imnet_64 = np.array([934335440,	1091364824,	1245822296,	1399186088,	1551979520,	1704696512])   / 1000000000

bandwidth_ade20_20 = np.array([6093331576,	7049548528,	7985972384,	8896302968,	9807188568,	10685352024]) / 1000000000
bandwidth_ade20_64 = np.array([7004879488,	8858244736,	10692626048,	12500557952,	14310337152,	16085475456]) / 1000000000

bandwidth_voc12_20 = np.array([11757613696,	13783743656,	15664828024,	17558623712,	19413113208,	21202476336]) / 1000000000
bandwidth_voc12_64 = np.array([13765095040,	17783405696,	21574989952,	25407395456,	29186553984,	32841516672]) / 1000000000


bandwidth_boost_imnet = bandwidth_imnet_baseline / bandwidth_imnet_16
bandwidth_boost_cifar = bandwidth_cifar_baseline / bandwidth_cifar_16
bandwidth_boost_ade20 = bandwidth_ade20_baseline / bandwidth_ade20_20
bandwidth_boost_voc12 = bandwidth_voc12_baseline / bandwidth_voc12_20

bandwidth_boost_imnet_64 = bandwidth_imnet_baseline / bandwidth_imnet_64
bandwidth_boost_cifar_64 = bandwidth_cifar_baseline / bandwidth_cifar_64
bandwidth_boost_ade20_64 = bandwidth_ade20_baseline / bandwidth_ade20_64
bandwidth_boost_voc12_64 = bandwidth_voc12_baseline / bandwidth_voc12_64


ticks_font_size = 16
axis_label_font_size = 20
plot_lw = 5
plot_markersize = 15

plt.figure(figsize=(8, 12))
plt.subplot(211)
plt.plot(perf_ade20, bandwidth_boost_ade20, ".-", color=green, lw=plot_lw, markersize=plot_markersize, label="ADE20K - Seg.")
plt.plot(perf_voc12, bandwidth_boost_voc12, ".-", color=purple, lw=plot_lw, markersize=plot_markersize, label="VOC12 - Seg.")
plt.plot(perf_imnet, bandwidth_boost_imnet, ".-", color=blue, lw=plot_lw, markersize=plot_markersize, label="ImageNet - Cls.")
plt.plot(perf_cifar, bandwidth_boost_cifar, ".-", color=red, lw=plot_lw, markersize=plot_markersize, label="CIFAR100 - Cls.")
plt.plot(perf_ade20, bandwidth_boost_ade20_64, ".--", color=green, lw=plot_lw, markersize=plot_markersize)
plt.plot(perf_voc12, bandwidth_boost_voc12_64, ".--", color=purple, lw=plot_lw, markersize=plot_markersize)
plt.plot(perf_imnet, bandwidth_boost_imnet_64, ".--", color=blue, lw=plot_lw, markersize=plot_markersize)
plt.plot(perf_cifar, bandwidth_boost_cifar_64, ".--", color=red, lw=plot_lw, markersize=plot_markersize)

# peft = [perf_ade20[1], perf_voc12[1], perf_imnet[1], perf_cifar[1]]
# bandwidth_boost = [bandwidth_boost_ade20[1], bandwidth_boost_voc12[1], bandwidth_boost_imnet[1], bandwidth_boost_cifar[1]]
# bandwidth = [bandwidth_ade20_20[1], bandwidth_voc12_20[1], bandwidth_imnet_16[1], bandwidth_cifar_16[1]]
# bandwidth = [str(round(x, 2)) for x in bandwidth]
# ax = plt.gca()
# for i in range(4):
#     ax.annotate(str(bandwidth[i]) + "GB", (peft[i]+0.25, bandwidth_boost[i]+0.25), fontsize=14, weight='bold')



ax = plt.gca()

ax.set_ylabel('Comm. Factor', fontsize=axis_label_font_size, labelpad=7)
[i.set_linewidth(2) for i in ax.spines.values()]
ax.set_ylim([1, 12.0])
ax.set_xlim([90, 108])
ax.set_yticklabels([None, "x2", "x4", "x6", "x8", "x10"], fontsize=ticks_font_size)
ax.set_xticklabels([None] + [str(x) + "%" for x in np.arange(92, 107, 2)], fontsize=ticks_font_size)
# ax.set_xlabel('Performance Relative to Baseline', fontsize=axis_label_font_size, labelpad=22)
plt.minorticks_on()
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

ax.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
ax.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.5, lw=1, axis='y')
ax.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.5, lw=1, axis='x')
ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.8, lw=2, axis='x')
# plt.legend(prop={'size': 16}, loc='lower right')
ax.set_xticklabels([])

plt.subplot(212)

plt.plot(perf_ade20, runtime_boost_ade20, ".-", color=green, lw=plot_lw, markersize=plot_markersize, label="ADE20K - Seg.")
plt.plot(perf_voc12, runtime_boost_voc12, ".-", color=purple, lw=plot_lw, markersize=plot_markersize, label="VOC12 - Seg.")
plt.plot(perf_imnet, runtime_boost_imnet, ".-", color=blue, lw=plot_lw, markersize=plot_markersize, label="ImageNet - Cls.")
plt.plot(perf_cifar, runtime_boost_cifar, ".-", color=red, lw=plot_lw, markersize=plot_markersize, label="CIFAR100 - Cls.")

plt.plot(perf_ade20, runtime_boost_ade20_64, ".--", color=green, lw=plot_lw, markersize=plot_markersize)
plt.plot(perf_voc12, runtime_boost_voc12_64, ".--", color=purple, lw=plot_lw, markersize=plot_markersize)
plt.plot(perf_imnet, runtime_boost_imnet_64, ".--", color=blue, lw=plot_lw, markersize=plot_markersize)
plt.plot(perf_cifar, runtime_boost_cifar_64, ".--", color=red, lw=plot_lw, markersize=plot_markersize)

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

ax.set_ylabel('Runtime Factor', fontsize=axis_label_font_size, labelpad=7)
[i.set_linewidth(2) for i in ax.spines.values()]
ax.set_ylim([1, 12.0])
ax.set_xlim([90, 108])
ax.set_yticklabels([None, "x2", "x4", "x6", "x8", "x10"], fontsize=ticks_font_size)
ax.set_xticklabels([None] + [str(x) + "%" for x in np.arange(92, 107, 2)], fontsize=ticks_font_size)
ax.set_xlabel('Performance Relative to Baseline', fontsize=axis_label_font_size, labelpad=22)
plt.minorticks_on()
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

ax.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
ax.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.5, lw=1, axis='y')
ax.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.5, lw=1, axis='x')
ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.8, lw=2, axis='x')
plt.legend(prop={'size': 16}, loc='lower right')


plt.subplots_adjust(left=0.12, right=0.98, top=0.99, bottom=0.1, hspace=0.08)

plt.savefig("/home/yakir/approx_drelu.png")
