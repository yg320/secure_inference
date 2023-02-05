





# 1x1 9608704
# 1x2 4834816
# 2x2 2434816
# 2x3 1701120
# 3x3 1189632
# 3x4 874240
# 4x4 644224

# plt.figure(figsize=(8, 4))
# plt.subplot(121)
#
# run_time_24_4x4 = 14.01/3
# run_time_24_3x4 = 14.59/3
#
# run_time_32_4x4 = 14.18/3
# run_time_32_3x4 = 15.01/3
#
# run_time_40_4x4 = 14.68/3
# run_time_40_3x4 = 15.77/3
#
# run_time_64_4x4 = 15.66/3
# run_time_64_3x4 = 17.42/3
#
# perf_64_4x4 = 71.88
# perf_40_4x4 = 71.96
# perf_32_4x4 = 71.82
#
# perf_64_3x4 = 74.36
# perf_40_3x4 = 74.26
# perf_32_3x4 = 74.56
#
# run_time_baseline = 84.3 / 3
#
# perf_baseline = 76.55
#
# run_time_boost = run_time_baseline / np.array([run_time_baseline, run_time_32_3x4, run_time_32_4x4])
# performance_degradation = 100 * np.array([perf_baseline, perf_32_3x4, perf_32_4x4]) / perf_baseline
# plt.plot(performance_degradation, run_time_boost, 'o-', label='32-bit')
#
# run_time_boost_wan = 6.525 / np.array([6.525, 1.159, 1.075])
# performance_degradation = 100 * np.array([perf_baseline, perf_32_3x4, perf_32_4x4]) / perf_baseline
# plt.plot(performance_degradation, run_time_boost_wan, 'o-', label='32-bit')
#
# run_time_32 = [run_time_32_4x4, run_time_32_3x4, run_time_baseline]
# perf_32 = [perf_32_4x4, perf_32_3x4, perf_baseline]
#
# run_time_40 = [run_time_40_4x4, run_time_40_3x4, run_time_baseline]
# perf_40 = [perf_40_4x4, perf_40_3x4, perf_baseline]
#
# run_time_64 = [run_time_baseline/run_time_64_4x4, run_time_baseline/run_time_64_3x4, run_time_baseline/run_time_baseline]
# perf_64 = [perf_64_4x4, perf_64_3x4, perf_baseline]
#
#
# plt.plot(run_time_32, perf_32, marker='o', label='32')
# plt.plot(run_time_40, perf_40, marker='o', label='40')
# plt.plot(run_time_64, perf_64, marker='o', label='64')
#
# plt.plot([5, 6], [76.55, 76.55], '--',label='Baseline', color="black")
#
# plt.plot([4.726666666666667, 5.003333333333333, 28],[71.82, 74.56, 76.55] )
# performance_relative_to_baseline_classification = 100*np.array([0.934, 0.952, 0.967, 0.99])
# performance_relative_to_baseline_segmentation = 100*np.array([0.91, 0.92, 0.93, 0.95])
#
# plt.plot([35.1, 35.5, 40.14, 40.28, 41.15, 50, 60, 100, 300], [71.88, 72, 72.3,  72.7, 73.5, 73.8, 74.4, 76.7, 76.8], marker='o', label='Baseline')
# plt.semilogx()
# # plt.subplot(211)
# plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_classification, '.-', color="#3399e6", lw=3, markersize=15)
# plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_segmentation, '.-', color="#69b3a2", lw=3, markersize=15)
# plt.xlabel("Relative DReLU Count (%)", fontsize=13, labelpad=7)
# plt.ylabel("Relative Decrease in Performance (%)", fontsize=13, labelpad=7)
#
# plt.gca().xaxis.set_major_locator(MultipleLocator(1))
# plt.gca().yaxis.set_major_locator(MultipleLocator(1))
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
# plt.gca().tick_params(axis='both', which='major', labelsize=11)
#
# plt.xlim([5, 20])
# plt.ylim([90, 100])
# plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
# plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.subplot(122)
# dReLUs_relative_to_baseline = 100*np.array([644224/9608704, 874240/9608704, 1189632/9608704, 1701120/9608704])
# performance_relative_to_baseline_classification = 100*np.array([0.934, 0.952, 0.967, 0.99])
# performance_relative_to_baseline_segmentation = 100*np.array([0.91, 0.92, 0.93, 0.95])
#
# # plt.subplot(211)
# plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_classification, '.-', color="#3399e6", lw=3, markersize=15)
# plt.plot(dReLUs_relative_to_baseline, performance_relative_to_baseline_segmentation, '.-', color="#69b3a2", lw=3, markersize=15)
# plt.xlabel("Relative DReLU Count (%)", fontsize=13, labelpad=7)
# plt.ylabel("Relative Decrease in Performance (%)", fontsize=13, labelpad=7)
#
# plt.gca().xaxis.set_major_locator(MultipleLocator(1))
# plt.gca().yaxis.set_major_locator(MultipleLocator(1))
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
# plt.gca().tick_params(axis='both', which='major', labelsize=11)
#
# plt.xlim([5, 20])
# plt.ylim([90, 100])
# plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
# plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.tight_layout()
# # plt.savefig("/home/yakir/figure_6.png")
#
#
#
#














#
# plt.gca().xaxis.set_major_locator(MultipleLocator(1))
# plt.gca().yaxis.set_major_locator(MultipleLocator(1))
# plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
# plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
# plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
# plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# # plt.gca().tick_params(axis='both', which='major', labelsize=11)
# # plt.ylim([0, 10])
#
# performance_degradation
# array([100.        ,  97.4003919 ,  93.82103201])
# run_time_boost
# array([1.        , 5.61625583, 5.94499295])
# run_time_boost_wan
# array([1.        , 5.62985332, 6.06976744])
#
#
#
#
#
# #
# #
# #
# #
# #
# # import matplotlib.pyplot as plt
# #
# # from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
# # from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# #
# # import numpy as np
# #
# # def get_demo_image():
# #     from matplotlib.cbook import get_sample_data
# #     import numpy as np
# #     f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
# #     z = np.load(f)
# #     # z is a numpy array of 15x15
# #     return z, (-3,4,-4,3)
# #
# # fig, ax = plt.subplots(figsize=[5,4])
# #
# # # prepare the demo image
# # Z, extent = get_demo_image()
# # Z2 = np.zeros([150, 150], dtype="d")
# # ny, nx = Z.shape
# # Z2[30:30+ny, 30:30+nx] = Z
# #
# # # extent = [-3, 4, -4, 3]
# # ax.imshow(Z2, extent=extent, interpolation="nearest",
# #           origin="lower")
# #
# # axins = zoomed_inset_axes(ax, 6, loc=1) # zoom = 6
# # axins.imshow(Z2, extent=extent, interpolation="nearest",
# #              origin="lower")
# #
# # # sub region of the original image
# # x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
# # axins.set_xlim(x1, x2)
# # axins.set_ylim(y1, y2)
# #
# # plt.xticks(visible=False)
# # plt.yticks(visible=False)
# #
# # # draw a bbox of the region of the inset axes in the parent axes and
# # # connecting lines between the bbox and the inset axes area
# # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
# #
# # plt.draw()
# # plt.show()