import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# https://xkcd.com/color/rgb/


colors = ["#03C03C", "#976ED7", "#C23B23", "#579ABE"]
colors = ["#042e60", "#c0022f"]
colors = ["#3399e6", "#69b3a2"]
ticks_font_size = 14
axis_label_font_size = 14
plot_lw = 3
plot_markersize = 10
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 10), constrained_layout=True)


ax1 = axes[0]
performance_from_baseline = 100 * np.array([71.82,  74.56, 75.67, 76.01, 76.55 ][::-1]) / 76.55
run_time_boost_lan = [5.94, 5.61, 5.28, 4.78, 1][::-1]
run_time_boost_wan = [6.06, 5.62, 5.12, 4.49, 1][::-1]
# color = 'tab:red'

# ax1.set_xlabel('Accuracy Relative to Baseline (%)', fontsize=axis_label_font_size, labelpad=7, fontweight='bold')
ax1.set_ylabel('LAN (s)', color=colors[0], fontsize=axis_label_font_size, labelpad=7, fontweight='bold')
ax1.plot(performance_from_baseline, run_time_boost_lan, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)

ax1.tick_params(axis='y', labelcolor=colors[0])
ax1.set_ylim([0.8, 6.5])
ax1.set_yticklabels(["", "28.1", "14.0", "9.3", "7.0", "5.6", "4.6"], fontsize=ticks_font_size, fontweight='bold', rotation=0)
ax1.set_xlim([93, 101])
#

ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
ax1.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


ax2.set_ylabel('WAN (s)', color=colors[1], fontsize=axis_label_font_size, labelpad=7, fontweight='bold') # we already handled the x-label with ax1
ax2.plot(performance_from_baseline, run_time_boost_wan, ".-", color=colors[1], lw=plot_lw, markersize=plot_markersize)
ax2.tick_params(axis='y', labelcolor=colors[1])
ax2.set_yticklabels(["421", "210", "140", "105", "84", "70"], fontweight='bold', fontsize=ticks_font_size)

ax1.set_xticklabels(labels = [None, 94, 95, 96, 97, 98, 99, 100],fontsize=ticks_font_size,  fontweight='bold')

ax2.set_xticks(np.arange(93,101,1))
ax2.set_yticks(np.arange(1,7,1))
ax2.set_ylim([0.8, 6.5])
ax2.set_xlim([93, 101])


for i in range(1, 7):
    ax1.text(100.5, i, f'x{i}',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=15, color='black')

# ax2.xaxis.set_major_locator(MultipleLocator(1))
# ax2.yaxis.set_major_locator(MultipleLocator(1))
# ax2.xaxis.set_minor_locator(MultipleLocator(0.25))
# ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
ax2.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
ax2.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)

#
ax1 = axes[1]
# performance_from_baseline = [100.        ,  97.4003919 ,  93.82103201][::-1]
# run_time_boost_lan = [1.        , 5.61625583, 5.94499295][::-1]
# run_time_boost_wan = [1.        , 5.62985332, 6.06976744][::-1]
# # color = 'tab:red'
#
# # ax1.set_xlabel('Accuracy Relative to Baseline (%)', fontsize=axis_label_font_size, labelpad=7, fontweight='bold')
# ax1.set_ylabel('LAN (s)', color=colors[0], fontsize=axis_label_font_size, labelpad=7, fontweight='bold')
# ax1.plot(performance_from_baseline, run_time_boost_lan, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)
#
# ax1.tick_params(axis='y', labelcolor=colors[0])
# ax1.set_ylim([0.8, 6.5])
# ax1.set_yticklabels([None, "28.1", "14.0", "9.3", "7.0", "5.6", "4.6"], fontsize=ticks_font_size, fontweight='bold', rotation=0)
# ax1.set_xlim([93.5, 100])
# plt.text(7, 96, 'Close Me!', dict(size=30))
# #
# ax1.xaxis.set_major_locator(MultipleLocator(1))
# ax1.yaxis.set_major_locator(MultipleLocator(1))
# ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
# ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
# ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
# ax1.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.1)
ax1.set_xlabel('Performance Relative to Baseline (%)', fontsize=axis_label_font_size, labelpad=7, fontweight='bold')
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#
# ax2.set_ylabel('WAN (s)', color=colors[1], fontsize=axis_label_font_size, labelpad=7, fontweight='bold') # we already handled the x-label with ax1
# ax2.plot(performance_from_baseline, run_time_boost_wan, "-o", color=colors[1], lw=plot_lw, markersize=plot_markersize)
# ax2.tick_params(axis='y', labelcolor=colors[1])
# ax2.set_yticklabels([None, "421", "210", "140", "105", "84", "70"], fontweight='bold', fontsize=ticks_font_size)
#
# ax1.set_xticklabels(labels = [94, 95, 96, 97, 98, 99, 100],fontsize=ticks_font_size,  fontweight='bold')
#
# ax2.set_xticks(np.arange(93.5,101,1))
# # plt.yticks(np.arange(1,7,1))
# ax2.set_ylim([0.8, 6.5])
#
#
for i in range(1, 7):
    ax1.text(100.4, i, f'x{i}',
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=15, color='black')
# fig.canvas.draw()
# fig.tight_layout()
