import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


colors = ["#03C03C", "#976ED7", "#C23B23", "#579ABE"]
colors = ["#042e60", "#c0022f"]
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))
ax1 = axes[0]
performance_from_baseline = [100.        ,  97.4003919 ,  93.82103201][::-1]
run_time_boost_lan = [1.        , 5.61625583, 5.94499295][::-1]
run_time_boost_wan = [1.        , 5.62985332, 6.06976744][::-1]
# color = 'tab:red'

ax1.set_xlabel('Accuracy Relative to Baseline (%)', fontsize=13, labelpad=7, fontweight='bold')
ax1.set_ylabel('LAN Run-Time (s)', color=colors[0], fontsize=13, labelpad=7, fontweight='bold')
ax1.plot(performance_from_baseline, run_time_boost_lan, "-o", color=colors[0], lw=3, markersize=5)
ax1.tick_params(axis='y', labelcolor=colors[0])
ax1.set_ylim([0.8, 6.5])
ax1.set_yticklabels([None, "28.12\n(x1)", "14.06\n(x2)", "9.37\n(x3)", "7.03\n(x4)", "5.62\n(x5)", "4.68\n(x6)"], fontsize=10, fontweight='bold', rotation=0)
ax1.set_xlim([93.5, 100])

#
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.8)
ax1.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


ax2.set_ylabel('WAN Run-Time (s)', color=colors[1], fontsize=13, labelpad=7, fontweight='bold') # we already handled the x-label with ax1
ax2.plot(performance_from_baseline, run_time_boost_wan, "-o", color=colors[1], lw=3, markersize=5)
ax2.tick_params(axis='y', labelcolor=colors[1])
ax2.set_yticklabels([None, "421.8\n(x1)", "210.9\n(x2)", "140.5\n(x3)", "105.4\n(x4)", "84.3\n(x5)", "70.1\n(x6)"], fontweight='bold', fontsize=10)

ax1.set_xticklabels(labels = [l.get_text() for l in ax1.get_xticklabels()], fontweight='bold')

ax2.set_xticks(np.arange(93.5,101,1))
# plt.yticks(np.arange(1,7,1))
ax2.set_ylim([0.8, 6.5])














ax1 = axes[1]
performance_from_baseline = [100.        ,  97.4003919 ,  93.82103201][::-1]
run_time_boost_lan = [1.        , 5.61625583, 5.94499295][::-1]
run_time_boost_wan = [1.        , 5.62985332, 6.06976744][::-1]
# color = 'tab:red'
ax1.set_xlabel('mIoU Relative to Baseline - (%)', fontsize=12, labelpad=7, fontweight='bold')
ax1.set_ylabel('LAN Run-Time (s)', color=colors[0], fontsize=12, labelpad=7, fontweight='bold')
ax1.plot(performance_from_baseline, run_time_boost_lan, "-o", color=colors[0], lw=3, markersize=5)
ax1.tick_params(axis='y', labelcolor=colors[0])
ax1.set_ylim([0.8, 6.5])
ax1.set_yticklabels([None, "28.12s\n(x1)", "14.06\n(x2)", "9.37\n(x3)", "7.03\n(x4)", "5.62\n(x5)", "4.68\n(x6)"], fontweight='bold', rotation=0, fontsize=10)
ax1.set_xlim([93.5, 100])

#
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.8)
ax1.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('WAN Run-Time (seconds)', color=colors[1], fontsize=12, labelpad=7, fontweight='bold') # we already handled the x-label with ax1
ax2.plot(performance_from_baseline, run_time_boost_wan, "-o", color=colors[1], lw=3, markersize=5)
ax2.tick_params(axis='y', labelcolor=colors[1])
ax2.set_yticklabels([None, "421.8\n(x1)", "210.9\n(x2)", "140.5\n(x3)", "105.4\n(x4)", "84.3\n(x5)", "70.1\n(x6)"], fontweight='bold', rotation=0, fontsize=10)

ax1.set_xticklabels(labels = [l.get_text() for l in ax1.get_xticklabels()], fontweight='bold')

ax2.set_xticks(np.arange(93.5,101,1))
# plt.yticks(np.arange(1,7,1))
ax2.set_ylim([0.8, 6.5])

fig.tight_layout()
