import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# https://xkcd.com/color/rgb/



colors = ["#3399e6", "#69b3a2"]
ticks_font_size = 12
axis_label_font_size = 12
plot_lw = 3
plot_markersize = 10
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), constrained_layout=True)
fig.subplots_adjust(hspace=0)


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
        performance_from_baseline = 100 * np.array([71.82,  74.56, 75.67, 76.01, 76.55 ][::-1]) / 76.55
        run_time_boost_lan = [5.94, 5.61, 5.28, 4.78, 1][::-1]
        run_time_boost_wan = [6.06, 5.62, 5.12, 4.49, 1][::-1]
        # color = 'tab:red'

        # ax1.set_xlabel('Accuracy Relative to Baseline (%)', fontsize=axis_label_font_size, labelpad=7, fontweight='bold')
        ax1.set_ylabel('LAN (s)', fontsize=axis_label_font_size, labelpad=7)
        ax1.plot(performance_from_baseline, run_time_boost_lan, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)

        # ax1.tick_params(axis='y', labelcolor=colors[0])
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


    if True:
        ax1 = axes[1][0]
        [i.set_linewidth(1.5) for i in ax1.spines.values()]

        performance_from_baseline = 100 * np.array([71.82,  74.56, 75.67, 76.01, 76.55 ][::-1]) / 76.55
        run_time_boost_wan = [6.06, 5.62, 5.12, 4.49, 1][::-1]
        ax1.set_ylabel('WAN (s)', fontsize=axis_label_font_size, labelpad=7)
        ax1.plot(performance_from_baseline, run_time_boost_wan, ".-", color=colors[0], lw=plot_lw, markersize=plot_markersize)

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
        performance_from_baseline = [107.1, 97]
        run_time_boost_lan = [1, 10]
        # color = 'tab:red'

        # ax1.set_xlabel('Accuracy Relative to Baseline (%)', fontsize=axis_label_font_size, labelpad=7, fontweight='bold')
        ax1.plot(performance_from_baseline, run_time_boost_lan, ".-", color=colors[0], lw=plot_lw,
                 markersize=plot_markersize)

        # ax1.tick_params(axis='y', labelcolor=colors[0])
        ax1.set_ylim([0.0, 11])
        ax1.set_yticks([1,3,5,7,9])
        ax1.set_yticklabels(["1", "3", "5", "7", "9"], fontsize=ticks_font_size)

        ax1.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
        ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5, lw=2, axis='x')

        ax1.set_xticks(np.arange(88, 108, 2))

        ax1.set_xlim([96, 108])
        for i in [1,3,5,7,9]:
            ax1.text(107.25, i + 0.22, f'x{i}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10, color='black', fontweight='bold')

    if True:
        ax1 = axes[1][1]
        [i.set_linewidth(1.5) for i in ax1.spines.values()]
        ax1.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        performance_from_baseline = [107.1, 97]
        run_time_boost_wan = [1, 10]
        # color = 'tab:red'

        # ax1.set_xlabel('Accuracy Relative to Baseline (%)', fontsize=axis_label_font_size, labelpad=7, fontweight='bold')
        ax1.plot(performance_from_baseline, run_time_boost_wan, ".-", color=colors[0], lw=plot_lw,
                 markersize=plot_markersize)

        # ax1.tick_params(axis='y', labelcolor=colors[0])
        ax1.set_ylim([0.0, 11])
        ax1.set_yticks([1,3,5,7,9])
        ax1.set_yticklabels(["1", "3", "5", "7", "9"], fontsize=ticks_font_size)

        ax1.grid(visible=True, which='major', color='#666666', linestyle='--', alpha=1, lw=1, axis='y')
        ax1.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5, lw=2, axis='x')

        ax1.set_xticks(np.arange(96, 108, 2))

        ax1.set_xlim([88, 108])
        for i in [1,3,5,7,9]:
            ax1.text(107.25, i + 0.22, f'x{i}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10, color='black', fontweight='bold')

plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.05)

