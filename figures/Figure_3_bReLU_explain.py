import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches

np.random.seed(124)
activation = np.random.uniform(low=-1, high=1, size=(6, 6)).round(1)


bReLU_1x1 = activation.copy()
bReLU_1x1[bReLU_1x1 < 0] = 0

bReLU_3x3 = activation.copy()
if bReLU_3x3[:3, :3].mean() < 0:
    bReLU_3x3[:3, :3] = 0
if bReLU_3x3[3:, :3].mean() < 0:
    bReLU_3x3[3:, :3] = 0
if bReLU_3x3[:3, 3:].mean() < 0:
    bReLU_3x3[:3, 3:] = 0
if bReLU_3x3[3:, 3:].mean() < 0:
    bReLU_3x3[3:, 3:] = 0

bReLU_2x3 = activation.copy()
if bReLU_2x3[:2, :3].mean() < 0:
    bReLU_2x3[:2, :3] = 0
if bReLU_2x3[2:4, :3].mean() < 0:
    bReLU_2x3[2:4, :3] = 0
if bReLU_2x3[4:6, :3].mean() < 0:
    bReLU_2x3[4:6, :3] = 0

if bReLU_2x3[:2, 3:].mean() < 0:
    bReLU_2x3[:2, 3:] = 0
if bReLU_2x3[2:4, 3:].mean() < 0:
    bReLU_2x3[2:4, 3:] = 0
if bReLU_2x3[4:6, 3:].mean() < 0:
    bReLU_2x3[4:6, 3:] = 0

height, width = activation.shape
box_color = "#3399e6" #"g"
fig = plt.figure(figsize=(5, 3))
images = [activation, bReLU_2x3]
for i in range(2):
    j = 0

    ax = fig.add_subplot(1, 2, i+1, aspect='equal')

    if i == 0 and j == 0:
        plt.title("Input activation", fontsize=12)
    if i == 1 and j == 0:
        plt.title("Output activation 2x3", fontsize=12)


    image = images[i]
    for x in range(width):
        for y in range(height):
            ax.annotate(str(image[x][y]), xy=(y, x), ha='center', va='center', fontsize=10)

    offset = .5
    ax.set_xlim(-offset, width - offset)
    ax.set_ylim(-offset, height - offset)

    ax.hlines(y=np.arange(height+1) - offset, xmin=-offset, xmax=width-offset, color='black')
    ax.vlines(x=np.arange(width+1) - offset, ymin=-offset, ymax=height-offset, color='black')

    plt.xticks([])
    plt.yticks([])

    if i == 0 and j == 0:
        rect = patches.Rectangle((-0.5, -0.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((-0.5, 1.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((-0.5, 3.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((2.5, -0.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((2.5, 1.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((2.5, 3.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

    if i == 1 and j == 0:
        rect = patches.Rectangle((-0.5, -0.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor=matplotlib.colors.to_rgba(box_color, 0.2))
        ax.add_patch(rect)
        rect = patches.Rectangle((-0.5, 1.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((-0.5, 3.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor=matplotlib.colors.to_rgba(box_color, 0.2))
        ax.add_patch(rect)

        rect = patches.Rectangle((2.5, -0.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor=matplotlib.colors.to_rgba(box_color, 0.2))
        ax.add_patch(rect)
        rect = patches.Rectangle((2.5, 1.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor=matplotlib.colors.to_rgba(box_color, 0.2))
        ax.add_patch(rect)
        rect = patches.Rectangle((2.5, 3.5), 3, 2, linewidth=5, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

    if i == 1 and j == 1:
        rect = patches.Rectangle((-0.5, -0.5), 3, 3, linewidth=5, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((-0.5, 2.5), 3, 3, linewidth=5, edgecolor=box_color, facecolor=matplotlib.colors.to_rgba(box_color, 0.2))
        ax.add_patch(rect)

        rect = patches.Rectangle((2.5, -0.5), 3, 3, linewidth=5, edgecolor=box_color, facecolor=matplotlib.colors.to_rgba(box_color, 0.2))
        ax.add_patch(rect)
        rect = patches.Rectangle((2.5, 2.5), 3, 3, linewidth=5, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

    if i == 0 and j == 1:
        for lll in range(6):
            for mmm in range(6):
                if bReLU_1x1[mmm][lll] == 0:
                    rect = patches.Rectangle((-0.5+lll, -0.5+mmm), 1, 1, linewidth=5, edgecolor=box_color, facecolor=matplotlib.colors.to_rgba(box_color, 0.2))
                else:
                    rect = patches.Rectangle((-0.5 + lll, -0.5 + mmm), 1, 1, linewidth=5, edgecolor=box_color, facecolor='none')
                ax.add_patch(rect)

plt.tight_layout()
plt.savefig("/home/yakir/Figure_3.png")
