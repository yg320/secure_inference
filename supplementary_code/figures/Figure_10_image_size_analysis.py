import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


import matplotlib.pyplot as plt
import numpy as np

# data from https://allisonhorst.github.io/palmerpenguins/
plt.figure()
species = (
    str(32),
    str(64),
    str(128),
    str(256),
    str(512),
    str(1024),
)
conv_cost = np.array([5642570880, 1666370688, 742837376, 492817536, 429263968, 413375616][::-1])
relu_cost = np.array([111031615488, 27808235520, 6931873792, 1731919872, 432980000, 108244992][::-1])
conv_ratio = conv_cost / (conv_cost + relu_cost)
relu_ratio = relu_cost / (conv_cost + relu_cost)

weight_counts = {
    "Below": conv_ratio,
    "Above": relu_ratio,
}
width = 0.5

bars = plt.bar(species, relu_ratio, width,  color="#3399e6")
for bar in bars:
    bar.set_edgecolor("black")
    bar.set_linewidth(1.8)
plt.yticks(np.arange(0, 1.1, 0.1))
[i.set_linewidth(1.8) for i in plt.gca().spines.values()]

plt.gca().tick_params(axis='both', which='major', labelsize=16)
plt.xlabel("Image Size", fontsize=18)
plt.gca().yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.8)
plt.gca().set_ylabel("Ratio of ReLUs in Comm. Cost", fontsize=18, labelpad=18)
plt.tight_layout()
plt.savefig("/home/yakir/Figure_10.png")


# # plt.bar(species, relu_ratio, width, color = "red")
# fig, ax = plt.subplots()
# bottom = np.zeros(5)
#
# for boolean, weight_count in weight_counts.items():
#     p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
#     bottom += weight_count
#
# ax.set_title("Number of penguins with above average body mass")
# ax.legend(loc="upper right")
#
# plt.show()