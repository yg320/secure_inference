import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

plt.figure(figsize=(5, 5))
plt.subplot(111)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.text(0.25, 0.5, r'DReLU stats', fontsize=25)
plt.savefig("/home/yakir/Figure_4.png")
