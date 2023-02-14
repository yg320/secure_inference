import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

plt.figure(figsize=(5, 5))
plt.subplot(111)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.text(0.15, 0.5, r'Pipeline Summary', fontsize=25)
plt.savefig("/home/yakir/Figure_5.png")
