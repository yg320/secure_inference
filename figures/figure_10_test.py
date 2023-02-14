import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

budget = [7.17, 12.28, 14.33, 24.57, 28.67, 49.15, 57.34, 114.69, 196.61, 229.38, 557.06]
theirs = [62.3, 64.97, 65.36, 68.41, 68.68, 69.50, 72.68, 74.72, 75.51, 76.22, 74.46]
ours =   [59.1, 62.3,  64.4, 68.40, 69.82, 73.21, 73.52, 75.49, 77.80, 77.92, 77.93]

plt.plot(budget, theirs, label="Theirs")
plt.plot(budget, ours, label="ours")
plt.semilogx()