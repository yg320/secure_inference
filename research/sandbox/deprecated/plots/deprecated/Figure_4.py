import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

COLOR_A = "#69b3a2"
COLOR_B = "#3399e6"
COLOR_C = "#c74c52"

plt.figure(figsize=(10,8))
x = np.random.uniform(size=(57, 57))
for j in range(57):
    for i in range(j+1, 57):
        x[i,j] = 0

plt.imshow(x)
plt.xlabel("Layer Index", fontsize=14)
plt.ylabel("Mutual Information", fontsize=14)
plt.xticks(np.arange(0.10,0.31,0.01))
plt.yticks(np.arange(80,101,1))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim([0.10,0.30])
plt.ylim([80,100])
plt.legend()
