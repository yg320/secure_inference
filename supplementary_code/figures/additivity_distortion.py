import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import numpy as np

content = pickle.load(open("/home/yakir/distortion_additivity.pickle", "rb"))
a = np.array(content["additive_noises"][:3000])
b = np.array(content["noises"][:3000])

plt.scatter(a, b)
a = np.stack([a, a])
a[1,:] = 1

teta = np.invert(a @ a.T) @ a.T @ b

aaa = []
for i in range(100000):
    x0 = np.random.choice(3000)
    x1 = np.random.choice(3000)

    aaa.append(
        np.logical_or(np.logical_and(a[x0] <= a[x1], b[x0] <= b[x1]),
                      np.logical_and(a[x0] > a[x1], b[x0] > b[x1]))
    )
print(np.mean(aaa))