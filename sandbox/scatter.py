import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pickle

x = pickle.load(open("/home/yakir/x_0.pickle", 'rb'))["Noise"]
y = pickle.load(open("/home/yakir/y_0.pickle", 'rb'))["Noise"]
y[0]
plt.scatter(x.flatten(), y.flatten())
plt.scatter(x[:,-1], y[:,-1])
# plt.scatter(x[:,-2], y[:,-2])