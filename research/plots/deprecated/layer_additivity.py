import collections

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import glob
import collections

losses = []
additive_losses = []
noises = []
additive_noises = []

files = glob.glob("/home/yakir/distortion_approximation/get_layer_additivity/*.pickle")
for f in files:
    content = pickle.load(open(f,'rb'))

    losses.append(content['loss_distorted'][0])
    additive_losses.append(content['loss_additive_layer_distorted'].sum(axis=0)[0])
    noises.append(content['noises_distorted'].sum())
    additive_noises.append(content['noises_additive_layer_distorted'].sum())
import numpy as np
np.corrcoef(additive_noises, noises)
plt.scatter(additive_noises, noises)
plt.()