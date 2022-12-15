import collections

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import glob
import collections

noises_distorted = []
noises_additive_layer_distorted = []
losses_distorted = []
loss_additive_layer_distorted = []

snr_additive_distorted = []
snr_distorted = []

files =glob.glob("/home/yakir/distortion_approximation/get_network_additivity_by_layer_batch_large_debug_2/*.pickle")
for f in files:
    content = pickle.load(open(f, 'rb'))

    noises_distorted.append(content['noises_distorted'])
    noises_additive_layer_distorted.append(content['noises_additive_layer_distorted'])

noises_distorted = np.array(noises_distorted)
noises_additive_layer_distorted = np.array(noises_additive_layer_distorted)

np.corrcoef(noises_distorted[:,0,0], noises_additive_layer_distorted.sum(axis=1)[:,0,0])
plt.scatter(noises_distorted[:,0,0], noises_additive_layer_distorted.sum(axis=1)[:,0,0])
    # losses_distorted.append(content['losses_distorted'].mean())
    # loss_additive_layer_distorted.append(content['loss_additive_layer_distorted'].mean(axis=0).mean())
    #
    # snr_additive_distorted.append(
    #     np.log(content['signals_additive_layer_distorted'].sum(axis=0).mean()/content['noises_additive_layer_distorted'].sum(axis=0).mean())
    # )
    #
    # snr_distorted.append(
    #     np.log(content['signals_distorted'].sum(axis=0).mean() / content['noises_distorted'].sum(axis=0).mean())
    #
    # )

# from sklearn
# np.corrcoef(noises_additive_layer_distorted, noises_distorted)
# np.corrcoef(loss_additive_layer_distorted, losses_distorted)
# plt.scatter(loss_additive_layer_distorted, losses_distorted)
# plt.scatter(noises_additive_layer_distorted, noises_distorted)
# plt.scatter(snr_additive_distorted, snr_distorted)
# np.corrcoef(snr_additive_distorted, snr_distorted)