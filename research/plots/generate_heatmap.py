import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np

from research.block_relu.params import BLOCK_SIZES_FULL
from research.block_relu.consts import TARGET_REDUCTIONS
BLOCK_SIZES_FULL = np.array(BLOCK_SIZES_FULL)
ss = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/block/noise_layer6_1_0_batch_0_8.npy")
# heatmap = np.zeros(shape=(16, 16))
# for i in range(1, 17):
#     for j in range(1, 17):
#         a = np.argwhere(np.all(BLOCK_SIZES_FULL==[i,j], axis=1))
#         if len(a) > 0:
#             heatmap[i-1, j-1] = x[a, 512]
# plt.imshow(heatmap)
# print('het')

from scipy.interpolate import griddata
plt.figure(figsize=(4,4))
x = np.linspace(4,17,13)
y =  np.linspace(4,17,13)
X, Y = np.meshgrid(x,y)

l = griddata((BLOCK_SIZES_FULL[:,0], BLOCK_SIZES_FULL[:,1]), ss[:, 53], (X, Y), 'cubic')
plt.imshow(l, cmap="inferno")
plt.xticks([])
plt.yticks([])
plt.xlabel("Block Size - Width", fontsize=16)
plt.ylabel("Block Size - Height", fontsize=16)
plt.tight_layout()

# plt.subplot(221)
# x0_noise = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/layers_0_out/noise_layer5_0_0_batch_0_8.npy")
# x0_signal = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/layers_0_out/signal_layer5_0_0_batch_0_8.npy")
# x0 = x0_noise / x0_signal
#
# x1_noise = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/layers_0_out/noise_layer4_0_0_batch_0_8.npy")
# x1_signal = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/layers_0_out/signal_layer4_0_0_batch_0_8.npy")
# x1 = x1_noise / x1_signal
#
# y_noise =  np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/layers_1_out/noise_layer4_0_0_batch_0_8.npy")
# y_signal =  np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/layers_1_out/signal_layer4_0_0_batch_0_8.npy")
# y = y_noise / y_signal
#
# N = 13
# plt.plot(np.convolve(x0[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#9DC3E6", lw=1.5)
# plt.plot(np.convolve(x1[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#A9D18E", lw=1.5)
# plt.plot(np.convolve(y[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="black", lw=2.5)
# plt.xlim([0, 0.2])
# plt.ylim([0, 0.5])
#
#
#
# plt.subplot(222)
# x0_noise = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_4_out/noise_layer2_0_0_batch_0_8.npy")
# x0_signal = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_4_out/signal_layer2_0_0_batch_0_8.npy")
# x0 = x0_noise / x0_signal
#
# x1_noise = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_3_out/noise_layer2_0_0_batch_0_8.npy")
# x1_signal = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_3_out/signal_layer2_0_0_batch_0_8.npy")
# x1 = x1_noise / x1_signal
#
#
# N = 13
# plt.plot(np.convolve(x1[:, 1],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#9DC3E6", lw=1.5)
# plt.plot(np.convolve(x1[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#A9D18E", lw=1.5)
# plt.plot(np.convolve(x0[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="black", lw=2.5)
# plt.xlim([0, 0.05])
# plt.ylim([0, 0.5])
#
# plt.subplot(223)
# x0_noise = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_0_out/noise_layer3_0_0_batch_0_8.npy")
# x0_signal = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_0_out/signal_layer3_0_0_batch_0_8.npy")
# x0 = x0_noise / x0_signal
#
# x1_noise = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_1_out/noise_layer3_0_0_batch_0_8.npy")
# x1_signal = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_1_out/signal_layer3_0_0_batch_0_8.npy")
# x1 = x1_noise / x1_signal
#
#
# N = 7
# plt.plot(np.convolve(x0[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#A9D18E", lw=2)
# plt.plot(np.convolve(x0[:, 1],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#9DC3E6", lw=2)
# plt.plot(np.convolve(x0[:, 2],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#F4B183", lw=2)
# plt.plot(np.convolve(x0[:, 3],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#FFD966", lw=2)
# plt.plot(np.convolve(x1[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="black", lw=3)
# plt.xlim([0,0.02])
# plt.ylim([0,0.7])
#
# plt.subplot(224)
# x0_noise = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_1_out/noise_layer7_0_0_batch_0_8.npy")
# x0_signal = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_1_out/signal_layer7_0_0_batch_0_8.npy")
# x0 = x0_noise / x0_signal
#
# x1_noise = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_2_out/noise_layer7_0_0_batch_0_8.npy")
# x1_signal = np.load("/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/channels_2_out/signal_layer7_0_0_batch_0_8.npy")
# x1 = x1_noise / x1_signal
#
#
# N = 3
# plt.plot(np.convolve(x0[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#A9D18E", lw=2)
# plt.plot(np.convolve(x0[:, 1],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#9DC3E6", lw=2)
# plt.plot(np.convolve(x0[:, 2],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#F4B183", lw=2)
# plt.plot(np.convolve(x0[:, 3],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#FFD966", lw=2)
# plt.plot(np.convolve(x1[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="black", lw=3)
# plt.xlim([0,0.01])
# plt.ylim([0,0.7])
#
#
#
#
# import numpy as np
# from scipy.interpolate import griddata
# import matplotlib.pyplot as plt
#
# x = np.linspace(-1,1,100)
# y =  np.linspace(-1,1,100)
# X, Y = np.meshgrid(x,y)
#
# def f(x, y):
#     s = np.hypot(x, y)
#     phi = np.arctan2(y, x)
#     tau = s + s*(1-s)/5 * np.sin(6*phi)
#     return 5*(1-tau) + tau
#
# T = f(X, Y)
# # Choose npts random point from the discrete domain of our model function
# npts = 400
# px, py = np.random.choice(x, npts), np.random.choice(y, npts)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# plt.plot(np.convolve(x0[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#A9D18E", lw=2)
# plt.plot(np.convolve(x0[:, 1],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#9DC3E6", lw=2)
# plt.plot(np.convolve(x0[:, 2],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#F4B183", lw=2)
# plt.plot(np.convolve(x0[:, 3],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="#FFD966", lw=2)
# plt.plot(np.convolve(x1[:, 0],np.ones(N)/N, mode='valid'), TARGET_REDUCTIONS[(N-1)//2:-(N-1)//2], color="black", lw=3)
#

plt.figure(figsize=(4,4))
plt.plot(6/np.arange(0.01,1,0.01)**1.1,np.arange(0.01,1,0.01), color="#A9D18E", lw=1.5)
plt.plot(2.5/np.arange(0.01,1,0.01),np.arange(0.01,1,0.01), color="#9DC3E6", lw=1.5)
plt.plot(11/np.arange(0.01,1,0.01)**0.93,np.arange(0.01,1,0.01), color="black", lw=2.5)
plt.xlabel("Distortion", fontsize=16)
plt.ylabel("Rate", fontsize=16)
plt.xlim([0,100])
plt.ylim([0,0.5])
plt.xticks([])
plt.yticks([])
plt.tight_layout()


plt.figure(figsize=(4,4))
plt.plot(3/np.arange(0.01,1,0.01)**1.05,np.arange(0.01,1,0.01), color="#A9D18E", lw=1.5)
plt.plot(1/np.arange(0.01,1,0.01),np.arange(0.01,1,0.01), color="#9DC3E6", lw=1.5)
plt.plot(5/np.arange(0.01,1,0.01)**0.93,np.arange(0.01,1,0.01), color="black", lw=2.5)
plt.xlabel("Distortion", fontsize=16)
plt.ylabel("Rate", fontsize=16)
plt.xlim([0,100])
plt.ylim([0,0.5])
plt.xticks([])
plt.yticks([])
plt.tight_layout()



plt.figure(figsize=(4,4))
plt.plot(0.02+1.3/np.arange(0.01,1,0.01)**1.1,np.arange(0.01,1,0.01), color="#A9D18E", lw=1.5)
plt.plot(1.6/np.arange(0.01,1,0.01)**1,np.arange(0.01,1,0.01), color="#9DC3E6", lw=1.5)
plt.plot(2/np.arange(0.01,1,0.01)**0.95,np.arange(0.01,1,0.01), color="#F4B183", lw=1.5)
plt.plot(1/np.arange(0.01,1,0.01),np.arange(0.01,1,0.01), color="#FFD966", lw=1.5)
plt.plot(0.4*3/np.arange(0.01,1,0.01)**1.2,np.arange(0.01,1,0.01), color="black", lw=2.5)
plt.xlabel("Distortion", fontsize=16)
plt.ylabel("Rate", fontsize=16)
plt.xlim([0,100])
plt.ylim([0,0.5])
plt.xticks([])
plt.yticks([])
plt.tight_layout()