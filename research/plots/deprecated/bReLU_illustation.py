import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F

x = np.load("/home/yakir/Data2/activation_layers/Stem-2/sample-1.npy")

channels = [3,5,14, 24]
# for i in range(4):
#     plt.subplot(2,2,i + 1)
#     plt.imshow(x[0,channels[i],:256:4,:256:4] > np.median(x[0,channels[i],:256,:256]), cmap='hot')#, cmap='gray')
#     plt.title(i)

images = [torch.Tensor(x[:,channels[i]:channels[i]+1,:256,:256]) - np.median(x[:,channels[i]:channels[i]+1,:256,:256]) for i in range(4)]

scale_0 = [1, 3]
scale_1 = [2, 2]
scale_2 = [8, 4]
scale_3 = [3, 2]
ap_0 = torch.nn.AvgPool2d(kernel_size=scale_0, stride=scale_0, ceil_mode=True)(images[0])
ap_1 = torch.nn.AvgPool2d(kernel_size=scale_1, stride=scale_1, ceil_mode=True)(images[1])
ap_2 = torch.nn.AvgPool2d(kernel_size=scale_2, stride=scale_2, ceil_mode=True)(images[2])
ap_3 = torch.nn.AvgPool2d(kernel_size=scale_3, stride=scale_3, ceil_mode=True)(images[3])

rl_0 = ap_0.sign().add(1).div(2)
rl_1 = ap_1.sign().add(1).div(2)
rl_2 = ap_2.sign().add(1).div(2)
rl_3 = ap_3.sign().add(1).div(2)

us_0 = F.interpolate(input=rl_0, scale_factor=scale_0)
us_1 = F.interpolate(input=rl_1, scale_factor=scale_1)
us_2 = F.interpolate(input=rl_2, scale_factor=scale_2)
us_3 = F.interpolate(input=rl_3, scale_factor=scale_3)

out_0 = us_0[:,:,:256,:256] * images[0]
out_1 = us_1[:,:,:256,:256] * images[1]
out_2 = us_2[:,:,:256,:256] * images[2]
out_3 = us_3[:,:,:256,:256] * images[3]

import cv2

cv2.imwrite("/home/yakir/tmp/im_0.png", ((images[0][0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/im_1.png", ((images[1][0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/im_2.png", ((images[2][0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/im_3.png", ((images[3][0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))

cv2.imwrite("/home/yakir/tmp/ap_0.png", ((ap_0[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/ap_1.png", ((ap_1[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/ap_2.png", ((ap_2[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/ap_3.png", ((ap_3[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))

cv2.imwrite("/home/yakir/tmp/rl_0.png", ((rl_0[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/rl_1.png", ((rl_1[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/rl_2.png", ((rl_2[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/rl_3.png", ((rl_3[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))

cv2.imwrite("/home/yakir/tmp/us_0.png", ((us_0[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/us_1.png", ((us_1[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/us_2.png", ((us_2[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/us_3.png", ((us_3[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))

cv2.imwrite("/home/yakir/tmp/out_0.png", ((out_0[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/out_1.png", ((out_1[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/out_2.png", ((out_2[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))
cv2.imwrite("/home/yakir/tmp/out_3.png", ((out_3[0, 0].numpy().clip(-1, 1) + 1) * 127.5).astype(np.uint8))