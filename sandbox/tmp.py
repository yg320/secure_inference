import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

x = np.load("/home/yakir/tiny_tesnet18/distortions/conv_stride_2/distortion_collected/layer1_0_1.npy")
plt.imshow(x)

# m = 64
# o = 64
# i = 64
# f = 3
#
# (m**2*o*(6*8+19)) / (2*m**2*i+2*f**2*o*i+m**2*o)