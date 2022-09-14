import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from research.block_relu.params import BLOCK_SIZE_COMPLETE
import numpy as np

x0 = np.load("/home/yakir/Data2/assets_v4/deformations/coco_stuff164k/ResNetV1c/block/noise_stem_2_batch_0_8.npy")
y0 = np.load("/home/yakir/Data2/assets_v4/deformations/coco_stuff164k/ResNetV1c/block/signal_stem_2_batch_0_8.npy")
x1 = np.load("/home/yakir/Data2/assets_v4/deformations/coco_stuff164k/ResNetV1c/block/noise_stem_2_batch_1_8.npy")
y1 = np.load("/home/yakir/Data2/assets_v4/deformations/coco_stuff164k/ResNetV1c/block/signal_stem_2_batch_1_8.npy")

distortion = (x0 + x1)[:,1] / (y0 + y1)[:,1]
distortion[0] = 0
distortion_x = np.arange(0, distortion.max(), distortion.max()/10000)
rate = np.array([1/x[0]/x[1] for x in BLOCK_SIZE_COMPLETE])



COLOR_TEMPERATURE = "#69b3a2"
COLOR_PRICE = "#3399e6"

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(distortion.reshape((64, 64)), cmap="inferno")
plt.xlabel("Block Size - Width", fontsize=16)
plt.ylabel("Block Size - Height", fontsize=16)
plt.subplot(122)
plt.scatter(distortion, rate, s=4, color=COLOR_PRICE)
plt.plot(distortion_x, [rate[distortion<=d].min() for d in distortion_x], lw=2, color=COLOR_TEMPERATURE)
plt.xlabel("Distortion", fontsize=16)
plt.ylabel("Rate", fontsize=16)
plt.tight_layout()