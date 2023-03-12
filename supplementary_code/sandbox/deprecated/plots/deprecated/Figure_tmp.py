# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
# import numpy as np
#
#
#
# # rates_channel_0 = [0.5, 0.5, 0.33, 0.25]
# # distortion_channel_0 = [0.12, 0.18, 0.2, 0.24]
# #
# #
# # rates_channel_1 = [0.5, 0.5, 0.33, 0.25]
# # distortion_channel_1 = [0.1, 0.12, 0.15, 0.18]
# #
# # plt.scatter(distortion_channel_0, rates_channel_0)
# # plt.scatter(distortion_channel_1, rates_channel_1)
#
# x = np.load("/home/yakir/Data2/assets_v3_tmp/deformations/coco_stuff164k/ResNetV1c/noise_vector_single.npy")
# y = np.load("/home/yakir/Data2/assets_v3_tmp/deformations/coco_stuff164k/ResNetV1c/noise_vector_agg.npy")
#
# cumm_sum = []
# for i in range(1, x.shape[0] + 1):
#     cumm_sum.append(x[:i].sum())
#
# plt.scatter([128,256,768, 896, 1024, 1536, 1664, 1792], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# [128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512]
#
# plt.plot(range(len(cumm_sum)), np.array(cumm_sum))
# plt.plot(range(len(cumm_sum)), y)
# plt.plot(range(len(cumm_sum)), x)
# y[-1], x.sum()
#
# x = np.load("/home/yakir/Data2/assets_v3_tmp/deformations/coco_stuff164k/ResNetV1c/tmp_3.npy")
# y = x[0,0,0]
#
# a = [
#     y[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     y[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     y[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     y[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     y[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     y[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     y[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     y[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     y[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     y[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     y[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     y[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     y[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     y[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     y[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#     y[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#     y[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# ]
#
# plt.scatter(range(17), a)
# plt.scatter([0], [y[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
#
#
#
# plt.plot(x[:,0,0])
# plt.imshow(x[:,0,:])
#
# x[4,0,0] + x[0,4,0] + x[0,0,4]
# x[4, 4, 4]
#
#
#
#
#
# import matplotlib
#
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
# import numpy as np
# from research.block_relu.params import BLOCK_SIZE_COMPLETE
# import numpy as np
#
# x0 = np.load("/home/yakir/Data2/assets_v4/deformations/coco_stuff164k/ResNetV1c/block/noise_stem_2_batch_0_8.npy")
# y0 = np.load("/home/yakir/Data2/assets_v4/deformations/coco_stuff164k/ResNetV1c/block/signal_stem_2_batch_0_8.npy")
# x1 = np.load("/home/yakir/Data2/assets_v4/deformations/coco_stuff164k/ResNetV1c/block/noise_stem_2_batch_1_8.npy")
# y1 = np.load("/home/yakir/Data2/assets_v4/deformations/coco_stuff164k/ResNetV1c/block/signal_stem_2_batch_1_8.npy")
#
# distortion = (x0 + x1)[:,1] / (y0 + y1)[:,1]
# distortion[0] = 0
# distortion_x = np.arange(0, distortion.max(), distortion.max()/10000)
# rate = np.array([1/x[0]/x[1] for x in BLOCK_SIZE_COMPLETE])
#
#
#
# COLOR_TEMPERATURE = "#69b3a2"
# COLOR_PRICE = "#3399e6"
#
# plt.figure(figsize=(10,5))
# plt.subplot(121)
# plt.imshow(distortion.reshape((64, 64)), cmap="inferno")
# plt.xlabel("Block Size - Width", fontsize=16)
# plt.ylabel("Block Size - Height", fontsize=16)
# # plt.subplot(122)
# plt.scatter(distortion[[1,2,64, 65]], rate[[1,2,64, 65]], s=30, color=COLOR_PRICE)
# plt.scatter(distortion, rate, s=0.02, color=COLOR_PRICE)
# plt.plot(distortion_x, [rate[distortion<=d].min() for d in distortion_x], lw=2, color=COLOR_TEMPERATURE)
# plt.xlabel("Distortion", fontsize=16)
# plt.ylabel("Rate", fontsize=16)
# plt.tight_layout()
#
#
#
#





import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

import numpy as np
# agg = np.hstack([np.load("/home/yakir/Data2/additive_deform/deformations/coco_stuff164k/ResNetV1c/noise_estimated_0.npy"),
#                  np.load("/home/yakir/Data2/additive_deform/deformations/coco_stuff164k/ResNetV1c/noise_estimated_1.npy")])
# real = np.hstack([np.load("/home/yakir/Data2/additive_deform/deformations/coco_stuff164k/ResNetV1c/noise_real_0.npy"),
#                   np.load("/home/yakir/Data2/additive_deform/deformations/coco_stuff164k/ResNetV1c/noise_real_1.npy")])

agg = np.hstack([np.load("/home/yakir/Data2/additive_deformation_estimation/deformations/coco_stuff164k/ResNetV1c/noise_estimated_0.npy"),
                 np.load("/home/yakir/Data2/additive_deformation_estimation/deformations/coco_stuff164k/ResNetV1c/noise_estimated_1.npy")])
real = np.hstack([np.load("/home/yakir/Data2/additive_deformation_estimation/deformations/coco_stuff164k/ResNetV1c/noise_real_0.npy"),
                  np.load("/home/yakir/Data2/additive_deformation_estimation/deformations/coco_stuff164k/ResNetV1c/noise_real_1.npy")])


plt.scatter(real, agg, alpha=0.2)
plt.xlabel("Real Distortion")
plt.ylabel("Estimated Distortion")
plt.plot([0,1.], [0,1.])



from tqdm import tqdm
import numpy as np
import time

value = np.random.uniform()
num_relus = 127402496 / 4 / 4 / 10
num_relus = round(num_relus)
weight = 10000
current = np.random.uniform(size=(num_relus,))
last = np.random.uniform(size=(num_relus,))
z = np.random.uniform(size=(num_relus,))

t0 = time.time()
for k in tqdm(range(weight, num_relus)):
    if last[k - weight] > 0:
        current[k] = max(current[k], last[k - weight]  + value)
print(time.time() - t0)














