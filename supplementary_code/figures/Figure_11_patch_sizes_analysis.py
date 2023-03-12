import pickle
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np
content_18 = pickle.load(open('/home/yakir/assets/mobilenet_ade/block_spec/0.18.pickle', 'rb'))
content_9 = pickle.load(open('/home/yakir/assets/mobilenet_ade/block_spec/0.09.pickle', 'rb'))

layer_names = [
    'layer3_0_1',
    'layer3_1_0',
    'layer3_1_1',
    'layer3_2_0',
    'layer3_2_1',
    'layer4_0_0',
    'layer4_0_1',
    'layer4_1_0',
    'layer4_1_1',
    'layer4_2_0',
    'layer4_2_1',
    'layer4_3_0',
    'layer4_3_1',
    'layer5_0_0',
    'layer5_0_1',
    'layer5_1_0',
    'layer5_1_1',
    'layer5_2_0',
    'layer5_2_1',
    'layer6_0_0',
    'layer6_0_1',
    'layer6_1_0',
    'layer6_1_1',
    'layer6_2_0',
    'layer6_2_1',
    'layer7_0_0',
    'layer7_0_1',
    'decode_1',
    'decode_2',
    'decode_3',
    'decode_4',
    'decode_5',
]
block_sizes_18 = np.vstack([content_18[k] for k in layer_names])
block_sizes_9 = np.vstack([content_9[k] for k in layer_names])
BLOCK_SIZES_64x64 = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [1, 3],
    [3, 1],
    [1, 4],
    [2, 2],
    [4, 1],
    [1, 5],
    [5, 1],
    [1, 6],
    [2, 3],
    [3, 2],
    [6, 1],
    [2, 4],
    [4, 2],
    [3, 3],
    [2, 5],
    [5, 2],
    [2, 6],
    [3, 4],
    [4, 3],
    [6, 2],
    [3, 5],
    [5, 3],
    [2, 8],
    [4, 4],
    [8, 2],
    [3, 6],
    [6, 3],
    [4, 5],
    [5, 4],
    [2, 12],
    [3, 8],
    [4, 6],
    [6, 4],
    [8, 3],
    [12, 2],
    [5, 5],
    [3, 9],
    [9, 3],
    [5, 6],
    [6, 5],
    [2, 16],
    [4, 8],
    [8, 4],
    [16, 2],
    [3, 12],
    [6, 6],
    [12, 3],
    [5, 8],
    [8, 5],
    [4, 12],
    [6, 8],
    [8, 6],
    [12, 4],
    [7, 7],
    [5, 10],
    [10, 5],
    [6, 9],
    [9, 6],
    [5, 12],
    [6, 10],
    [10, 6],
    [12, 5],
    [4, 16],
    [8, 8],
    [16, 4],
    [6, 12],
    [12, 6],
    [5, 15],
    [15, 5],
    [8, 10],
    [10, 8],
    [9, 9],
    [6, 16],
    [8, 12],
    [12, 8],
    [16, 6],
    [10, 10],
    [9, 12],
    [12, 9],
    [8, 14],
    [14, 8],
    [11, 11],
    [8, 16],
    [16, 8],
    [12, 12],
    [10, 16],
    [16, 10],
    [12, 14],
    [14, 12],
    [13, 13],
    [12, 16],
    [16, 12],
    [14, 14],
    [16, 16],
    [32, 32],
    [64, 64],
    [0, 1],
    [1, 0]
])

r_18 = []
r_9 = []
for block_size in BLOCK_SIZES_64x64:
    r_18.append(np.all(block_sizes_18 == block_size, axis=1).sum())
    r_9.append(np.all(block_sizes_9 == block_size, axis=1).sum())

r_18 = np.array(r_18)
r_9 = np.array(r_9)

top_block_r_9 = BLOCK_SIZES_64x64[np.argsort(r_9)[::-1][:100]]
top_block_r_18 = BLOCK_SIZES_64x64[np.argsort(r_18)[::-1][:100]]

a = list(set(tuple(x) for x in top_block_r_9).union(set(tuple(x) for x in top_block_r_18)))
top_blocks = np.array(sorted(a, key=lambda x: x[0] * x[1]))

r_18 = []
r_9 = []
for block_size in top_blocks:
    r_18.append(np.all(block_sizes_18 == block_size, axis=1).sum())
    r_9.append(np.all(block_sizes_9 == block_size, axis=1).sum())

block_names = [f'{x[0]}x{x[1]}' for x in top_blocks]
assert block_names[0] == '0x1'
block_names[0] = "Identity"
block_names.append("Other")
r_9.append(block_sizes_9.shape[0] - sum(r_9))
r_18.append(block_sizes_18.shape[0] - sum(r_18))

bars = plt.bar(block_names, r_9, color="#3399e6", alpha=0.75, label="DeepReDuce")
for bar in bars:
    bar.set_edgecolor("black")
    bar.set_linewidth(1.2)

bars = plt.bar(block_names, r_18, color="#69b3a2", alpha=0.75, label="DeepReDuce")
for bar in bars:
    bar.set_edgecolor("black")
    bar.set_linewidth(1.2)



print('fds')