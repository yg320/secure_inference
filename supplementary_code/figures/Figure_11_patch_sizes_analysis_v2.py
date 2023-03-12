import pickle
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np
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


r_9 = []
for block_size in BLOCK_SIZES_64x64:
    r_9.append(np.all(block_sizes_9 == block_size, axis=1).sum())

r_9 = np.array(r_9)
top_block_r_9 = BLOCK_SIZES_64x64[np.argsort(r_9)[::-1][:10]]
values = r_9[np.argsort(r_9)[::-1][:15]]

index = np.argsort(top_block_r_9[:,0] * top_block_r_9[:,1])
top_blocks = top_block_r_9[index]
values = values[index]



block_names = [f'{x[0]}x{x[1]}' for x in top_blocks]
assert block_names[0] == '0x1'
block_names[0] = "Identity"
block_names.append("Other")
values = list(values)
values.append(sum(r_9) - sum(values))

plt.figure(figsize=(6.4, 4.8))

bars = plt.bar(block_names, values, color="#3399e6", alpha=0.75, label="DeepReDuce")
plt.xticks(range(len(block_names)), block_names, rotation='vertical', fontsize=14)
plt.gca().set_yticklabels([str(x) + "K" for x in range( 9)], fontsize=14)

for bar in bars:
    bar.set_edgecolor("black")
    bar.set_linewidth(1.5)

plt.xlabel("Block Size", fontsize=16, labelpad=-5)
plt.ylabel("# Channels", fontsize=16, labelpad=12)
plt.gca().yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)

plt.subplots_adjust(bottom=0.21, top=0.97, left=0.12, right=0.99)
plt.savefig("/home/yakir/Figure_11.png")
