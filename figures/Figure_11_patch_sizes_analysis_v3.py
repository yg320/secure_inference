import pickle
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np
content_9 = pickle.load(open('/home/yakir/assets/mobilenet_ade/block_spec/0.09.pickle', 'rb'))

block_sizes_9 = np.vstack([content_9[k] for k in content_9.keys()])
BLOCK_SIZES = np.array(list(set(tuple(x) for x in block_sizes_9)))

r_9 = []
for block_size in BLOCK_SIZES:
    r_9.append(np.all(block_sizes_9 == block_size, axis=1).sum())

r_9 = np.array(r_9)
top_block_r_9 = BLOCK_SIZES[np.argsort(r_9)[::-1][:10]]
values = r_9[np.argsort(r_9)[::-1][:15]]

index = np.argsort(top_block_r_9[:,0] * top_block_r_9[:,1])
top_blocks = top_block_r_9[index]
values = values[index]



block_names = [f'{x[0]}x{x[1]}' for x in top_blocks]
assert block_names[0] == '0x1'
assert block_names[1] == '1x1'
block_names[0] = "Identity"
block_names[1] = "1x1"
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

plt.xlabel("Patch Size", fontsize=16, labelpad=-5)
plt.ylabel("# Channels", fontsize=16, labelpad=12)
plt.gca().yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.8)
[i.set_linewidth(1.7) for i in plt.gca().spines.values()]

plt.subplots_adjust(bottom=0.21, top=0.97, left=0.12, right=0.99)
plt.savefig("/home/yakir/Figure_11.png")
