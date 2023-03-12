import matplotlib
matplotlib.use("TkAgg")

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

def center_crop(tensor, size):
    if tensor.shape[1] < size or tensor.shape[2] < size:
        raise ValueError(tensor.shape)
    h = (tensor.shape[1] - size) // 2
    w = (tensor.shape[2] - size) // 2
    return tensor[:, h:h + size, w:w + size]

activations = [[] for _ in range(40)]
NUM_SAMPLES = 5
CORR_SIZE = 32
for i in range(NUM_SAMPLES*40):
    try:
        activations[i % 40].append(center_crop(np.load("/home/yakir/Data2/tmp_relus/relu_{}.npy".format(i))[0], 32))
    except ValueError:
        continue

del(activations[34])

activations = np.concatenate(activations, axis=1)

activations = activations > 0

# m = np.zeros(shape=(CORR_SIZE, CORR_SIZE))
# for i_space in range(CORR_SIZE):
#     for j_space in tqdm(range(CORR_SIZE), desc="i_space={}".format(i_space)):
#
#         l = []
#         for i in range(32 - i_space):
#             for j in range(32 - j_space):
#                 l.append(np.mean(activations[:,:,i,j] == activations[:,:,i+i_space,j+j_space]))
#         m[i_space, j_space] = np.mean(l)
#
# r = np.concatenate([m[::-1, ::-1], m[::-1][:,1:]], axis=1)
# r = np.concatenate([r, r[::-1][1:]], axis=0)
# np.save("/home/yakir/Data2/tmp_relus_processed.npy", r)
r = np.load("/home/yakir/Data2/tmp_relus_processed.npy")
plt.figure(figsize=(4, 3))
CROP = 21
plt.imshow(r[CROP:-CROP,CROP:-CROP], vmin=0.7, vmax=0.9, cmap="Blues")
plt.xticks(np.arange(0, r.shape[0] - 2*CROP, 5))
plt.yticks(np.arange(0, r.shape[0] - 2*CROP, 5))
plt.gca().set_yticklabels(np.arange(-31+CROP, 32-CROP, 5), fontsize=15)
plt.gca().set_xticklabels(np.arange(-31+CROP, 32-CROP, 5), fontsize=15)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.tight_layout()
[i.set_linewidth(1.5) for i in plt.gca().spines.values()]
cbar.outline.set_linewidth(1.5)


plt.savefig("/home/yakir/Figure_2.png")

plt.imshow()