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
for i in range(4*40):
    try:
        activations[i % 40].append(center_crop(np.load("/home/yakir/tmp_relus/relu_{}.npy".format(i))[0], 32))
    except ValueError:
        continue

del(activations[34])

activations = np.concatenate(activations, axis=1)

activations = activations > 0

m = np.zeros(shape=(12, 12))
for i_space in tqdm(range(12)):
    for j_space in range(12):

        l = []
        for i in range(32 - i_space):
            for j in range(32 - j_space):
                l.append(np.mean(activations[:,:,i,j] == activations[:,:,i+i_space,j+j_space]))
        m[i_space, j_space] = np.mean(l)

r = np.concatenate([m[::-1, ::-1], m[::-1][:,1:]], axis=1)
r = np.concatenate([r, r[::-1][1:]], axis=0)
plt.imshow(r, vmin=0.0, vmax=0.1, cmap="Blues")
plt.xticks(np.arange(0, 23, 2))
plt.yticks(np.arange(0, 23, 2))
plt.gca().set_yticklabels(np.arange(-11, 12, 2), fontsize=13)
plt.gca().set_xticklabels(np.arange(-11, 12, 2)[::-1], fontsize=13)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=13)
plt.tight_layout()
plt.savefig("/home/yakir/Figure_2.png")

plt.imshow()