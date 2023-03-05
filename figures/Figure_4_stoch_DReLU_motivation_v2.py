import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import numpy as np
from tqdm import tqdm
powers = np.flip(np.arange(64, dtype=np.uint8)[np.newaxis], axis=-1)

stats = True
rand = False
if stats:
    X = []
    Y = []
    Z = []
    for i in tqdm(range(1,231)):
        x = np.load(f"/home/yakir/Data2/secure_activation_statistics_cls/server/{i}.npy")
        y = np.load(f"/home/yakir/Data2/secure_activation_statistics_cls/client/{i}.npy")
        X.append(x.flatten())
        Y.append(y.flatten())
    X = np.hstack(X)
    Y = np.hstack(Y)
    Z = (X + Y) >= 0

    mat = np.zeros(shape=(64, 64))
    for lsb_ignore in tqdm(range(64)):
        for msb_ignore in range(64):
            if msb_ignore == 0 or lsb_ignore == 0:
                Y_ignore = (Y >> lsb_ignore) << lsb_ignore
                X_ignore = (X >> lsb_ignore) << lsb_ignore
                tmp_Z = (X_ignore + Y_ignore).astype(np.uint64)

                Z_approx = (tmp_Z & 2 ** (64 - 1 - msb_ignore)) == 0
                mat[lsb_ignore, msb_ignore] = 1 - (Z_approx == Z).mean()

    plt.figure(figsize=(6, 4))
    plt.plot(mat[0, :], color="#3399e6", lw=4, label="MSBs")
    plt.plot(mat[:, 0], color="#56ae57", lw=4, label="LSBs")
    plt.xlabel("Number of Bits Ignored", fontsize=16, labelpad=8)
    plt.ylabel("Probability of DReLU Error", fontsize=16, labelpad=12)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.legend(prop={'size': 14})
    plt.ylim([0, 0.01])

    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.001))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.0005))

    plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.6)
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.3)

    plt.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.16)
    [i.set_linewidth(1.5) for i in plt.gca().spines.values()]

    plt.savefig("/home/yakir/Figure_4.png")
