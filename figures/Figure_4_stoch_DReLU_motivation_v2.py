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
    plt.plot(mat[:, 0], color="#69b3a2", lw=4, label="LSBs")
    plt.xlabel("Number of Bits Ignored", fontsize=16, labelpad=8)
    plt.ylabel("Probability of DReLU Error", fontsize=16, labelpad=12)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.legend(prop={'size': 12})
    plt.ylim([0, 0.01])

    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.001))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.0005))

    plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    plt.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.16)

    plt.savefig("/home/yakir/Figure_4.png")

    plt.figure(figsize=(20, 15))
    plt.plot(mat[0, :], color="#3399e6", lw=8, label="MSBs")
    plt.plot(mat[:, 0], color="#69b3a2", lw=8, label="LSBs")
    plt.xlabel("Number of Bits Ignored", fontsize=32, labelpad=32)
    plt.ylabel("Probability of DReLU Error", fontsize=32, labelpad=32)
    plt.gca().tick_params(axis='both', which='major', labelsize=32)
    plt.legend(prop={'size': 32})
    plt.ylim([0, 0.05])
    plt.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.15)

    plt.savefig("/home/yakir/Figure_4.png")
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [3, 1]})

    # plt.subplot(211)
    # plt.imshow(mat[:8,:48], vmin=0.0, vmax=0.01, cmap="Blues")
    # plt.ylabel("LSBs Ignored", fontsize=13, labelpad=7)
    # plt.xlabel("MSBs Ignored", fontsize=13, labelpad=7)
    # plt.colorbar()


    axs[0].plot(mat[0, :], color="#3399e6", lw=3, label="MSBs")
    axs[0].plot(mat[:, 0], color="#69b3a2", lw=3, label="LSBs")
    # plt.subplot(212)
    axs[1].plot(mat[0, :], color="#3399e6", lw=3, label="MSBs")
    axs[1].plot(mat[:, 0], color="#69b3a2", lw=3, label="LSBs")
    plt.ylim([0,0.001])
    plt.xlabel("Number of Bits Ignored", fontsize=13, labelpad=7)
    plt.ylabel("Probability of DReLU Error", fontsize=13, labelpad=7)
    plt.legend()
    plt.subplots_adjust(left=0.11, right=0.88, top=0.98, bottom=0.07,
                        hspace=0.35)  # , right=0.99, top=0.98, bottom=0.1, hspace=0.02, wspace=0.15)

    plt.tight_layout()
    zzz = []
    for i in tqdm(range(64)):
        Y_ignore = (Y >> i) << i
        X_ignore = (X >> i) << i
        Z_approx = (X_ignore + Y_ignore) >= 0
        Z_approx = ((X + Y).astype(np.uint64) & 2 ** 63) == 0
        zzz.append(1 - (Z_approx == Z).mean())
    plt.plot(range(64), zzz)  # statistics

if rand:
    X = np.random.randint(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max, size=(10000000,), dtype=np.int64)
    Y = np.random.randint(low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max, size=(10000000,), dtype=np.int64)
    Z = (X + Y) >= 0
    zzz = []
    for i in range(64):
        Y_ignore = (Y >> i) << i
        X_ignore = (X >> i) << i
        Z_approx = (X_ignore + Y_ignore) >= 0
        zzz.append(1 - (Z_approx == Z).mean())

    plt.plot(range(64), zzz)  # statistics
    plt.plot([0] + [1/2**(64-i)-1/2**64 for i in range(1, 64)])  # theoretical


#
# # Now run a similar analysis, but theorectically this time
# plt.plot(range(64), zzz)  # statistics
# plt.plot([1/2**(64-i-1) for i in range(64)])  # theoretical
#
# plt.plot([1/2**(64-i) - 1/2**64 for i in range(64)])
# plt.plot([1/2**(64-i+1) for i in range(64)])
#
# Xs = np.hstack(Xs)
# Xs = Xs.reshape(-1, 1)
# Ys = np.hstack(Ys)
# Ys = Ys.reshape(-1, 1)
# Zs = np.hstack(Zs)
# Zs = Zs.reshape(-1, 1)
#
# zzz = []
# for i in range(64):
#     Ys_ignore = (Ys >> i) << i
#     Xs_ignore = (Xs >> i) << i
#     zzz.append(1 - (((Xs_ignore + Ys_ignore) > 0) == (Zs > 0)).mean())
# plt.plot(zzz)
#
#
# # Xs_bits = np.right_shift(Xs, powers)
# # Xs_bits = np.bitwise_and(Xs_bits, 1)
# # Ys_bits = np.right_shift(Ys, powers)
# # Ys_bits = np.bitwise_and(Ys_bits, 1)
# # Zs_bits = np.right_shift(Zs, powers)
# # Zs_bits = np.bitwise_and(Zs_bits, 1)
# #
# # print('fsd')
# # #
# # # l = [np.mean(value_bits[:, 0] == value_bits[:, i]) for i in tqdm(range(64))]
# # # rr = [value_bits[:,1:-i].all(axis=1).mean() * (2**i - 1) / 2**(1+i) for i in range(0, 63)]
# #
# # # start = 44
# # # for end in range(start, 64):
# # #     print(end, (value_bits[:, start:end].sum(axis=1) == end - start).mean())
# # #
# # # lsb_err_prob = [(value_bits[:, 44:-i].sum(axis=1) == 20 - i).mean() * ((2**i - 1) / 2**(1+i)) for i in range(1, 20)]
# # # plt.plot()
# # # end = 50
# # # (value_bits[:, start:end].sum(axis=1) == end - start).mean()
# #
# # plt.figure(figsize=(5, 4))
# #
# # plt.plot([1-x for x in l], color="#3399e6", lw=3,  label="MSBs")
# # # plt.subplot(122)
# # plt.plot(rr,  color="#69b3a2", lw=3,label="LSBs")
# # plt.xlabel("Number of Bits Ignored", fontsize=13, labelpad=7)
# # plt.ylabel("Probability of DReLU Error", fontsize=13, labelpad=7)
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig("/home/yakir/Figure_4_a.png")
# # # plt.xticks([])
# # # plt.yticks([])
# # # plt.tight_layout()
# # plt.text(0.25, 0.5, r'DReLU stats', fontsize=25)
# # plt.savefig("/home/yakir/Figure_4.png")
# #
# #
# # plt.imshow(np.arange(64).reshape(1, -1), cmap="gray")
# # plt.tight_layout()