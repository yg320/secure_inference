import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
powers = np.flip(np.arange(64, dtype=np.uint8)[np.newaxis], axis=-1)
value = []

counter = 0
num_neg = 0
for i in tqdm(range(1,9201)):
    x = np.load(f"/home/yakir/Data2/secure_activation_statistics/server/{i}.npy")
    y = np.load(f"/home/yakir/Data2/secure_activation_statistics/client/{i}.npy")
    z = x + y
    z = z.flatten()

    # value.append(z)
    counter += len(z)
    num_neg += (z < 0).sum()

value = np.hstack(value)
value = value.reshape(-1, 1)

value_bits = np.right_shift(value, powers)
value_bits = np.bitwise_and(value_bits, 1)

l = [np.mean(value_bits[:, 0] == value_bits[:, i]) for i in tqdm(range(64))]




rr = [value_bits[:,1:-i].all(axis=1).mean() * (2**i - 1) / 2**(1+i) for i in range(0, 63)]
start = 44
for end in range(start, 64):
    print(end, (value_bits[:, start:end].sum(axis=1) == end - start).mean())

lsb_err_prob = [(value_bits[:, 44:-i].sum(axis=1) == 20 - i).mean() * ((2**i - 1) / 2**(1+i)) for i in range(1, 20)]
plt.plot()
end = 50
(value_bits[:, start:end].sum(axis=1) == end - start).mean()

plt.figure(figsize=(5, 4))

plt.plot([1-x for x in l], color="#3399e6", lw=3,  label="MSBs")
# plt.subplot(122)
plt.plot(rr,  color="#69b3a2", lw=3,label="LSBs")
plt.xlabel("Number of Bits Ignored", fontsize=13, labelpad=7)
plt.ylabel("Probability of DReLU Error", fontsize=13, labelpad=7)
plt.legend()
plt.tight_layout()
plt.savefig("/home/yakir/Figure_4_a.png")
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()
plt.text(0.25, 0.5, r'DReLU stats', fontsize=25)
plt.savefig("/home/yakir/Figure_4.png")


plt.imshow(np.arange(64).reshape(1, -1), cmap="gray")
plt.tight_layout()