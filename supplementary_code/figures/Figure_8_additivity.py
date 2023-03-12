import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# plt.rcParams['text.usetex'] = True

import pickle
import numpy as np

plt.figure(figsize=(6.4,3.5))
content = pickle.load(open("/home/yakir/distortion_additivity.pickle", "rb"))
plt.scatter(content["additive_noises"][:3000], content["noises"][:3000], facecolors=matplotlib.colors.to_rgba('#3399e6', 0.7), edgecolors=matplotlib.colors.to_rgba("black", 0.4))
plt.ylim(7, 56)
plt.xlim(11, 16)
plt.xlabel("Additive Distortion", fontsize=16)
plt.ylabel("Real Distortion", fontsize=16)
plt.gca().tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
[i.set_linewidth(1.5) for i in plt.gca().spines.values()]

plt.savefig("/home/yakir/Figure_8.png")
print(np.corrcoef(content["additive_noises"][:3000], content["noises"][:3000])[0,1])

# plt.title("Correlation between additive noise and noise level: 0.99", fontsize=13)
# plt.subplot(131)
# plt.title(np.corrcoef(content["additive_noises"], content["noises"])[0, 1])
# plt.subplot(132)
# plt.scatter(content["additive_noises"], content["losses"])
# plt.title(np.corrcoef(content["additive_noises"], content["losses"])[0, 1])
# plt.subplot(133)
# plt.scatter(content["noises"], content["losses"])
# plt.title(np.corrcoef(content["noises"], content["losses"])[0, 1])
