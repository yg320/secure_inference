import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pickle
import numpy as np
content = pickle.load(open("/home/yakir/distortion_additivity.pickle", "rb"))
plt.figure()
plt.subplot(131)
plt.scatter(content["additive_noises"], content["noises"])
plt.title(np.corrcoef(content["additive_noises"], content["noises"])[0, 1])
plt.subplot(132)
plt.scatter(content["additive_noises"], content["losses"])
plt.title(np.corrcoef(content["additive_noises"], content["losses"])[0, 1])
plt.subplot(133)
plt.scatter(content["noises"], content["losses"])
plt.title(np.corrcoef(content["noises"], content["losses"])[0, 1])
