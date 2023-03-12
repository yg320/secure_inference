import pickle
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np



import cv2
import os
import glob

imgs, scores, classes = pickle.load(open('/home/yakir/imnet_results/test.pkl', 'rb'))

img = imgs[0]
score = scores[0]
argsort = np.argsort(score)[::-1]
score = score[argsort[:5]]
cur_classes = [classes[i] for i in argsort[:5]]
plt.subplot(121)
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.bar(cur_classes, score)
plt.xticks(rotation='vertical')