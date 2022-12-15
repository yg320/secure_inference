# DenseCRF
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


import numpy as np
import mmcv
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch

def entropy(probs):
    log_probs = np.log(probs)
    entropy = -np.sum(probs * log_probs, axis=0)
    return entropy
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

image_index = 157
str_image_index = str(image_index).zfill(8)
IM_FILE = f'/home/yakir/Data/ade/ADEChallengeData2016/images/validation/ADE_val_{str_image_index}.jpg'
LOGIT_FILE = f'/home/yakir/Data2/logits_160/ADE_val_{str_image_index}.npy'
LOGIT_DOWN_FILE = f'/home/yakir/Data2/logits_160_not_rescaled/ADE_val_{str_image_index}.npy'
ANNOT = f'/home/yakir/Data/ade/ADEChallengeData2016/annotations/validation/ADE_val_{str_image_index}.png'

logits = np.load(LOGIT_FILE)[0]
# logits_down = np.load(LOGIT_DOWN_FILE)[0]
img = mmcv.imread(IM_FILE)[..., ::-1]
annot = mmcv.imread(ANNOT)
# x = np.random.randint(low=0, high=100000000, size=logits_down.shape[1:])
# logits_2 = F.interpolate(torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(torch.float64), annot.shape[:2], scale_factor=None, mode='bilinear', align_corners=False)[0].numpy()

ax = plt.subplot(221)
plt.imshow(img)
plt.subplot(222, sharex=ax, sharey=ax)
plt.imshow(annot[...,0])
plt.subplot(223, sharex=ax, sharey=ax)
plt.imshow(logits.argmax(axis=0) + 1)
plt.subplot(224, sharex=ax, sharey=ax)
plt.imshow(entropy(softmax(logits)))