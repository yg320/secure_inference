import pickle
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np

def center_crop(tensor, size):
    if tensor.shape[0] < size or tensor.shape[1] < size:
        raise ValueError(tensor.shape)
    h = (tensor.shape[0] - size) // 2
    w = (tensor.shape[1] - size) // 2
    return tensor[h:h + size, w:w + size]

import cv2
import os
import glob

im_names = [
    "2007_000033.jpg",
    "2007_000042.jpg",
    "2007_000061.jpg",
    "2007_000123.jpg",
    "2007_000129.jpg",
    "2007_000175.jpg",
    "2007_000187.jpg",
    "2007_000323.jpg",
    "2007_000332.jpg",
    "2007_000346.jpg",
    "2007_000452.jpg",
    "2007_000464.jpg",
    "2007_000491.jpg",
    "2007_000529.jpg",
    "2007_000559.jpg",
    "2007_000572.jpg",
    "2007_000629.jpg",
    "2007_000636.jpg",
    "2007_000661.jpg",
    "2007_000663.jpg",
]
dirs = [
    "plots_baseline",
    # "plots_0.18",
    "plots_0.15",
    # "plots_0.12",
    "plots_0.09",
    # "plots_0.06",
    "plots_0.03"
]
dir_to_name = {
    "plots_baseline": "100%",
    "plots_0.18": "18%",
    "plots_0.15": "15%",
    "plots_0.12": "12%",
    "plots_0.09": "9%",
    "plots_0.06": "6%",
    "plots_0.03": "3%",
}
fig = plt.figure(figsize=(2*4, 3*len(im_names)))
im_index = 0
for im_name in im_names:

    for dir_ in dirs:
        im_index += 1
        plt.subplot(len(im_names) // 2, 2 * len(dirs), im_index)
        if im_index <= 2 * len(dirs):
            plt.title(dir_to_name[dir_], fontsize=16)
        plt.xticks([])
        plt.yticks([])
        im = cv2.imread(os.path.join("/home/yakir/voc_plots", dir_, im_name))
        if im.shape[0] > im.shape[1]:
            new_scale = 512, int(im.shape[0] * 512 / im.shape[1])
        else:
            new_scale = int(im.shape[1] * 512 / im.shape[0]), 512
        im = cv2.resize(im, new_scale)
        im = center_crop(im, 512)
        plt.imshow(im)

plt.tight_layout()
plt.savefig("/home/yakir/gallary.png")