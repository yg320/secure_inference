import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import shutil
import sys
from tqdm import tqdm
import pickle
import glob

import mmcv
from mmseg.datasets import build_dataset
import os
import numpy as np


import mmcv
from mmseg.datasets import build_dataloader, build_dataset

import os
import numpy as np
import glob

num_images = 400
directories = [dir_ for dir_ in glob.glob("/home/yakir/tmp_3/*") if "orig" not in dir_]
test_loader_cfg = {'num_gpus': 1, 'dist': False, 'shuffle': False, 'samples_per_gpu': 1, 'workers_per_gpu': 8}
test_dataset = {'type': 'ADE20KDataset', 'data_root': 'data/ade/ADEChallengeData2016', 'img_dir': 'images/validation',
                    'ann_dir': 'annotations/validation', 'pipeline': [{'type': 'LoadImageFromFile'},
                                                                      {'type': 'MultiScaleFlipAug', 'img_scale': (768, 192),
                                                                       'flip': False, 'transforms':
                                                                           [{'type': 'Resize', 'keep_ratio': True},
                                                                            {'type': 'RandomFlip'}, {'type': 'Normalize',
                                                                                                     'mean': [123.675,
                                                                                                              116.28,
                                                                                                              103.53],
                                                                                                     'std': [58.395, 57.12,
                                                                                                             57.375],
                                                                                                     'to_rgb': True},
                                                                            {'type': 'ImageToTensor', 'keys': ['img']},
                                                                            {'type': 'Collect', 'keys': ['img']}]}],
                    'test_mode': True}

dataset = build_dataset(test_dataset)
data_loader = build_dataloader(dataset, **test_loader_cfg)
loader_indices = data_loader.batch_sampler


batch_indices = [307]
image_index = str(batch_indices[0] + 1).zfill(8)
IM_FILE = f'/home/yakir/Data/ade/ADEChallengeData2016/images/validation/ADE_val_{image_index}.jpg'
img = mmcv.imread(IM_FILE)[..., ::-1]
content_crf = np.load(f'/home/yakir/tmp_3/75.702_7.5776_7.4594/{image_index}.npy')
content_orig = np.load(f'/home/yakir/tmp/orig/{image_index}.npy')

x_crf = dataset.evaluate(dataset.pre_eval(content_crf, indices=batch_indices), **{'metric': ['mIoU']}, logger='silent')
x_orig = dataset.evaluate(dataset.pre_eval(content_orig, indices=batch_indices), **{'metric': ['mIoU']}, logger='silent')

x_crf = {k: v for k, v in x_crf.items() if not np.isnan(v) and "IoU." in k}
x_orig = {k: v for k, v in x_orig.items() if not np.isnan(v) and "IoU." in k}

for k, v in x_crf.items():
    print(k, "crf=", v, "orig=", x_orig[k])
# for k, v in x_orig.items():
#     print(k,v)

plt.subplot(131)

plt.imshow(img)
plt.subplot(132)
plt.imshow(content_crf)
plt.subplot(133)
plt.imshow(content_orig)
