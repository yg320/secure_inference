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
from mmseg.datasets import build_dataset
import os
import numpy as np
image_index = 7
metric_orig = []
metric_refined = []
for image_index in tqdm(range(1, 200)):
    str_image_index = str(image_index).zfill(8)
    test_dataset = {'type': 'ADE20KDataset', 'data_root': 'data/ade/ADEChallengeData2016', 'img_dir': 'images/validation', 'ann_dir': 'annotations/validation', 'pipeline': [{'type': 'LoadImageFromFile'}, {'type': 'MultiScaleFlipAug', 'img_scale': (768, 192), 'flip': False, 'transforms': [{'type': 'Resize', 'keep_ratio': True}, {'type': 'RandomFlip'}, {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, {'type': 'ImageToTensor', 'keys': ['img']}, {'type': 'Collect', 'keys': ['img']}]}], 'test_mode': True}
    ds_test = build_dataset(test_dataset)
    metric_orig.append(ds_test.evaluate(ds_test.pre_eval(np.load(f"/home/yakir/tmp/orig_{str_image_index}.npy"), indices=[image_index - 1]), **{'metric': ['mIoU']})["mIoU"])
    metric_refined.append(ds_test.evaluate(ds_test.pre_eval(np.load(f"/home/yakir/tmp/refined_{str_image_index}.npy"), indices=[image_index - 1]), **{'metric': ['mIoU']})["mIoU"])
# print(metric['mIoU'] / 0.2094)