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

num_images = 150
directories = [dir_ for dir_ in glob.glob("/home/yakir/tmp_6/*") if "orig" not in dir_]
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

results_orig = []
for batch_indices, data in zip(loader_indices, data_loader):
    image_index = data['img_metas'][0].data[0][0]['ori_filename'].split("_")[-1].split(".jpg")[0]
    results_orig.extend(dataset.pre_eval(np.load(os.path.join("/home/yakir/tmp_4/orig", f"{image_index}.npy")), indices=batch_indices))
    if batch_indices[0] == num_images - 1:
        metric_orig = dataset.evaluate(results_orig, **{'metric': ['mIoU']}, logger='silent')['mIoU']
        break

for cur_dir in directories:

    dataset = build_dataset(test_dataset)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    loader_indices = data_loader.batch_sampler
    results_refined = []

    for batch_indices, data in zip(loader_indices, data_loader):
        image_index = data['img_metas'][0].data[0][0]['ori_filename'].split("_")[-1].split(".jpg")[0]

        results_refined.extend(dataset.pre_eval(np.load(os.path.join(cur_dir, f"{image_index}.npy")), indices=batch_indices))
        if batch_indices[0] == num_images - 1:

            metric_refined = dataset.evaluate(results_refined, **{'metric': ['mIoU']}, logger='silent')['mIoU']

            print(metric_refined/metric_orig)
            break