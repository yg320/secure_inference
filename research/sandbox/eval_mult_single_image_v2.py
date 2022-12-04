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

prevals_orig = []
for batch_indices, data in zip(loader_indices, data_loader):
    image_index = data['img_metas'][0].data[0][0]['ori_filename'].split("_")[-1].split(".jpg")[0]

    file = os.path.join("/home/yakir/tmp/orig", f"{image_index}.npy")
    content = np.load(file)
    prevals_orig.extend(dataset.pre_eval(content, indices=batch_indices))
    # mIoU = dataset.evaluate(preval, **{'metric': ['mIoU']}, logger='silent')['mIoU']
    # results_orig.append(mIoU)
    if batch_indices[0] == num_images - 1:
        break


for cur_dir in directories:

    dataset = build_dataset(test_dataset)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    loader_indices = data_loader.batch_sampler
    prevals_refined = []

    for batch_indices, data in zip(loader_indices, data_loader):
        image_index = data['img_metas'][0].data[0][0]['ori_filename'].split("_")[-1].split(".jpg")[0]
        file = os.path.join(cur_dir, f"{image_index}.npy")
        content = np.load(file)
        prevals_refined.extend(dataset.pre_eval(content, indices=batch_indices))
        # mIoU = dataset.evaluate(preval, **{'metric': ['mIoU']}, logger='silent')['mIoU']
        # results_refined.append(mIoU)
        if batch_indices[0] == num_images - 1:
            break

    x_crf_per_image = [dataset.evaluate([x], **{'metric': ['mIoU']}, logger='silent') for x in prevals_refined]
    x_orig_per_image = [dataset.evaluate([x], **{'metric': ['mIoU']}, logger='silent') for x in prevals_orig]
    x_crf = dataset.evaluate(prevals_refined, **{'metric': ['mIoU']}, logger='silent')
    x_orig = dataset.evaluate(prevals_orig, **{'metric': ['mIoU']}, logger='silent')

    crfs = []
    origs = []
    keys = []
    for k in x_crf.keys():
        if "IoU." in k:
            keys.append(k)
            print(k , x_crf[k], x_orig[k])
            crfs.append(x_crf[k])
            origs.append(x_orig[k])
    plt.scatter(crfs, origs)
    plt.plot([0,1], [0, 1], color="black")
    plt.xlabel("CRF")
    plt.ylabel("Orig")

    print('lkfjdsl')
    # print(np.mean(results_refined)/np.mean(results_orig))
    # plt.scatter(results_refined, results_orig)
    # plt.plot([0,1], [0, 1], color="black")
    # plt.xlabel("CRF")
    # plt.ylabel("Orig")

    # print('jey')
        # image_index = data['img_metas'][0].data[0][0]['ori_filename'].split("_")[-1].split(".jpg")[0]
        #
        # results_refined.extend(dataset.pre_eval(np.load(os.path.join(cur_dir, f"{image_index}.npy")), indices=batch_indices))
        # if batch_indices[0] == num_images - 1:
        #
        #     metric_refined = dataset.evaluate(results_refined, **{'metric': ['mIoU']}, logger='silent')['mIoU']
        #
        #     print(metric_refined/metric_orig)
        #     break