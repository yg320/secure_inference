import argparse
import copy
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import pickle
import mmcv

from research.distortion.parameters.factory import param_factory
import glob
from research.mmlab_extension.classification.resnet import MyResNet  # TODO: why is this needed?
from research.distortion.arch_utils.factory import arch_utils_factory
from mmcls.datasets import build_dataloader
import torch
from research.distortion.utils import get_model, get_data
from research.utils import build_data
import time
import multiprocessing

class SimpleTest:
    def __init__(self, device_ids, params, cfg, block_size_spec_files):
        self.device_ids = device_ids
        self.params = params
        self.cfg = cfg

        self.model = get_model(
            config=self.cfg,
            gpu_id=None,
            checkpoint_path=checkpoint_path
        )

        self.dataset = build_data(self.cfg, train=True)

        train_loader_cfg = {
            'num_gpus': len(self.device_ids),
            'dist': True,
            'round_up': True,
            'seed': 1563879445,
            'sampler_cfg': None,
            'samples_per_gpu': 256 // len(self.device_ids),
            'workers_per_gpu': multiprocessing.cpu_count() // len(self.device_ids)
        }

        self.data_loader = build_dataloader(self.dataset, **train_loader_cfg)

        self.arch_utils = arch_utils_factory(self.cfg)

        self.model.train()
        self.model.cuda()

        self.replicas = torch.nn.parallel.replicate(self.model, self.device_ids)

        self.block_size_specs = [pickle.load(open(block_size_spec_file, 'rb')) for block_size_spec_file in block_size_spec_files]
        # for block_size_spec in self.block_size_specs:
        #     for layer_name in block_size_spec:
        #         block_size_spec[layer_name][:,0] = 4
        #         block_size_spec[layer_name][:,1] = 4
        #         np.random.shuffle(block_size_spec[layer_name])

    def get_loss(self, batch, kwargs_tup, block_size_spec):
        for replica in self.replicas:
            self.arch_utils.set_bReLU_layers(replica, block_size_spec)
        outputs = torch.nn.parallel.parallel_apply(modules=self.replicas,
                                                   inputs=batch,
                                                   kwargs_tup=kwargs_tup,
                                                   devices=self.device_ids)

        loss = sum([float(x['loss'].cpu()) for x in outputs]) / len(self.device_ids)

        return loss

    def extract_deformation_channel_ord(self):

        losses = np.zeros((len(self.block_size_specs), 2*512))
        for data_index, data in enumerate(self.data_loader):
            torch.cuda.empty_cache()
            batch = torch.nn.parallel.scatter(data['img'], self.device_ids)
            gt = torch.nn.parallel.scatter(data['gt_label'], self.device_ids)
            kwargs_tup = tuple(dict(return_loss=True, gt_label=gt[i]) for i in range(len(self.device_ids)))

            for block_size_spec_index, block_size_spec in enumerate(self.block_size_specs):

                loss = self.get_loss(batch, kwargs_tup, block_size_spec)
                losses[block_size_spec_index, data_index] = loss

            out = np.mean(losses[:,:data_index+1], axis=1)
            print("======= {} =======".format(data_index))
            for i in range(len(out)):
                print(f'{i}: {out[i]}')



if __name__ == "__main__":
    checkpoint_path = "/home/yakir/epoch_14_avg_pool.pth"
    config = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"
    device_ids = [0, 1]

    cfg = mmcv.Config.fromfile(config)
    params = param_factory(cfg)
    block_size_spec_files = glob.glob("/home/yakir/PycharmProjects/secure_inference/with_noise/snr_0.00/*.pickle")[:1]
    block_size_spec_files += glob.glob("/home/yakir/with_noise_v2/snr_100/*.pickle")[:2]
    block_size_spec_files += glob.glob("/home/yakir/with_noise_v2/snr_250/*.pickle")[:2]
    block_size_spec_files += glob.glob("/home/yakir/with_noise_v2/snr_1000/*.pickle")[:2]
    block_size_spec_files += glob.glob("/home/yakir/with_noise/snr_2500/*.pickle")[:2]
    block_size_spec_files += glob.glob("/home/yakir/with_noise/snr_5000/*.pickle")[:2]
    block_size_spec_files += glob.glob("/home/yakir/with_noise/snr_10000/*.pickle")[:2]
    with torch.no_grad():
        SimpleTest(device_ids=device_ids,
                   params=params,
                   cfg=cfg,
                   block_size_spec_files=block_size_spec_files).extract_deformation_channel_ord()
