import argparse
import copy
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import pickle
import mmcv

from research.distortion.parameters.factory import param_factory
from research.distortion.distortion_utils import DistortionUtils, get_block_spec_num_relus, get_num_relus
from research.distortion.utils import get_channel_order_statistics, get_num_of_channels
from research.distortion.utils import get_channels_subset
from research.mmlab_extension.classification.resnet import MyResNet  # TODO: why is this needed?
from research.distortion.arch_utils.factory import arch_utils_factory
from mmcls.datasets import build_dataloader
import torch
from research.distortion.utils import get_model, get_data
from research.utils import build_data

class SimulatedAnnealingHandler:
    def __init__(self, device_ids, params, cfg, input_block_spec_path, output_block_spec_path, checkpoint_path):
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
            'dist': False,
            'round_up': True,
            'seed': 1563879445,
            'sampler_cfg': None,
            'samples_per_gpu': 64,
            'workers_per_gpu': 4
        }

        self.data_loader = build_dataloader(self.dataset, **train_loader_cfg)

        self.keys = ["Noise", "Signal"]
        self.block_size_spec = pickle.load(open(input_block_spec_path, "rb"))
        self.output_block_spec_path = output_block_spec_path

        self.channel_order_to_layer, self.channel_order_to_channel, self.channel_order_to_dim = get_channel_order_statistics(self.params)
        self.num_channels = get_num_of_channels(self.params)
        self.dim_to_channels = {dim: np.argwhere(self.channel_order_to_dim == dim)[:, 0] for dim in np.unique(self.channel_order_to_dim)}

        self.flipped = 0
        self.arch_utils = arch_utils_factory(self.cfg)

        self.model.train()
        self.model.cuda()

        self.replicas = torch.nn.parallel.replicate(self.model, self.device_ids)

    def get_sibling_channels(self):
        random_channel_a = np.random.choice(self.num_channels)
        channels_b = self.dim_to_channels[self.channel_order_to_dim[random_channel_a]]
        random_channel_b = np.random.choice(channels_b)
        return random_channel_a, random_channel_b

    def get_suggested_block_size(self):

        while True:
            suggest_block_size_spec = copy.deepcopy(self.block_size_spec)
            sibling_channel_a, sibling_channel_b = self.get_sibling_channels()
            layer_name_a = self.channel_order_to_layer[sibling_channel_a]
            layer_name_b = self.channel_order_to_layer[sibling_channel_b]
            channel_a = self.channel_order_to_channel[sibling_channel_a]
            channel_b = self.channel_order_to_channel[sibling_channel_b]

            if not np.all(suggest_block_size_spec[layer_name_a][channel_a] == suggest_block_size_spec[layer_name_b][channel_b]):

                tmp = suggest_block_size_spec[layer_name_a][channel_a]
                suggest_block_size_spec[layer_name_a][channel_a] = suggest_block_size_spec[layer_name_b][channel_b]
                suggest_block_size_spec[layer_name_b][channel_b] = tmp
                return suggest_block_size_spec

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

        steps = []
        distorted_losses = []
        baseline_losses = []
        iteration = 0

        while True:
            for _, data in tqdm(enumerate(self.data_loader)):
                iteration += 1
                print(iteration)
                torch.cuda.empty_cache()

                suggest_block_size_spec = self.get_suggested_block_size()

                batch = torch.nn.parallel.scatter(data['img'], self.device_ids)
                gt = torch.nn.parallel.scatter(data['gt_label'], self.device_ids)
                kwargs_tup = tuple(dict(return_loss=True, gt_label=gt[i]) for i in range(len(self.device_ids)))

                baseline_loss = self.get_loss(batch, kwargs_tup, self.block_size_spec)
                distorted_loss = self.get_loss(batch, kwargs_tup, suggest_block_size_spec)

                if (distorted_loss / baseline_loss) < 0.994:
                    steps.append(iteration)
                    distorted_losses.append(distorted_loss)
                    baseline_losses.append(baseline_loss)
                    self.flipped += 1
                    self.block_size_spec = suggest_block_size_spec

                    if self.flipped % 10 == 0:
                        pickle.dump(obj=self.block_size_spec, file=open(self.output_block_spec_path, "wb"))
                        pickle.dump(obj=(steps, distorted_losses, baseline_losses), file=open("/storage/yakir/secure_inference/data_2.pickle", "wb"))



if __name__ == "__main__":
    checkpoint_path = "/home/yakir/epoch_14.pth"
    input_block_spec_path = "/home/yakir/block_size_spec_4x4_algo.pickle"
    output_block_spec_path = "/home/yakir/block_size_spec_4x4_algo_out.pickle"
    config = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"
    device_ids = [0, 1]
    #
    # checkpoint_path = "./outputs/classification/resnet50_8xb32_in1k/finetune_0.0001_avg_pool/epoch_14.pth"
    # input_block_spec_path = "./relu_spec_files/classification/resnet50_8xb32_in1k/iterative/num_iters_1/iter_0/block_size_spec_4x4_algo.pickle"
    # output_block_spec_path = "./relu_spec_files/classification/resnet50_8xb32_in1k/iterative/num_iters_1/iter_0/block_size_spec_4x4_algo_simulated_annealing_v6.pickle"
    # config = "/storage/yakir/secure_inference/research/configs/classification/resnet/iterative/iter01_algo4x4_0.001_4_baseline.py"
    # device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

    cfg = mmcv.Config.fromfile(config)
    params = param_factory(cfg)

    with torch.no_grad():
        SimulatedAnnealingHandler(device_ids=device_ids,
                                  params=params,
                                  cfg=cfg,
                                  input_block_spec_path=input_block_spec_path,
                                  output_block_spec_path=output_block_spec_path,
                                  checkpoint_path=checkpoint_path).extract_deformation_channel_ord()