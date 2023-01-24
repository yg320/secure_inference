
import numpy as np
import pickle
import mmcv

from tqdm import tqdm
from research.distortion.parameters.factory import param_factory
import glob
from research.mmlab_extension.classification.resnet import MyResNet  # TODO: why is this needed?
from research.distortion.arch_utils.factory import arch_utils_factory
from mmcls.datasets import build_dataloader
import torch
from research.distortion.utils import get_model, get_data
from research.utils import build_data
import multiprocessing
import os


class SimpleTest:
    def __init__(self, device_ids, params, cfg):
        self.device_ids = device_ids
        self.params = params
        self.cfg = cfg

        self.model = get_model(
            config=self.cfg,
            gpu_id=None,
            checkpoint_path=checkpoint_path
        )

        self.train_loader_cfg = {
            'num_gpus': len(self.device_ids),
            'dist': True,
            'round_up': True,
            'seed': 1563879445,
            'sampler_cfg': None,
            'samples_per_gpu': 256 // len(self.device_ids),
            'workers_per_gpu': multiprocessing.cpu_count() // len(self.device_ids)
        }


        self.arch_utils = arch_utils_factory(self.cfg)

        self.model.train()
        self.model.cuda()

        self.replicas = torch.nn.parallel.replicate(self.model, self.device_ids)

    def get_loss(self, batch, kwargs_tup, block_size_spec):
        for replica in self.replicas:
            self.arch_utils.set_bReLU_layers(replica, block_size_spec)
        outputs = torch.nn.parallel.parallel_apply(modules=self.replicas,
                                                   inputs=batch,
                                                   kwargs_tup=kwargs_tup,
                                                   devices=self.device_ids)

        loss = sum([float(x['loss'].cpu()) for x in outputs]) / len(self.device_ids)

        return loss

    def get_block_size_spec_loss(self, block_size_spec_path):

        dataset = build_data(self.cfg, train=True)
        data_loader = build_dataloader(dataset, **self.train_loader_cfg)

        block_size_spec = pickle.load(open(block_size_spec_path, 'rb'))
        losses = []

        for data_index, data in enumerate(data_loader):
            torch.cuda.empty_cache()
            batch = torch.nn.parallel.scatter(data['img'], self.device_ids)

            gt = torch.nn.parallel.scatter(data['gt_label'], self.device_ids)
            kwargs_tup = tuple(dict(return_loss=True, gt_label=gt[i]) for i in range(len(self.device_ids)))

            loss = self.get_loss(batch, kwargs_tup, block_size_spec)
            losses.append(loss)

            if data_index == 64:
                break

        return np.mean(losses)


if __name__ == "__main__":
    checkpoint_path = "/home/yakir/epoch_14_avg_pool.pth"
    config_path = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"
    channel_distortion_path = "/home/yakir/iter_0_collected"
    snr = "20000"
    PYTHON_PATH_EXPORT = 'export PYTHONPATH=\"${PYTHONPATH}:/home/yakir/PycharmProjects/secure_inference\"; '
    block_size_spec_file_format = "/home/yakir/relu_spec_files_test/{seed}.pickle"
    snr_seed_file = "/home/yakir/snr_seed_file.pickle"
    knap_script = "/home/yakir/PycharmProjects/secure_inference/research/knapsack/multiple_choice_knapsack_v2.py"
    device_ids = [0, 1]

    cfg = mmcv.Config.fromfile(config_path)
    params = param_factory(cfg)

    simple_test = SimpleTest(device_ids=device_ids, params=params, cfg=cfg)

    loss = np.inf
    seeds_to_use = []
    pickle.dump(obj=seeds_to_use, file=open(snr_seed_file, 'wb'))

    for seed in range(100000):
        seeds_to_use.append(seed)
        pickle.dump(obj=seeds_to_use, file=open(snr_seed_file, 'wb'))

        block_size_spec_file = block_size_spec_file_format.format(seed=seed)

        if seed > 0:
            os.system(PYTHON_PATH_EXPORT +
                      f'python {knap_script} '
                      f'--block_size_spec_file_name {block_size_spec_file} '
                      f'--channel_distortion_path {channel_distortion_path} '
                      f'--config {config_path} '
                      '--cost_type ReLU '
                      '--division 1 '
                      '--cur_iter 0 '
                      '--num_iters 1 '
                      '--max_cost 644224 '
                      f'--target_snr {snr} '
                      f'--snr_seed_file {snr_seed_file} > /dev/null 2>&1') #

        block_size_spec = pickle.load(open(file=block_size_spec_file, mode='rb'))
        cur_loss = simple_test.get_block_size_spec_loss(block_size_spec_path=block_size_spec_file)
        if cur_loss < loss:
            loss = cur_loss
            pickle.dump(obj=seeds_to_use, file=open(file=snr_seed_file, mode='wb'))
        else:
            seeds_to_use.pop()
        print(f"seed: {seed}, loss: {loss}, cur_loss {cur_loss}")
