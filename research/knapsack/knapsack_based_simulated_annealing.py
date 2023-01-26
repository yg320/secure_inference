
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
import time

class SimpleTest:
    def __init__(self, device_ids, params, cfg, batch_size, num_batches):
        self.device_ids = device_ids
        self.params = params
        self.cfg = cfg
        self.num_batches = num_batches
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
            'samples_per_gpu': batch_size // len(self.device_ids),
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

    def get_block_size_spec_loss(self, block_size_spec_paths):

        dataset = build_data(self.cfg, mode="distortion_extraction_val")
        data_loader = build_dataloader(dataset, **self.train_loader_cfg)

        block_size_specs = [pickle.load(open(block_size_spec_path, 'rb')) for block_size_spec_path in block_size_spec_paths]
        losses = []

        for data_index, data in enumerate(data_loader):
            torch.cuda.empty_cache()
            batch = torch.nn.parallel.scatter(data['img'], self.device_ids)

            gt = torch.nn.parallel.scatter(data['gt_label'], self.device_ids)
            kwargs_tup = tuple(dict(return_loss=True, gt_label=gt[i]) for i in range(len(self.device_ids)))
            cur_losses = []
            for block_size_spec in block_size_specs:
                loss = self.get_loss(batch, kwargs_tup, block_size_spec)
                cur_losses.append(loss)
            losses.append(cur_losses)

            if data_index == self.num_batches:
                break

        return np.array(losses).mean(axis=0)


def add_noise_to_distortion(source_dir, target_dir, snr, seed, params):
    assert os.path.exists(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    random_generator = np.random.default_rng(seed=seed)

    for layer_name in params.LAYER_NAMES:
        distortion = np.load(os.path.join(source_dir, f"{layer_name}.npy"))

        noise = np.sqrt(1 / snr) * random_generator.normal(loc=0,
                                                           scale=distortion.std(),
                                                           size=distortion.shape)

        np.save(os.path.join(target_dir, layer_name), distortion + noise)

if __name__ == "__main__":
    # export PYTHONPATH=/storage/yakir/secure_inference; python research/knapsack/knapsack_based_simulated_annealing.py
    snr = 100000

    out_stat_dir = "/output_3/knap_base_dim_annel_100000"
    checkpoint_path = "./outputs/classification/resnet50_8xb32_in1k/finetune_0.0001_avg_pool/epoch_14.pth"
    config_path = "/storage/yakir/secure_inference/research/configs/classification/resnet/knapsack.py"
    optimal_channel_distortion_path = "outputs/distortions/classification/resnet50_8xb32_in1k_iterative/num_iters_1/iter_0_collected/"
    PYTHON_PATH_EXPORT = 'export PYTHONPATH=\"${PYTHONPATH}:/storage/yakir/secure_inference\"; '
    knap_script = "research/knapsack/multiple_choice_knapsack_v2.py"
    device_ids = [0, 1]
    batch_size = 256
    num_batches = 64
    #
    # out_stat_dir = "/home/yakir/knap_base_dim_annel_dis_ext"
    # checkpoint_path = "/home/yakir/epoch_14_avg_pool.pth"
    # config_path = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"
    # optimal_channel_distortion_path = "/home/yakir/iter_0_collected"
    # PYTHON_PATH_EXPORT = 'export PYTHONPATH=\"${PYTHONPATH}:/home/yakir/PycharmProjects/secure_inference\"; '
    # knap_script = "/home/yakir/PycharmProjects/secure_inference/research/knapsack/multiple_choice_knapsack_v2.py"
    # device_ids = [0, 1]
    # batch_size = 256
    # num_batches = 64

    stats_out_path = os.path.join(out_stat_dir, "stats_out_path.pickle")
    noised_channel_distortion_base_path = os.path.join(out_stat_dir, "iter_0_collected_with_noise")
    block_size_spec_files_dir = os.path.join(out_stat_dir, "relu_spec_files")

    os.makedirs(block_size_spec_files_dir)
    os.makedirs(noised_channel_distortion_base_path)

    cfg = mmcv.Config.fromfile(config_path)
    params = param_factory(cfg)

    simple_test = SimpleTest(device_ids=device_ids, params=params, cfg=cfg, batch_size=batch_size, num_batches=num_batches)
    loss = np.inf

    all_losses = []
    seeds_to_use = []
    all_cur_losses = []

    for seed_group in range(100000):
        print('==================================')
        block_size_spec_paths = []
        iter_seeds = []

        t0 = time.time()

        for device in device_ids:
            seed = len(device_ids) * seed_group + device
            iter_seeds.append(seed)
            cur_iter_dir = os.path.join(noised_channel_distortion_base_path, str(seed))
            add_noise_to_distortion(source_dir=optimal_channel_distortion_path,
                                    target_dir=cur_iter_dir,
                                    snr=snr,
                                    seed=seed,
                                    params=params)

            block_size_spec_file = os.path.join(block_size_spec_files_dir, f"{seed}.pickle")
            block_size_spec_paths.append(block_size_spec_file)
            os.system(PYTHON_PATH_EXPORT +
                      f'python {knap_script} '
                      f'--block_size_spec_file_name {block_size_spec_file} '
                      f'--channel_distortion_path {cur_iter_dir} '
                      f'--config {config_path} '
                      '--cost_type ReLU '
                      '--division 1 '
                      '--cur_iter 0 '
                      '--num_iters 1 '
                      '--max_cost 644224 '
                      f'--device {device} > /dev/null 2>&1 & ')

        while True:
            if all([os.path.exists(x) for x in block_size_spec_paths]):
                break
            else:
                time.sleep(1)

        t1 = time.time()
        print(f"Knapsack took {t1 - t0} seconds")
        cur_losses = simple_test.get_block_size_spec_loss(block_size_spec_paths=block_size_spec_paths)

        t2 = time.time()
        print(f"Loss took {t2 - t1} seconds")
        print(cur_losses)

        min_loss = np.min(cur_losses)
        argmin_loss = np.argmin(cur_losses)
        if min_loss < loss:
            loss = min_loss
            optimal_device = argmin_loss
            seeds_to_use.append(iter_seeds[optimal_device])
            optimal_channel_distortion_path = os.path.join(noised_channel_distortion_base_path, str(seeds_to_use[-1]))

        all_cur_losses.append(cur_losses)
        print(loss)
        print(seeds_to_use)
        all_losses.append(loss)
        pickle.dump(obj=(seeds_to_use, all_losses, all_cur_losses), file=open(stats_out_path, 'wb'))
