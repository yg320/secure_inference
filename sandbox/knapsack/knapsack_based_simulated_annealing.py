
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

        self.baseline_model = get_model(
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

        self.baseline_model.head.loss = lambda x, y: {"last_activation_layer": x}
        self.model.head.loss = lambda x, y: {"last_activation_layer": x}

        self.baseline_model.train()
        self.baseline_model.cuda()

        self.model.train()
        self.model.cuda()

        self.replicas = torch.nn.parallel.replicate(self.model, self.device_ids)
        self.replicas_baseline = torch.nn.parallel.replicate(self.baseline_model, self.device_ids)

    def get_loss(self, batch, kwargs_tup, block_size_spec):
        for replica in self.replicas:
            self.arch_utils.set_bReLU_layers(replica, block_size_spec)

        outputs_baseline = torch.nn.parallel.parallel_apply(modules=self.replicas_baseline,
                                                            inputs=batch,
                                                            kwargs_tup=kwargs_tup,
                                                            devices=self.device_ids)
        torch.cuda.empty_cache()
        outputs = torch.nn.parallel.parallel_apply(modules=self.replicas,
                                                   inputs=batch,
                                                   kwargs_tup=kwargs_tup,
                                                   devices=self.device_ids)
        outputs = np.stack([x['last_activation_layer'].cpu().detach().numpy() for x in outputs]).reshape(len(self.device_ids), -1)
        outputs_baseline = np.stack([x['last_activation_layer'].cpu().detach().numpy() for x in outputs_baseline]).reshape(len(self.device_ids), -1)
        distortion = ((outputs - outputs_baseline) ** 2).mean()
        return distortion

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

# from research.distortion.utils import get_channel_order_statistics
#
# def add_noise_to_distortion(source_dir, target_dir, snr_interval, seed, params):
#     assert os.path.exists(source_dir)
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#     channel_order_to_layer, channel_order_to_channel, channel_order_to_dim = get_channel_order_statistics(params)
#     num_channels = len(channel_order_to_dim)
#
#
#     random_generator = np.random.default_rng(seed=seed)
#     channels = random_generator.choice(num_channels, size=5)
#     # snr = random_generator.uniform(low=snr_interval[0], high=snr_interval[1])
#     # print(f"snr: {snr}")
#     for layer_name in params.LAYER_NAMES:
#         distortion = np.load(os.path.join(source_dir, f"{layer_name}.npy"))
#
#         channels_to_use = [channel_order_to_channel[channel] for channel in channels if channel_order_to_layer[channel] == layer_name]
#         for channel in channels_to_use:
#             if seed > 0:
#                 noise = random_generator.uniform(low=0.7, high=1.3)
#             else:
#                 noise = 1
#             distortion[channel] = distortion[channel] * noise
#
#         np.save(os.path.join(target_dir, layer_name), distortion)

def add_noise_to_distortion(source_dir, target_dir, snr_interval, seed, params):
    assert os.path.exists(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    random_generator = np.random.default_rng(seed=seed)
    # snr = random_generator.uniform(low=snr_interval[0], high=snr_interval[1])
    # print(f"snr: {snr}")
    for layer_name in params.LAYER_NAMES:
        distortion = np.load(os.path.join(source_dir, f"{layer_name}.npy"))
        if seed > 0:
            noise = random_generator.normal(loc=0,
                                            scale=1000.0,
                                            size=distortion.shape)
        else:
            noise = 0
        np.save(os.path.join(target_dir, layer_name), np.zeros_like(distortion))


if __name__ == "__main__":
    # export PYTHONPATH=/storage/yakir/secure_inference; python research/knapsack/knapsack_based_simulated_annealing.py
    snr_interval = [150000, 200000]

    # out_stat_dir = "/output_3/knap_base_dim_annel_100000"
    # checkpoint_path = "./outputs/classification/resnet50_8xb32_in1k/finetune_0.0001_avg_pool/epoch_14.pth"
    # config_path = "/storage/yakir/secure_inference/research/configs/classification/resnet/knapsack.py"
    # optimal_channel_distortion_path = "outputs/distortions/classification/resnet50_8xb32_in1k_iterative/num_iters_1/iter_0_collected/"
    # PYTHON_PATH_EXPORT = 'export PYTHONPATH=\"${PYTHONPATH}:/storage/yakir/secure_inference\"; '
    # knap_script = "research/knapsack/multiple_choice_knapsack_v2.py"
    # device_ids = [0, 1]
    # batch_size = 256
    # num_batches = 64
    #
    out_stat_dir = "/home/yakir/simulated_annealing_knapsack/v5_tst2s2"
    checkpoint_path = "/home/yakir/epoch_14_avg_pool.pth"
    config_path = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"
    optimal_channel_distortion_path = "/home/yakir/iter_0_collected"
    PYTHON_PATH_EXPORT = 'export PYTHONPATH=\"${PYTHONPATH}:/home/yakir/PycharmProjects/secure_inference\"; '
    knap_script = "/home/yakir/PycharmProjects/secure_inference/research/knapsack/multiple_choice_knapsack_v2.py"
    device_ids = [0, 1]
    batch_size = 128
    num_batches = 128

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

        # for device in device_ids:
        #     seed = len(device_ids) * seed_group + device
        #     iter_seeds.append(seed)
        #     cur_iter_dir = os.path.join(noised_channel_distortion_base_path, str(seed))
        #     add_noise_to_distortion(source_dir=optimal_channel_distortion_path,
        #                             target_dir=cur_iter_dir,
        #                             snr_interval=snr_interval,
        #                             seed=seed,
        #                             params=params)
        #
        #     block_size_spec_file = os.path.join(block_size_spec_files_dir, f"{seed}.pickle")
        #     block_size_spec_paths.append(block_size_spec_file)
        #     os.system(PYTHON_PATH_EXPORT +
        #               f'python {knap_script} '
        #               f'--block_size_spec_file_name {block_size_spec_file} '
        #               f'--channel_distortion_path {cur_iter_dir} '
        #               f'--config {config_path} '
        #               '--cost_type ReLU '
        #               '--division 1 '
        #               '--cur_iter 0 '
        #               '--num_iters 1 '
        #               '--max_cost 644224 '
        #               f'--device {device} > /dev/null 2>&1 & ')
        #
        # while True:
        #     if all([os.path.exists(x) for x in block_size_spec_paths]):
        #         break
        #     else:
        #         time.sleep(1)

        t1 = time.time()
        print(f"Knapsack took {t1 - t0} seconds")
        block_size_spec_paths = ["/home/yakir/3x3_naive.pickle", "/home/yakir/3x3.pickle"]
        cur_losses = simple_test.get_block_size_spec_loss(block_size_spec_paths=block_size_spec_paths)

        t2 = time.time()
        print(f"Loss took {t2 - t1} seconds")
        print(cur_losses)

        min_loss = np.min(cur_losses)
        argmin_loss = np.argmin(cur_losses)
        # if 50*(min_loss-loss) < np.abs(np.random.normal()):
        if min_loss < loss:
            loss = min_loss
            seeds_to_use.append(iter_seeds[argmin_loss])
            optimal_channel_distortion_path = os.path.join(noised_channel_distortion_base_path, str(seeds_to_use[-1]))

        all_cur_losses.append(cur_losses)
        print(loss)
        print(seeds_to_use)
        all_losses.append(loss)
        pickle.dump(obj=(seeds_to_use, all_losses, all_cur_losses), file=open(stats_out_path, 'wb'))