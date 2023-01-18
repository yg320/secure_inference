import pickle
import os
from tqdm import tqdm
import numpy as np

from research.distortion.parameters.classification.resent.resnet50_8xb32_in1k import Params
from research.distortion.utils import get_channels_subset, get_channel_order_statistics

params = Params()
channels_to_run, _ = get_channels_subset(seed=123, params=params, cur_iter=0, num_iters=3)

base_path = "/storage/yakir/secure_inference/outputs/distortions/classification/resnet50_8xb32_in1k_iterative/num_iters_1/iter_0/"
new_path = "/storage/yakir/secure_inference/outputs/distortions/classification/resnet50_8xb32_in1k_iterative/num_iters_3/iter_0/"
os.makedirs(new_path)
for layer_name in tqdm(params.LAYER_NAMES):
    for batch_index in range(8):
        cur_stats = pickle.load(open(os.path.join(base_path, f"{layer_name}_{batch_index}.pickle"), 'rb'))
        new_stats = {k: np.zeros_like(v) for k, v in cur_stats.items()}
        for channel_index in channels_to_run[layer_name]:
            for k in cur_stats.keys():
                new_stats[k][channel_index] = cur_stats[k][channel_index]

        pickle.dump(obj=new_stats, file=open(os.path.join(new_path, f"{layer_name}_{batch_index}.pickle"), 'wb'))