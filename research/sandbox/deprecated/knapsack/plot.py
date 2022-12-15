import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pickle
import torch
import numpy as np

from research.block_relu.params import ParamsFactory, MobileNetV2_256_Params

params = MobileNetV2_256_Params()
params.DATASET = "ade_20k"
params.CONFIG = "/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/deeplabv3_m-v2-d8_256x256_160k_ade20k.py"
params.CHECKPOINT = "/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_m-v2-d8_256x256_160k_ade20k/iter_160000.pth"


# baseline_block_size_spec = pickle.load(open('/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_iter_0123_0.0833.pickle', 'rb'))
# baseline_block_size_spec = pickle.load(open('/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_3x4.pickle', 'rb'))
baseline_block_size_spec = pickle.load(open('/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_2_groups_iter_01_0.0833.pickle', 'rb'))

def get_block_index_to_num_relus(block_sizes, layer_dim):
    block_index_to_num_relus = []
    for block_size_index, block_size in enumerate(block_sizes):
        avg_pool = torch.nn.AvgPool2d(
            kernel_size=tuple(block_size),
            stride=tuple(block_size), ceil_mode=True)

        cur_input = torch.zeros(size=(1, 1, layer_dim, layer_dim))
        cur_relu_map = avg_pool(cur_input)
        num_relus = cur_relu_map.shape[2] * cur_relu_map.shape[3]
        block_index_to_num_relus.append(num_relus)
    W = np.array(block_index_to_num_relus)
    return W

layer_names = [x for x in params.LAYER_NAMES if x in baseline_block_size_spec.keys()]
layer_ratios = []
for layer_name in layer_names:
    spec = np.array(baseline_block_size_spec[layer_name])
    b_sizes = np.unique(spec, axis=0) #np.array(params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])
    x = get_block_index_to_num_relus(b_sizes, params.LAYER_NAME_TO_DIMS[layer_name][1])
    layer_ratio = sum(x[np.array([np.argwhere(np.all(y == b_sizes, axis=1))[0,0] for y in spec])])/np.prod(params.LAYER_NAME_TO_DIMS[layer_name])
    layer_ratios.append(layer_ratio)

sum([np.array(layer_ratios)[layer_index] * np.prod(params.LAYER_NAME_TO_DIMS[layer_name]) for layer_index, layer_name in enumerate(baseline_block_size_spec.keys())])/sum([np.prod(params.LAYER_NAME_TO_DIMS[layer_name]) for layer_index, layer_name in enumerate(params.LAYER_NAMES)])
plt.bar(range(len(layer_ratios)) ,np.array(layer_ratios))