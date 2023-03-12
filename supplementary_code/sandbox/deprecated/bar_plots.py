import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from research.parameters.base import ParamsFactory
from research.distortion.distortion_utils import get_num_relus
import pickle
import numpy as np

params = ParamsFactory()("MobileNetV2_256_Params_2_Groups")

relu_spec_file = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity/block_spec_32.pickle"
# relu_spec_file = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity/block_spec_relu_reduction.pickle"
layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))

layers_num_relus_baseline = []
layers_num_relus = []
layer_names = []
for layer_name in params.LAYER_NAMES:
    if layer_name in layer_name_to_block_sizes:
        block_sizes = layer_name_to_block_sizes[layer_name]
        activation_dim = params.LAYER_NAME_TO_DIMS[layer_name][1]
        layer_num_relus = 0
        layer_num_relus_baseline = 0

        for block_size in block_sizes:
            layer_num_relus += get_num_relus(tuple(block_size), activation_dim=activation_dim)
            layer_num_relus_baseline += get_num_relus((1, 1), activation_dim=activation_dim)

        layers_num_relus_baseline.append(layer_num_relus_baseline)
        layers_num_relus.append(layer_num_relus)
        layer_names.append(layer_name)
    else:
        layers_num_relus_baseline.append(np.prod(params.LAYER_NAME_TO_DIMS[layer_name]))
        layers_num_relus.append(np.prod(params.LAYER_NAME_TO_DIMS[layer_name]))
        layer_names.append(layer_name)
layers_num_relus_baseline = np.array(layers_num_relus_baseline)
layers_num_relus = np.array(layers_num_relus)
layer_names = np.array(layer_names)
np.sum(layers_num_relus) / np.sum(layers_num_relus_baseline)
plt.bar(layer_names, layers_num_relus/layers_num_relus_baseline, alpha=0.5)