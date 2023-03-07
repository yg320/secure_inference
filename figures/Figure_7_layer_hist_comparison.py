import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
from research.distortion.utils import get_block_spec_num_relus
spec = '/home/yakir/deepreduce_comparison_v3/distortions/baseline/block_sizes/49.15K.pickle'
config = "/home/yakir/PycharmProjects/secure_inference/work_dirs/benchmark/baseline_49.15K/baseline_49.15K.py"
x = pickle.load(open(spec, 'rb'))
import numpy as np
import mmcv
from research.distortion.parameters.factory import param_factory
green = "#56ae57"
red = "#db5856"
purple = "tab:purple"
blue = "#3399e6"

plt.figure(figsize=(6.4,3.8))

cfg = mmcv.Config.fromfile(config)
params = param_factory(cfg)

LAYER_NAME_TO_DIMS = \
            {
                'stem':       [0, 32, 32],
                'layer1_0_1': [0, 32, 32],
                'layer1_0_2': [0, 32, 32],
                'layer1_1_1': [0, 32, 32],
                'layer1_1_2': [0, 32, 32],
                'layer2_0_1': [0, 16, 16],
                'layer2_0_2': [64, 16, 16],
                'layer2_1_1': [0, 16, 16],
                'layer2_1_2': [64, 16, 16],
                'layer3_0_1': [0, 8, 8],
                'layer3_0_2': [128, 8, 8],
                'layer3_1_1': [0, 8, 8],
                'layer3_1_2': [128, 8, 8],
                'layer4_0_1': [0, 4, 4],
                'layer4_0_2': [0, 4, 4],
                'layer4_1_1': [0, 4, 4],
                'layer4_1_2': [0, 4, 4]
            }
bars = plt.bar(LAYER_NAME_TO_DIMS.keys(), np.array([np.prod(x) for x in LAYER_NAME_TO_DIMS.values()]), color=green, alpha=0.75, label="DeepReDuce")
plt.gca().set_xticklabels(range(1,18), fontsize=13)
# plt.xticks(range(17), LAYER_NAME_TO_DIMS.keys(), rotation='vertical')
for bar in bars:
    bar.set_edgecolor("black")
    bar.set_linewidth(1.5)


reduction = []
for layer_name in x:
    reduction.append(get_block_spec_num_relus({layer_name: x[layer_name]}, params))

bars = plt.bar(range(len(reduction)), reduction, color=blue, alpha=0.75, label="Ours")
for bar in bars:
    bar.set_edgecolor("black")
    bar.set_linewidth(1.5)

plt.gca().set_yticklabels([None, "2", "4", "6", "8", "10", "12", "14", "16"], fontsize=13)
plt.legend(prop={'size': 16})
plt.xlabel("Layer", fontsize=18, labelpad=7)
plt.ylabel("Number of DReLUs (in K)", fontsize=18, labelpad=10)
[i.set_linewidth(1.5) for i in plt.gca().spines.values()]
plt.gca().yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.8)
plt.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.17)

plt.savefig("/home/yakir/Figure_7.png")
# # plt.semilogy()
# spec = "/home/yakir/specs/cls_specs/4x4.pickle"
# config = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py"
#
# plt.gca().tick_params(axis='both', which='major', labelsize=12)
# plt.xlabel("Layer Index", fontsize=14)
# plt.ylabel("Percentage of ReLUs Remained", fontsize=14)
# plt.tight_layout()
# plt.savefig("Figure_10.png")