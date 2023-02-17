import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
from research.distortion.utils import get_block_spec_num_relus
spec = "/home/yakir/tesnet18/distortions/resnet18_2xb64_cifar100/block_sizes/14.33K.pickle"
config = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/benchmark/resnet18_2xb64_cifar100_ft_14.3K_relus.py"
x = pickle.load(open(spec, 'rb'))
import numpy as np
import mmcv
from research.distortion.parameters.factory import param_factory

plt.figure()

cfg = mmcv.Config.fromfile(config)
params = param_factory(cfg)

reduction = []
for layer_name in x:
    reduction.append(get_block_spec_num_relus({layer_name: x[layer_name]}, params))

bars = plt.bar(range(len(reduction)), reduction, color="#3399e6")
for bar in bars:
    bar.set_edgecolor("black")
    bar.set_linewidth(1.2)
spec = "/home/yakir/specs/cls_specs/4x4.pickle"
config = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py"

plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Layer Index", fontsize=14)
plt.ylabel("Percentage of ReLUs Remained", fontsize=14)
plt.tight_layout()
plt.savefig("Figure_10.png")