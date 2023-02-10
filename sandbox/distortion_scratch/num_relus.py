import pickle
import mmcv
import numpy as np

from research.distortion.parameters.factory import param_factory
from research.distortion.utils import get_block_spec_num_relus

# config_path = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"
# block_size_spec_file = "/home/yakir/knap_base_dim_annel_dis_ext/relu_spec_files/0.pickle"
config_path = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/finetune_models/resnet18_2xb64_cifar100_12.3K_relus_lr_0.01.py"
block_size_spec_file = "/home/yakir/distortion_200/block_size/12.3K.pickle"
cfg = mmcv.Config.fromfile(config_path)
params = param_factory(cfg)
block_size_spec = pickle.load(open(block_size_spec_file, 'rb'))
# for k in block_size_spec.keys():
#     block_size_spec[k][:, 0] = 7
#     block_size_spec[k][:, 1] = 7

print(get_block_spec_num_relus(block_size_spec, params))

# Classification
# 1x1 9608704
# 1x2 4834816
# 2x2 2434816
# 2x3 1701120
# 3x3 1189632
# 3x4 874240
# 4x4 644224


# Segmentation
# 1x1 85262848
# 1x2 42631680
# 2x2 21316096
# 2x3 14580224
# 3x3 9973840
# 3x4 7290368
# 4x4 5329408
# 4x5 4330240
# 5x5 3518416
# 5x6 2968880
# 6x6 2505328
# 6x7 2251616
# 7x7 2024560

