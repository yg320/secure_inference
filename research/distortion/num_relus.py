import pickle
import mmcv
import numpy as np

from research.distortion.parameters.factory import param_factory
from research.distortion.distortion_utils import get_block_spec_num_relus

config_path = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"
block_size_spec_file = "/home/yakir/knap_base_dim_annel_dis_ext/relu_spec_files/0.pickle"
cfg = mmcv.Config.fromfile(config_path)
params = param_factory(cfg)
block_size_spec = pickle.load(open(block_size_spec_file, 'rb'))
for k in block_size_spec.keys():
    block_size_spec[k][:, 0] = 4
    block_size_spec[k][:, 1] = 4

print(get_block_spec_num_relus(block_size_spec, params))

# 1x1 9608704
# 1x2 4834816
# 2x2 2434816
# 2x3 1701120
# 3x3 1189632
# 3x4 874240
# 4x4 644224

