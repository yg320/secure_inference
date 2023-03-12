import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from mmseg.ops import resize
from research.distortion.utils import get_model
import torch.nn.functional as F
from mmseg.core import intersect_and_union
import torch
from tqdm import tqdm
from mmseg.datasets import build_dataset
import pickle
from research.distortion.arch_utils.factory import arch_utils_factory
import mmcv
from research.distortion.parameters.factory import param_factory
from research.utils import build_data
from research.mmlab_extension.classification.resnet import MyResNet  # TODO: why is this needed?
import numpy as np

config_path = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py"
model_path = "/home/yakir/epoch_25_0.06_imagenet.pth"
relu_spec_file = "/home/yakir/block_specs_aws/imagenet/0.06.pickle"

cfg = mmcv.Config.fromfile(config_path)
params = param_factory(cfg)

dataset = build_data(cfg, mode="test")

# TODO: use shared class with distortion
device = "1"
model = get_model(config=config_path, gpu_id=device, checkpoint_path=model_path)
model = model.eval()
if relu_spec_file is not None:
    layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))
    arch_utils = arch_utils_factory(cfg)
    arch_utils.set_bReLU_layers(model, layer_name_to_block_sizes)

results = {}
sample_ids = list(range(2200)) + list(range(15000,17200)) + list(range(30000, 32200)) + list(range(45000, 45400))
for sample_id in tqdm(sample_ids):
    img = dataset[sample_id]['img'].data.unsqueeze(0).to(f"cuda:{device}")
    gt = dataset.get_gt_labels()[sample_id]
    out = model.forward_test(img)
    results[sample_id] = np.argmax(out[0]) == gt

pickle.dump(obj=results, file=open("/home/yakir/results_imagenet_1.pickle", "wb"))
    # print(sample_id, np.mean(results))
    # if sample_id % 100 == 0:
    #     pickle.dump(obj=results, file=open("/home/yakir/results_imagenet.pickle", "wb"))