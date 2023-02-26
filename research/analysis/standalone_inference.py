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

config_path = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"
model_path = "/home/yakir/epoch_14_avg_pool.pth"
relu_spec_file = None #"/home/yakir/3x4.pickle"
# model_path = "/home/yakir/epoch_50.pth"
# relu_spec_file = "/home/yakir/4x4.pickle"
cfg = mmcv.Config.fromfile(config_path)
params = param_factory(cfg)

dataset = build_data(cfg, mode="test")

# TODO: use shared class with distortion
device = "cuda:0"
model = get_model(config=config_path, gpu_id=None, checkpoint_path=model_path)
model = model.eval()
if relu_spec_file is not None:
    layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))
    arch_utils = arch_utils_factory(cfg)
    arch_utils.set_bReLU_layers(model, layer_name_to_block_sizes)

results = []
for sample_id in tqdm(range(500)):
    img = dataset[sample_id]['img'].data.unsqueeze(0)
    gt = dataset.get_gt_labels()[sample_id]
    out = model.forward_test(img)
    results.append(np.argmax(out[0]) == gt)

    print(np.mean(results))
    # img_meta = dataset[sample_id]['img_metas'].data
    #
    # img_shape = img_meta['img_shape']
    # img_meta['img_shape'] = (256, 256, 3)
    # seg_map = dataset.get_gt_seg_map_by_idx(sample_id)
    # seg_map = seg_map[:min(seg_map.shape), :min(seg_map.shape)]
    # img_meta['ori_shape'] = (seg_map.shape[0], seg_map.shape[1], 3)
    # out = model.decode_head(model.backbone(img)).to("cpu")
    #
    # out = resize(
    #     input=out,
    #     size=img.shape[2:],
    #     mode='bilinear',
    #     align_corners=False)
    #
    # resize_shape = img_meta['img_shape'][:2]
    # seg_logit = out[:, :, :resize_shape[0], :resize_shape[1]]
    # size = img_meta['ori_shape'][:2]
    #
    # seg_logit = resize(
    #     seg_logit,
    #     size=size,
    #     mode='bilinear',
    #     align_corners=False,
    #     warning=False)
    #
    # output = F.softmax(seg_logit, dim=1)
    # seg_pred = output.argmax(dim=1)
    # seg_pred = seg_pred.cpu().numpy()[0]
    #
    # results.append(
    #     intersect_and_union(
    #         seg_pred,
    #         seg_map,
    #         len(dataset.CLASSES),
    #         dataset.ignore_index,
    #         label_map=dict(),
    #         reduce_zero_label=dataset.reduce_zero_label)
    # )
    # if sample_id % 10 == 0:
    #     print(dataset.evaluate(results, logger='silent', **{'metric': ['mIoU']})['mIoU'])


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

CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k_relu.py"
MODEL_PATH = "/home/yakir/assets/mobilenet_ade/models/ratio_0.09.pth"
RELU_SPEC_FILE = '/home/yakir/assets/mobilenet_ade/block_spec/0.09.pickle'

cfg = mmcv.Config.fromfile(CONFIG_PATH)
params = param_factory(cfg)

dataset = build_data(cfg, mode="test")

# TODO: use shared class with distortion
device = "0"
model = get_model(
    config=cfg,
    gpu_id=device,
    checkpoint_path=MODEL_PATH
)
model = model.eval()
if RELU_SPEC_FILE is not None:
    layer_name_to_block_sizes = pickle.load(open(RELU_SPEC_FILE, 'rb'))
    arch_utils = arch_utils_factory(cfg)
    arch_utils.set_bReLU_layers(model, layer_name_to_block_sizes)

results = []
for sample_id in tqdm(range(87)):
    img = dataset[sample_id]['img'][0].data.unsqueeze(0)
    img_meta = dataset[sample_id]['img_metas'][0].data
    seg_map = dataset.get_gt_seg_map_by_idx(sample_id)

    out = model.decode_head(model.backbone(img.to("cuda:0")))

    out = resize(
        input=out,
        size=img.shape[2:],
        mode='bilinear',
        align_corners=False)

    resize_shape = img_meta['img_shape'][:2]
    seg_logit = out[:, :, :resize_shape[0], :resize_shape[1]]
    size = img_meta['ori_shape'][:2]

    seg_logit = resize(
        seg_logit,
        size=size,
        mode='bilinear',
        align_corners=False,
        warning=False)

    output = F.softmax(seg_logit, dim=1)
    seg_pred = output.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()[0]

    results.append(
        intersect_and_union(
            seg_pred,
            seg_map,
            len(dataset.CLASSES),
            dataset.ignore_index,
            label_map=dict(),
            reduce_zero_label=dataset.reduce_zero_label)
    )

    print(sample_id, dataset.evaluate(results, logger='silent', **{'metric': ['mIoU']})['mIoU'])
