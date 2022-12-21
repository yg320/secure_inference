import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from mmseg.ops import resize
from research.distortion.utils import get_model
from research.distortion.utils import get_data
import torch.nn.functional as F
from research.distortion.utils import center_crop
from mmseg.core import intersect_and_union
import torch
from tqdm import tqdm
from mmseg.datasets import build_dataset

config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py"
model_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth"

cfg = {'type': 'ADE20KDataset',
       'data_root': 'data/ade/ADEChallengeData2016',
       'img_dir': 'images/validation',
       'ann_dir': 'annotations/validation',
       'pipeline': [
           {'type': 'LoadImageFromFile'},
           {'type': 'LoadAnnotations', 'reduce_zero_label': True},
           {'type': 'Resize', 'img_scale': (1024, 256), 'keep_ratio': True},
           {'type': 'RandomFlip', 'prob': 0.0},
           {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True},
           {'type': 'DefaultFormatBundle'},
           {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}]
       }

dataset = build_dataset(cfg)
# TODO: use shared class with distortion
device = "cuda:0"
model_baseline = get_model(config=config_path, gpu_id=None, checkpoint_path=model_path)
results = []
for sample_id in tqdm(range(20)):
    img = dataset[sample_id]['img'].data.unsqueeze(0)[:,:,:256,:256]
    img_meta = dataset[sample_id]['img_metas'].data

    img_shape = img_meta['img_shape']
    img_meta['img_shape'] = (256, 256, 3)
    seg_map = dataset.get_gt_seg_map_by_idx(sample_id)
    seg_map = seg_map[:min(seg_map.shape), :min(seg_map.shape)]
    img_meta['ori_shape'] = (seg_map.shape[0], seg_map.shape[1], 3)
    out = model_baseline.decode_head(model_baseline.backbone(img)).to("cpu")

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

print(dataset.evaluate(results, logger='silent', **{'metric': ['mIoU']})['mIoU'])
