from research.secure_inference_3pc.backend import backend
from tqdm import tqdm
import argparse
import os

from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.base import get_assets, TypeConverter

from research.secure_inference_3pc.model_securifier import get_secure_model, init_prf_fetcher
from research.secure_inference_3pc.const import CLIENT, SERVER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, DUMMY_RELU, PRF_PREFETCH
from mmseg.ops import resize


import torch.nn.functional as F
from mmseg.core import intersect_and_union
from research.secure_inference_3pc.parties.client.prf_modules import PRFFetcherConv2D, PRFFetcherReLU, \
    PRFFetcherMaxPool, \
    PRFFetcherSecureModelSegmentation, PRFFetcherSecureModelClassification, PRFFetcherBlockReLU
import mmcv
from research.utils import build_data
from research.secure_inference_3pc.parties.client.secure_modules import SecureConv2DClient, SecureReLUClient, \
    SecureMaxPoolClient, \
    SecureBlockReLUClient

from research.mmlab_extension.segmentation.secure_aspphead import SecureASPPHead
from research.mmlab_extension.segmentation.resnet_seg import AvgPoolResNetSeg
from research.mmlab_extension.classification.resnet_cifar_v2 import ResNet_CIFAR_V2  # TODO: why is this needed?
from research.mmlab_extension.classification.resnet import MyResNet  # TODO: why is this needed?

from research.secure_inference_3pc.timer import timer
import numpy as np


def build_secure_fully_connected(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False,
                                 device="cpu"):
    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DClient
    shape = tuple(conv_module.weight.shape) + (1, 1)

    stride = (1, 1)
    dilation = (1, 1)
    padding = (0, 0)
    groups = 1

    return conv_class(
        W_shape=shape,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        device=device

    )


def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False, device="cpu"):
    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DClient

    return conv_class(
        W_shape=conv_module.weight.shape,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        groups=conv_module.groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        device=device

    )


def build_secure_relu(crypto_assets, network_assets, is_prf_fetcher=False, dummy_relu=False, **kwargs):
    relu_class = PRFFetcherReLU if is_prf_fetcher else SecureReLUClient
    return relu_class(crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu, **kwargs)


class SecureModelClassification(SecureModule):
    def __init__(self, model, **kwargs):
        super(SecureModelClassification, self).__init__(**kwargs)
        self.model = model

    @timer(name="Inference", avg=False)
    def forward(self, img):
        I = TypeConverter.f2i(img)
        I1 = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, dtype=SIGNED_DTYPE, size=img.shape)
        I0 = I - I1
        out = self.model.backbone(I0)[0]
        out = self.model.neck(out)
        out_0 = self.model.head.fc(out)
        out_1 = self.network_assets.receiver_01.get()
        out = out_1 + out_0
        out = TypeConverter.i2f(out)
        out = out.argmax()

        return out


class SecureModelSegmentation(SecureModule):
    def __init__(self, model, **kwargs):
        super(SecureModelSegmentation, self).__init__(**kwargs)
        self.model = model

    @timer(name="Inference", avg=False)
    def forward(self, img, img_meta):
        I = TypeConverter.f2i(img)
        I1 = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, dtype=SIGNED_DTYPE, size=img.shape)
        I0 = I - I1
        out_0 = self.model.decode_head(self.model.backbone(I0))
        out_1 = self.network_assets.receiver_01.get()
        out = out_1 + out_0
        out = TypeConverter.i2f(out)

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

        return seg_pred


def run_inference_classification(img, gt, dataset, img_meta=None, is_dummy=False):
    out = model(img)
    if not is_dummy:
        return gt == out
        # results_gt.append(gt)
        # results_pred.append(out)
        # print((backend.array(results_gt) == backend.array(results_pred)).mean())


def run_inference_segmentation(img, gt, dataset, img_meta=None, is_dummy=False):
    seg_pred = model(img, img_meta)

    if not is_dummy:
        return intersect_and_union(
            seg_pred,
            gt,
            len(dataset.CLASSES),
            dataset.ignore_index,
            label_map=dict(),
            reduce_zero_label=dataset.reduce_zero_label)

def get_sample_data(dataset_type, dataset, sample_id):
    if dataset is None:
        if "VOC" in dataset_type:
            img = np.zeros((1, 3, 512, 713), dtype=np.float32)
            img_meta = {
                'filename': 'data/ade/ADEChallengeData2016/images/validation/ADE_val_00000001.jpg',
                'ori_filename': 'ADE_val_00000001.jpg',
                'ori_shape': (512, 713, 3), 'img_shape': (512, 713, 3), 'pad_shape': (512, 713, 3),
                'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
                'flip': False, 'flip_direction': 'horizontal', 'img_norm_cfg':
                    {'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                     'std': np.array([58.395, 57.12, 57.375], dtype=np.float32), 'to_rgb': True}
            }
            gt = np.zeros((512, 673), dtype=np.uint8)
        elif "ADE20K" in dataset_type:
            img = np.zeros((1, 3, 512, 673), dtype=np.float32)
            img_meta = {
                'filename': 'data/ade/ADEChallengeData2016/images/validation/ADE_val_00000001.jpg',
                'ori_filename': 'ADE_val_00000001.jpg',
                'ori_shape': (512, 673, 3), 'img_shape': (512, 673, 3), 'pad_shape': (512, 673, 3),
                'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
                'flip': False, 'flip_direction': 'horizontal', 'img_norm_cfg':
                    {'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                     'std': np.array([58.395, 57.12, 57.375], dtype=np.float32), 'to_rgb': True}
            }
            gt = np.zeros((512, 673), dtype=np.uint8)
        elif 'CIFAR100' in dataset_type:
            img = np.zeros((3, 32, 32), dtype=np.float32)
            gt = 0
            img_meta = None
        elif 'ImageNet' in dataset_type:
            img = np.zeros((3, 224, 224), dtype=np.float32)
            gt = 0
            img_meta = None
        else:
            raise NotImplementedError
    else:
        if ("VOC" in dataset_type) or ("ADE20K" in dataset_type):
            img = dataset[sample_id]['img'][0].data.unsqueeze(0)
            img_meta = dataset[sample_id]['img_metas'][0].data
            gt = dataset.get_gt_seg_map_by_idx(sample_id)
        elif ('CIFAR100' in dataset_type) or ('ImageNet' in dataset_type):
            img = dataset[sample_id]['img'].data
            img = img.reshape((1,) + img.shape)
            gt = dataset.get_gt_labels()[sample_id]
            img_meta = None
        else:
            raise NotImplementedError

    return img, gt, img_meta

def full_inference(cfg, model, image_start, image_end, device, network_assets, dummy=False, dump_dir=None, skip_existing=False):
    is_segmentation = cfg.model.type == "EncoderDecoder"
    run_inference_func = run_inference_segmentation if is_segmentation else run_inference_classification
    datadtype = cfg.data['test']['type']
    if not dummy:
        dataset = build_data(cfg, mode="test")
    else:
        dataset = None

    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.fetch(model=model.prf_fetcher)
    model.eval()

    results = []

    for sample_id in tqdm(range(image_start, image_end)):
        if skip_existing and os.path.exists(os.path.join(dump_dir, f"{sample_id}.npy")):
            continue
        img, gt, img_meta = get_sample_data(datadtype, dataset, sample_id)
        if model.prf_fetcher:
            model.prf_fetcher.prf_handler.fetch_image(image=backend.zeros(shape=img.shape, dtype=SIGNED_DTYPE))

        # Handshake
        network_assets.sender_01.put(np.array(img.shape))
        network_assets.sender_02.put(np.array(img.shape))
        network_assets.receiver_01.get()
        network_assets.receiver_02.get()
        cur_result = run_inference_func(img, gt, dataset, img_meta=img_meta, is_dummy=dummy)

        if dump_dir is not None:
            np.save(os.path.join(dump_dir, f"{sample_id}.npy"), cur_result)

        results.append(cur_result)

        if is_segmentation:
            print(sample_id, dataset.evaluate(results, logger='silent', **{'metric': ['mIoU']})['mIoU'])
        else:
            print(sample_id, np.mean(results))

    network_assets.sender_01.put(np.array([0]))
    network_assets.sender_02.put(np.array([0]))

    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.done()


if __name__ == "__main__":
    party = 0
    # CHECKPOINT=/home/yakir/assets/resnet_voc/models/iter_20000.pth
    # RELU_SPEC_FILE=/home/yakir/assets/resnet_voc/block_spec/0.12.pickle
    # SECURE_CONFIG_PATH=/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_avg_pool_secure_aspp.py
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dummy_image', action='store_true', default=False)
    parser.add_argument('--dump_dir', type=str, default=None)
    parser.add_argument('--image_start', type=int, default=0)
    parser.add_argument('--image_end', type=int, default=1)
    parser.add_argument('--skip_existing', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--secure_config_path', type=str,
                        default="/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_avg_pool_secure_aspp.py")
    parser.add_argument('--relu_spec_file', type=str, default=None)

    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.secure_config_path)

    crypto_assets, network_assets = get_assets(party, device=args.device)

    if PRF_PREFETCH:
        prf_fetcher = init_prf_fetcher(
            cfg=cfg,
            checkpoint_path=None,
            max_pool=PRFFetcherMaxPool,
            build_secure_conv=build_secure_conv,
            build_secure_relu=build_secure_relu,
            build_secure_fully_connected=build_secure_fully_connected,
            prf_fetcher_secure_model=PRFFetcherSecureModelSegmentation if cfg.model.type == "EncoderDecoder" else PRFFetcherSecureModelClassification,
            secure_block_relu=PRFFetcherBlockReLU,
            relu_spec_file=args.relu_spec_file,
            crypto_assets=crypto_assets,
            network_assets=network_assets,
            dummy_relu=DUMMY_RELU,
            device=args.device,
        )
    else:
        prf_fetcher = None

    model = get_secure_model(
        cfg,
        checkpoint_path=None,
        build_secure_conv=build_secure_conv,
        build_secure_relu=build_secure_relu,
        build_secure_fully_connected=build_secure_fully_connected,
        max_pool=SecureMaxPoolClient,
        secure_model_class=SecureModelSegmentation if cfg.model.type == "EncoderDecoder" else SecureModelClassification,
        block_relu=SecureBlockReLUClient,
        relu_spec_file=args.relu_spec_file,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        dummy_relu=DUMMY_RELU,
        prf_fetcher=prf_fetcher,
        device=args.device
    )

    full_inference(cfg,
                   model,
                   args.image_start,
                   args.image_end,
                   args.device,
                   network_assets,
                   args.dummy_image,
                   args.dump_dir,
                   args.skip_existing)

    network_assets.done()

    print("Num of bytes sent 0 ",
          network_assets.sender_02.num_of_bytes_sent + network_assets.sender_01.num_of_bytes_sent)

# sudo apt-get update
# curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
# sudo apt-get install bzip2
# bash Anaconda3-2019.03-Linux-x86_64.sh
# exit
# conda create -n secure-inference python=3.7 -y
# conda activate secure-inference
# conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
# pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
# pip install mmsegmentation
# sudo apt-get install ffmpeg libsm6 libxext6  -y
# conda install numba
# conda install tqdm
# https://stackoverflow.com/questions/62436205/connecting-aws-ec2-instance-using-python-socket
# git clone https://github.com/yg320/secure_inference.git
