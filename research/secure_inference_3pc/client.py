from research.secure_inference_3pc.backend import backend
from tqdm import tqdm

from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.base import  get_assets, TypeConverter

from research.secure_inference_3pc.resnet_converter import get_secure_model, init_prf_fetcher
from research.secure_inference_3pc.const import CLIENT, SERVER, MIN_VAL, MAX_VAL, SIGNED_DTYPE
from mmseg.ops import resize

from research.secure_inference_3pc.timer import Timer

import torch.nn.functional as F
from mmseg.core import intersect_and_union
from research.secure_inference_3pc.modules.client import PRFFetcherConv2D, PRFFetcherReLU, PRFFetcherMaxPool, PRFFetcherSecureModelSegmentation, PRFFetcherSecureModelClassification, PRFFetcherBlockReLU
from research.secure_inference_3pc.params import Params
import mmcv
from research.utils import build_data
from research.secure_inference_3pc.modules.client import SecureConv2DClient, SecureReLUClient, SecureMaxPoolClient, SecureBlockReLUClient

from research.mmlab_extension.segmentation.secure_aspphead import SecureASPPHead
from research.mmlab_extension.resnet_cifar_v2 import ResNet_CIFAR_V2
from research.mmlab_extension.classification.resnet import AvgPoolResNet, MyResNet
from research.secure_inference_3pc.timer import timer
import numpy as np

def build_secure_fully_connected(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False, device="cpu"):
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
    def __init__(self, model,  **kwargs):
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


def full_inference_classification(cfg, model, num_images, device, network_assets, dummy=False):

    if not dummy:
        dataset = build_data(cfg, mode="test")
    results_gt = []
    results_pred = []
    model.eval()
    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.fetch(model=model.prf_fetcher)

    for sample_id in tqdm(range(num_images)):

        if dummy:
            img = np.zeros((3, 224, 224), dtype=np.float32)
            gt = 0
        else:
            img = dataset[sample_id]['img'].data
            gt = dataset.get_gt_labels()[sample_id]
        img = backend.put_on_device(img.reshape((1,) + img.shape), device)
        if model.prf_fetcher:
            model.prf_fetcher.prf_handler.fetch_image(image=backend.zeros(shape=img.shape, dtype=SIGNED_DTYPE))

        # Handshake
        network_assets.sender_01.put(np.array(img.shape))
        network_assets.sender_02.put(np.array(img.shape))
        network_assets.receiver_01.get()
        network_assets.receiver_02.get()

        out = model(img)
        results_gt.append(gt)
        results_pred.append(out)
        print((backend.array(results_gt) == backend.array(results_pred)).mean())
    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.done()


def full_inference_segmentation(cfg, model, num_images, device, network_assets, dummy=False):
    if not dummy:
        dataset = build_data(cfg, mode="distortion_extraction_val")
    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.fetch(model=model.prf_fetcher)

    results = []
    for sample_id in tqdm(range(num_images)):

        if dummy:
            img = np.zeros((1, 3, 512, 683), dtype=np.float32)
            img_meta = {'filename': 'data/ade/ADEChallengeData2016/images/validation/ADE_val_00000001.jpg', 'ori_filename': 'ADE_val_00000001.jpg',
                        'ori_shape': (512, 683, 3), 'img_shape': (512, 683, 3), 'pad_shape': (512, 683, 3), 'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),
                        'flip': False, 'flip_direction': 'horizontal', 'img_norm_cfg':
                {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}
            seg_map = np.zeros((512, 683), dtype=np.uint8)
        else:
            img = dataset[sample_id]['img'][0].data.unsqueeze(0)
            img_meta = dataset[sample_id]['img_metas'][0].data
            seg_map = dataset[sample_id]['gt_semantic_seg'][0].data.unsqueeze(0)

        if model.prf_fetcher:
            model.prf_fetcher.prf_handler.fetch_image(image=backend.zeros(shape=img.shape, dtype=SIGNED_DTYPE))

        # img_meta['img_shape'] = (256, 256, 3)
        # img = img[:, :, :256, :256]
        # seg_map = seg_map[:min(seg_map.shape), :min(seg_map.shape)]
        # img_meta['ori_shape'] = (seg_map.shape[0], seg_map.shape[1], 3)
        # Handshake
        network_assets.sender_01.put(np.array(img.shape))
        network_assets.sender_02.put(np.array(img.shape))
        network_assets.receiver_01.get()
        network_assets.receiver_02.get()

        seg_pred = model(img, img_meta)

        if not dummy:
            results.append(
                intersect_and_union(
                    seg_pred,
                    seg_map,
                    len(dataset.CLASSES),
                    dataset.ignore_index,
                    label_map=dict(),
                    reduce_zero_label=dataset.reduce_zero_label)
            )
            if sample_id % 10 == 0:
                print(sample_id, dataset.evaluate(results, logger='silent', **{'metric': ['mIoU']})['mIoU'])
    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.done()


if __name__ == "__main__":
    party = 0
    # assert (Params.RELU_SPEC_FILE is None) or (Params.DUMMY_RELU is False)
    cfg = mmcv.Config.fromfile(Params.SECURE_CONFIG_PATH)

    crypto_assets, network_assets = get_assets(party, device=Params.CLIENT_DEVICE, simulated_bandwidth=Params.SIMULATED_BANDWIDTH)

    if Params.PRF_PREFETCH:
        prf_fetcher = init_prf_fetcher(
            cfg=cfg,
            Params=Params,
            max_pool=PRFFetcherMaxPool,
            build_secure_conv=build_secure_conv,
            build_secure_relu=build_secure_relu,
            build_secure_fully_connected=build_secure_fully_connected,
            prf_fetcher_secure_model=PRFFetcherSecureModelSegmentation if cfg.model.type == "EncoderDecoder" else PRFFetcherSecureModelClassification,
            secure_block_relu=PRFFetcherBlockReLU,
            relu_spec_file=Params.RELU_SPEC_FILE,
            crypto_assets=crypto_assets,
            network_assets=network_assets,
            dummy_relu=Params.DUMMY_RELU,
            device=Params.CLIENT_DEVICE,
        )
    else:
        prf_fetcher = None

    model = get_secure_model(
        cfg,
        checkpoint_path=Params.MODEL_PATH,  # TODO: implement fc
        build_secure_conv=build_secure_conv,
        build_secure_relu=build_secure_relu,
        build_secure_fully_connected=build_secure_fully_connected,
        max_pool=SecureMaxPoolClient,
        secure_model_class=SecureModelSegmentation if cfg.model.type == "EncoderDecoder" else SecureModelClassification,
        block_relu=SecureBlockReLUClient,
        relu_spec_file=Params.RELU_SPEC_FILE,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        dummy_relu=Params.DUMMY_RELU,
        prf_fetcher=prf_fetcher,
        device=Params.CLIENT_DEVICE
    )



    if cfg.model.type == "EncoderDecoder":
        full_inference_segmentation(cfg, model, Params.NUM_IMAGES, Params.CLIENT_DEVICE, network_assets, Params.AWS_DUMMY)
    else:
        full_inference_classification(cfg, model, Params.NUM_IMAGES, Params.CLIENT_DEVICE, network_assets, Params.AWS_DUMMY)

    network_assets.done()

    # print("Num of bytes sent 0 -> 1", network_assets.sender_01.num_of_bytes_sent)
    # print("Num of bytes sent 0 -> 2", network_assets.sender_02.num_of_bytes_sent)
    print("Num of bytes sent 0 ", network_assets.sender_02.num_of_bytes_sent + network_assets.sender_01.num_of_bytes_sent)

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