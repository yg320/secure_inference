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
from research.pipeline.backbones.secure_resnet import AvgPoolResNet
from research.pipeline.backbones.secure_aspphead import SecureASPPHead
from research.mmlab_extension.resnet_cifar_v2 import ResNet_CIFAR_V2
from research.mmlab_extension.classification.resnet import AvgPoolResNet, MyResNet
from research.secure_inference_3pc.timer import timer
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

    @timer("Inference")
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


def full_inference_classification(cfg, model, num_images, device):
    dataset = build_data(cfg, train=False)
    results_gt = []
    results_pred = []
    model.eval()
    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.fetch(repeat=num_images, model=model.prf_fetcher,
                                            image=backend.zeros(shape=Params.IMAGE_SHAPE, dtype=SIGNED_DTYPE))
    for sample_id in tqdm(range(num_images)):
        img = dataset[sample_id]['img'].data
        img = backend.put_on_device(img.reshape((1,) + img.shape), device)
        gt = dataset.get_gt_labels()[sample_id]
        out = model(img)
        results_gt.append(gt)
        results_pred.append(out)
    print((backend.array(results_gt) == backend.array(results_pred)).mean())


def full_inference(cfg, model, num_images):
    dataset = build_data(cfg, train=False)

    # dataset = build_dataset({'type': 'ADE20KDataset',
    #        'data_root': 'data/ade/ADEChallengeData2016',
    #        'img_dir': 'images/validation',
    #        'ann_dir': 'annotations/validation',
    #        'pipeline': [
    #            {'type': 'LoadImageFromFile'},
    #            {'type': 'LoadAnnotations', 'reduce_zero_label': True},
    #            {'type': 'Resize', 'img_scale': (1024, 256), 'keep_ratio': True},
    #            {'type': 'RandomFlip', 'prob': 0.0},
    #            {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True},
    #            {'type': 'DefaultFormatBundle'},
    #            {'type': 'Collect', 'keys': ['img', 'gt_semantic_seg']}]
    #        })
    results = []
    for sample_id in tqdm(range(num_images)):
        img = dataset[sample_id]['img'][0].data.unsqueeze(0)
        img_meta = dataset[sample_id]['img_metas'][0].data
        seg_map = dataset.get_gt_seg_map_by_idx(sample_id)

        # img_meta['img_shape'] = (256, 256, 3)
        # img = img[:, :, :256, :256]
        # seg_map = seg_map[:min(seg_map.shape), :min(seg_map.shape)]
        # img_meta['ori_shape'] = (seg_map.shape[0], seg_map.shape[1], 3)

        with Timer("Inference"):
            seg_pred = model(img.numpy(), img_meta)

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


if __name__ == "__main__":
    party = 0
    assert (Params.RELU_SPEC_FILE is None) or (Params.DUMMY_RELU is False)
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
            dummy_max_pool=Params.DUMMY_MAX_POOL,
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
        dummy_max_pool=Params.DUMMY_MAX_POOL,
        prf_fetcher=prf_fetcher,
        device=Params.CLIENT_DEVICE
    )



    if cfg.model.type == "EncoderDecoder":
        full_inference(cfg, model, Params.NUM_IMAGES)
    else:
        full_inference_classification(cfg, model, Params.NUM_IMAGES, Params.CLIENT_DEVICE)

    network_assets.done()

    # print("Num of bytes sent 0 -> 1", network_assets.sender_01.num_of_bytes_sent)
    # print("Num of bytes sent 0 -> 2", network_assets.sender_02.num_of_bytes_sent)