import torch
import numpy as np
from tqdm import tqdm

from research.secure_inference_3pc.base import SecureModule, decompose, get_c, P, module_67, DepthToSpace, \
    SpaceToDepth, get_assets, TypeConverter
from research.secure_inference_3pc.conv2d import conv_2d, compile_numba_funcs, get_output_shape
from research.secure_inference_3pc.resnet_converter import securify_mobilenetv2_model
from functools import partial
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER
from mmseg.ops import resize
from mmseg.datasets import build_dataset

from research.secure_inference_3pc.timer import Timer
from research.distortion.utils import get_model
from research.distortion.utils import ArchUtilsFactory

from research.pipeline.backbones.secure_resnet import AvgPoolResNet
from research.pipeline.backbones.secure_aspphead import SecureASPPHead
from research.distortion.utils import get_data
import torch.nn.functional as F
from mmseg.core import intersect_and_union


class SecureConv2DClient(SecureModule):
    def __init__(self, W, stride, dilation, padding, groups, crypto_assets, network_assets):
        super(SecureConv2DClient, self).__init__(crypto_assets, network_assets)

        self.W_share = W.numpy()
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups


    def forward(self, X_share):
        X_share = X_share.numpy()

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=X_share.shape, dtype=np.int64)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=self.W_share.shape, dtype=np.int64)

        out_shape = get_output_shape(X_share, self.W_share, self.padding, self.dilation, self.stride)

        return torch.from_numpy(np.zeros(shape=out_shape, dtype=X_share.dtype))


class PrivateCompareClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PrivateCompareClient, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_0, r, beta):
        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=P, size=x_bits_0.shape, dtype=np.int32)


class ShareConvertClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertClient, self).__init__(crypto_assets, network_assets)
        self.private_compare = PrivateCompareClient(crypto_assets, network_assets)

    def forward(self, a_0):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=np.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [64], dtype=np.int8)
        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(self.min_val, self.max_val, size=a_0.shape, dtype=self.dtype)

        return a_0


class SecureMultiplicationClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationClient, self).__init__(crypto_assets, network_assets)

    def forward(self, X_share, Y_share):

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)

        return X_share


class SecureMSBClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBClient, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)
        self.private_compare = PrivateCompareClient(crypto_assets, network_assets)

    def forward(self, a_0):

        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=np.int8)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [64], dtype=np.int8)

        return a_0


class SecureDReLUClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUClient, self).__init__(crypto_assets, network_assets)

        self.share_convert = ShareConvertClient(crypto_assets, network_assets)
        self.msb = SecureMSBClient(crypto_assets, network_assets)

    def forward(self, X_share):
        assert X_share.dtype == self.dtype
        X0_converted = self.share_convert(X_share)
        MSB_0 = self.msb(X0_converted)
        return MSB_0


class SecureReLUClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUClient, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUClient(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)

    def forward(self, X_share):
        return self.forward_(X_share)

    def forward_(self, X_share):

        X_share_np = X_share.numpy()
        X_share_np = X_share_np.astype(self.dtype).flatten()
        MSB_0 = self.DReLU(X_share_np)
        relu_0 = self.mult(X_share_np, MSB_0)

        return X_share

def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module):
    # assert module.groups == 1
    return SecureConv2DClient(
        W=crypto_assets[CLIENT, SERVER].get_random_tensor_over_L(shape=conv_module.weight.shape),
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        groups=conv_module.groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def build_secure_relu(crypto_assets, network_assets):
    return SecureReLUClient(crypto_assets=crypto_assets, network_assets=network_assets)


def full_inference(model, num_images):

    dataset = build_dataset({'type': 'ADE20KDataset',
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
           })
    for sample_id in tqdm(range(num_images)):

        img = dataset[sample_id]['img'].data.unsqueeze(0)[:, :, :256, :256]
        img_meta = dataset[sample_id]['img_metas'].data

        img_meta['img_shape'] = (256, 256, 3)
        seg_map = dataset.get_gt_seg_map_by_idx(sample_id)
        seg_map = seg_map[:min(seg_map.shape), :min(seg_map.shape)]
        img_meta['ori_shape'] = (seg_map.shape[0], seg_map.shape[1], 3)

        with Timer("Inference"):
            I = TypeConverter.f2i(img)
            I1 = crypto_assets[CLIENT, SERVER].get_random_tensor_over_L(shape=I.shape)
            I0 = I - I1
            with Timer("PRFs"):
                out_0 = model.decode_head(model.backbone(I0))
            print('fdsfsd')


if __name__ == "__main__":

    config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py"
    secure_config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline_secure.py"
    model_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth"
    relu_spec_file = None #"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/test/block_size_spec_0.15.pickle"
    image_shape = (1, 3, 256, 256)
    num_images = 10

    model = get_model(
        config=secure_config_path,
        gpu_id=None,
        checkpoint_path=None
    )

    crypto_assets, network_assets = get_assets(0, repeat=num_images)

    securify_mobilenetv2_model(model,
                               build_secure_conv=partial(build_secure_conv, crypto_assets=crypto_assets, network_assets=network_assets),
                               build_secure_relu=partial(build_secure_relu, crypto_assets=crypto_assets, network_assets=network_assets),
                               block_relu=None,
                               relu_spec_file=None)

    full_inference(model, num_images)

    crypto_assets.done()
    network_assets.done()


