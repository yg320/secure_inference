import torch
import numpy as np
from tqdm import tqdm
import os
from research.secure_inference_3pc.base import SecureModule, decompose, get_c_party_0, P, module_67, DepthToSpace, \
    SpaceToDepth, get_assets, TypeConverter
from research.secure_inference_3pc.conv2d import conv_2d
from research.secure_inference_3pc.resnet_converter import get_secure_model, init_prf_fetcher
from functools import partial
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, NUM_OF_COMPARE_BITS
from mmseg.ops import resize
from mmseg.datasets import build_dataset
from research.secure_inference_3pc.modules.conv2d import get_output_shape
from research.secure_inference_3pc.timer import Timer
from research.distortion.utils import get_model
from research.secure_inference_3pc.conv2d_torch import Conv2DHandler
from research.bReLU import NumpySecureOptimizedBlockReLU

from research.pipeline.backbones.secure_resnet import AvgPoolResNet
from research.pipeline.backbones.secure_aspphead import SecureASPPHead
from research.mmlab_extension.resnet_cifar_v2 import ResNet_CIFAR_V2
from research.mmlab_extension.classification.resnet import AvgPoolResNet, MyResNet
import torch.nn.functional as F
from mmseg.core import intersect_and_union
from research.secure_inference_3pc.modules.client import PRFFetcherConv2D, PRFFetcherReLU, PRFFetcherMaxPool, PRFFetcherSecureModelSegmentation, PRFFetcherSecureModelClassification, PRFFetcherBlockReLU
from research.secure_inference_3pc.params import Params
import mmcv
from research.utils import build_data


class SecureConv2DClient(SecureModule):
    # forward_num = 0
    # out_path = "/home/yakir/debug/client"
    def __init__(self, W, stride, dilation, padding, groups, crypto_assets, network_assets):
        super(SecureConv2DClient, self).__init__(crypto_assets, network_assets)

        self.W_share = W
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.conv2d_handler = Conv2DHandler("cuda:0")

    def forward_(self, X_share):
        # SecureConv2DClient.forward_num += 1
        assert X_share.dtype == SIGNED_DTYPE
        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert (self.W_share.shape[1] == X_share.shape[1]) or self.groups > 1

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_share.shape, dtype=SIGNED_DTYPE)
        C_share = self.network_assets.receiver_02.get()
        # np.save(file=os.path.join(SecureConv2DClient.out_path, f"A_{SecureConv2DClient.forward_num}.npy"))
        E_share = X_share - A_share
        F_share = self.W_share - B_share

        share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(np.concatenate([E_share.flatten(), F_share.flatten()]))
        E_share_server, F_share_server = share_server[:E_share.size].reshape(E_share.shape), share_server[E_share.size:].reshape(F_share.shape)

        E = E_share_server + E_share
        F = F_share_server + F_share
        # out_numpy =  np.zeros(get_output_shape(X_share, self.W_share, self.padding, self.dilation, self.stride), dtype=X_share.dtype)
        out_numpy = self.conv2d_handler.conv2d(X_share, F, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
        out_numpy += self.conv2d_handler.conv2d(E, self.W_share, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
        # out_numpy = conv_2d(X_share, F, E, self.W_share, self.padding, self.stride, self.dilation, self.groups)
        # out_numpy = torch.conv2d(torch.from_numpy(X_share), torch.from_numpy(F), padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups).numpy()
        # out_numpy += torch.conv2d(torch.from_numpy(E), torch.from_numpy(self.W_share), padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups).numpy()

        out_numpy = out_numpy + C_share

        out = out_numpy // self.trunc
        # This is the proper way, but it's slower and takes more time
        # t = out_numpy.dtype
        # out = (out_numpy / self.trunc).round().astype(t)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=out.dtype)

        return out + mu_0

    def forward(self, X_share):
        # with Timer("SecureConv2DClient"):
        return self.forward_(X_share)

class PrivateCompareClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PrivateCompareClient, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_0, r, beta):
        # with Timer("PrivateCompareClient"):
        return self.forward_(x_bits_0, r, beta)

    def forward_(self, x_bits_0, r, beta):
        if np.any(r == np.iinfo(r.dtype).max):
            assert False
        # with Timer("PrivateCompareClient - Random"):
        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=P, size=x_bits_0.shape, dtype=np.int32)
        # u = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_0.shape, dtype=self.crypto_assets.numpy_dtype)
        r[beta] += 1
        bits = decompose(r)

        c_bits_0 = get_c_party_0(x_bits_0, bits, beta, np.int8(0))
        np.multiply(s, c_bits_0, out=s)
        d_bits_0 = module_67(s)

        d_bits_0 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_0, axis=-1)
        self.network_assets.sender_02.put(d_bits_0)

        return


class ShareConvertClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertClient, self).__init__(crypto_assets, network_assets)
        self.private_compare = PrivateCompareClient(crypto_assets, network_assets)

    def forward(self, a_0):
        return self.forward_(a_0)

    def forward_(self, a_0):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=np.int8)

        r = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val, size=a_0.shape, dtype=self.dtype)

        alpha = (r < r_0).astype(self.dtype)

        a_tild_0 = a_0 + r_0
        beta_0 = (a_tild_0 < a_0).astype(self.dtype)
        self.network_assets.sender_02.put(a_tild_0)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [NUM_OF_COMPARE_BITS], dtype=np.int8)
        delta_0 = self.network_assets.receiver_02.get()

        self.private_compare(x_bits_0, r - 1, eta_pp)

        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(self.min_val, self.max_val, size=a_0.shape, dtype=self.dtype)
        eta_pp = eta_pp.astype(self.dtype)
        t0 = eta_pp * eta_p_0
        t1 = self.add_mode_L_minus_one(t0, t0)
        t2 = self.sub_mode_L_minus_one(eta_pp, t1)
        eta_0 = self.add_mode_L_minus_one(eta_p_0, t2)

        t0 = self.add_mode_L_minus_one(delta_0, eta_0)
        t1 = self.sub_mode_L_minus_one(t0, self.dtype(1))
        t2 = self.sub_mode_L_minus_one(t1, alpha)
        theta_0 = self.add_mode_L_minus_one(beta_0, t2)

        y_0 = self.sub_mode_L_minus_one(a_0, theta_0)
        y_0 = self.add_mode_L_minus_one(y_0, mu_0)

        return y_0


class SecureMultiplicationClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationClient, self).__init__(crypto_assets, network_assets)

    def forward(self, X_share, Y_share):
        return self.forward_(X_share, Y_share)

    def forward_(self, X_share, Y_share):
        assert X_share.dtype == self.dtype
        assert Y_share.dtype == self.dtype

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape,
                                                           dtype=self.dtype)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape,
                                                           dtype=self.dtype)
        C_share = self.network_assets.receiver_02.get()

        E_share = X_share - A_share
        F_share = Y_share - B_share

        E_share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(E_share)
        F_share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)

        E = E_share_server + E_share
        F = F_share_server + F_share

        out = X_share * F + Y_share * E + C_share
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(np.iinfo(X_share.dtype).min, np.iinfo(X_share.dtype).max, size=out.shape, dtype=X_share.dtype)

        return out + mu_0


class SecureSelectShareClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureSelectShareClient, self).__init__(crypto_assets, network_assets)
        self.secure_multiplication = SecureMultiplicationClient(crypto_assets, network_assets)

    def forward(self, alpha, x, y):
        # if alpha == 0: return x else return 1
        dtype = alpha.dtype
        shape = alpha.shape
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)

        w = y - x
        c = self.secure_multiplication(alpha, w)
        z = x + c
        return z + mu_0


class SecureMSBClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBClient, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)
        self.private_compare = PrivateCompareClient(crypto_assets, network_assets)

    def forward(self, a_0):
        return self.forward_(a_0)

    def forward_(self, a_0):

        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=np.int8)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [NUM_OF_COMPARE_BITS], dtype=np.int8)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=a_0.dtype)

        x_0 = self.network_assets.receiver_02.get()
        x_bit_0_0 = self.network_assets.receiver_02.get()

        y_0 = self.add_mode_L_minus_one(a_0, a_0)
        r_0 = self.add_mode_L_minus_one(x_0, y_0)
        r_1 = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(r_0)
        r = self.add_mode_L_minus_one(r_0, r_1)

        r_mode_2 = r % 2
        self.private_compare(x_bits_0, r, beta)

        beta = beta.astype(self.dtype)
        beta_p_0 = self.network_assets.receiver_02.get()

        gamma_0 = beta_p_0 + (0 * beta) - (2 * beta * beta_p_0)
        delta_0 = x_bit_0_0 - (2 * r_mode_2 * x_bit_0_0)

        theta_0 = self.mult(gamma_0, delta_0)
        alpha_0 = gamma_0 + delta_0 - 2 * theta_0
        alpha_0 = alpha_0 + mu_0

        return alpha_0


class SecureDReLUClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUClient, self).__init__(crypto_assets, network_assets)

        self.share_convert = ShareConvertClient(crypto_assets, network_assets)
        self.msb = SecureMSBClient(crypto_assets, network_assets)

    def forward(self, X_share):
        assert X_share.dtype == self.dtype
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=X_share.dtype)

        X0_converted = self.share_convert(self.dtype(2) * X_share)
        MSB_0 = self.msb(X0_converted)

        return -MSB_0+mu_0


class SecureReLUClient(SecureModule):
    def __init__(self, crypto_assets, network_assets, dummy_relu=False):
        super(SecureReLUClient, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUClient(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        return self.forward_(X_share)

    def forward_(self, X_share):
        if self.dummy_relu:
            network_assets.sender_01.put(X_share)
            return np.zeros_like(X_share)
        else:

            shape = X_share.shape
            dtype = X_share.dtype
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)

            X_share = X_share.astype(self.dtype).flatten()
            MSB_0 = self.DReLU(X_share)
            relu_0 = self.mult(X_share, MSB_0).reshape(shape)
            ret = relu_0.astype(SIGNED_DTYPE)

            return ret + mu_0


class SecureMaxPoolClient(SecureModule):
    def __init__(self, kernel_size, stride, padding, crypto_assets, network_assets):
        super(SecureMaxPoolClient, self).__init__(crypto_assets, network_assets)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.select_share = SecureSelectShareClient(crypto_assets, network_assets)
        self.dReLU =  SecureDReLUClient(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)

        assert self.kernel_size == 3
        assert self.stride == 2
        assert self.padding == 1

    def forward(self, x):
        # self.network_assets.sender_01.put(x)

        assert x.shape[2] == 112
        assert x.shape[3] == 112

        x = np.pad(x, ((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant')
        x = np.stack([x[:, :, 0:-1:2, 0:-1:2],
                      x[:, :, 0:-1:2, 1:-1:2],
                      x[:, :, 0:-1:2, 2::2],
                      x[:, :, 1:-1:2, 0:-1:2],
                      x[:, :, 1:-1:2, 1:-1:2],
                      x[:, :, 1:-1:2, 2::2],
                      x[:, :, 2::2, 0:-1:2],
                      x[:, :, 2::2, 1:-1:2],
                      x[:, :, 2::2, 2::2]])

        out_shape = x.shape[1:]
        x = x.reshape((x.shape[0], -1)).astype(self.dtype)

        max_ = x[0]
        for i in range(1, 9):
            w = x[i] - max_
            beta = self.dReLU(w)
            max_ = self.select_share(beta, max_, x[i])
            #
            # a = self.mult(beta, x[i])
            # b = self.mult((1 - beta), max_)
            # max_ = a + b

        ret = max_.reshape(out_shape).astype(SIGNED_DTYPE)
        # self.network_assets.sender_01.put(ret)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)

        return ret + mu_0


class SecureBlockReLUClient(SecureModule, NumpySecureOptimizedBlockReLU):
    def __init__(self, block_sizes, crypto_assets, network_assets, dummy_relu=False):
        SecureModule.__init__(self, crypto_assets=crypto_assets, network_assets=network_assets)
        NumpySecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.secure_DReLU = SecureDReLUClient(crypto_assets, network_assets)
        self.secure_mult = SecureMultiplicationClient(crypto_assets, network_assets)

        self.dummy_relu = dummy_relu

    def mult(self, x, y):
        return self.secure_mult(x.astype(self.dtype), y.astype(self.dtype))

    def DReLU(self, activation):
        return self.secure_DReLU(activation.astype(self.dtype))

    def forward(self, activation):
        if self.dummy_relu:
            network_assets.sender_01.put(activation)
            return torch.zeros_like(activation)

        activation = NumpySecureOptimizedBlockReLU.forward(self, activation)
        activation = activation.astype(SIGNED_DTYPE)

        return activation


def build_secure_fully_connected(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False):
    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DClient
    shape = tuple(conv_module.weight.shape) + (1, 1)

    stride = (1, 1)
    dilation = (1, 1)
    padding = (0, 0)
    groups = 1

    if is_prf_fetcher:
        W = np.zeros(shape=shape, dtype=SIGNED_DTYPE)
    else:
        W = crypto_assets[CLIENT, SERVER].integers(low=MIN_VAL,
                                                   high=MAX_VAL,
                                                   size=shape,
                                                   dtype=SIGNED_DTYPE)

    return conv_class(
        W=W,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False):

    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DClient

    if is_prf_fetcher:
        W = np.zeros(shape=conv_module.weight.shape, dtype=SIGNED_DTYPE)
    else:
        W = crypto_assets[CLIENT, SERVER].integers(low=MIN_VAL,
                                                   high=MAX_VAL,
                                                   size=conv_module.weight.shape,
                                                   dtype=SIGNED_DTYPE)

    return conv_class(
        W=W,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        groups=conv_module.groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def build_secure_relu(crypto_assets, network_assets, is_prf_fetcher=False, dummy_relu=False):
    relu_class = PRFFetcherReLU if is_prf_fetcher else SecureReLUClient
    return relu_class(crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu)


class SecureModelClassification(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(SecureModelClassification, self).__init__( crypto_assets, network_assets)
        self.model = model
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
    def __init__(self, model,  crypto_assets, network_assets):
        super(SecureModelSegmentation, self).__init__( crypto_assets, network_assets)
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

def full_inference_classification(cfg, model, num_images):
    dataset = build_data(cfg, train=False)
    results_gt = []
    results_pred = []
    model.eval()
    for sample_id in tqdm(range(num_images)):
        img = dataset[sample_id]['img'].data[np.newaxis]
        gt = dataset.get_gt_labels()[sample_id]
        with Timer("Inference"):
            if model.prf_fetcher:
                model.prf_fetcher.prf_handler.fetch(repeat=1, model=model.prf_fetcher, image=np.zeros(shape=Params.IMAGE_SHAPE, dtype=SIGNED_DTYPE))
            out = model(img)
    #     # gt = dataset.gt_labels[sample_id]
        results_gt.append(gt)
        results_pred.append(out)
    print((np.array(results_gt) == np.array(results_pred)).mean())


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

    crypto_assets, network_assets = get_assets(party, repeat=Params.NUM_IMAGES, simulated_bandwidth=Params.SIMULATED_BANDWIDTH)

    if Params.PRF_PREFETCH:
        prf_fetcher = init_prf_fetcher(
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
            dummy_relu=Params.DUMMY_RELU)
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
        prf_fetcher=prf_fetcher
    )



    if cfg.model.type == "EncoderDecoder":
        full_inference(cfg, model, Params.NUM_IMAGES)
    else:
        full_inference_classification(cfg, model, Params.NUM_IMAGES)

    network_assets.done()

    print("Num of bytes sent 0 -> 1", network_assets.sender_01.num_of_bytes_sent)
    print("Num of bytes sent 0 -> 2", network_assets.sender_02.num_of_bytes_sent)