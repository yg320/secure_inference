import torch
import numpy as np
from tqdm import tqdm

from research.secure_inference_3pc.base import SecureModule, decompose, get_c, P, module_67, DepthToSpace, \
    SpaceToDepth, get_assets, TypeConverter
from research.secure_inference_3pc.conv2d import conv_2d
from research.secure_inference_3pc.resnet_converter import securify_mobilenetv2_model, init_prf_fetcher
from functools import partial
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, NUM_BITS
from mmseg.ops import resize
from mmseg.datasets import build_dataset

from research.secure_inference_3pc.timer import Timer
from research.distortion.utils import get_model

from research.pipeline.backbones.secure_resnet import AvgPoolResNet
from research.pipeline.backbones.secure_aspphead import SecureASPPHead

import torch.nn.functional as F
from mmseg.core import intersect_and_union
from research.secure_inference_3pc.modules.client import PRFFetcherConv2D, PRFFetcherReLU, PRFFetcherSecureModel, PRFFetcherBlockReLU
from research.secure_inference_3pc.params import Params


class SecureConv2DClient(SecureModule):
    def __init__(self, W, stride, dilation, padding, groups, crypto_assets, network_assets):
        super(SecureConv2DClient, self).__init__(crypto_assets, network_assets)

        self.W_share = W
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

    def forward_(self, X_share):

        assert X_share.dtype == SIGNED_DTYPE
        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert (self.W_share.shape[1] == X_share.shape[1]) or self.groups > 1

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_share.shape, dtype=SIGNED_DTYPE)
        C_share = self.network_assets.receiver_02.get()

        E_share = X_share - A_share
        F_share = self.W_share - B_share

        share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(np.concatenate([E_share.flatten(), F_share.flatten()]))
        E_share_server, F_share_server = \
            share_server[:E_share.size].reshape(E_share.shape), share_server[E_share.size:].reshape(F_share.shape)

        E = E_share_server + E_share
        F = F_share_server + F_share

        out_numpy = conv_2d(X_share, F, E, self.W_share, self.padding, self.stride, self.dilation, self.groups)

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

        c_bits_0 = get_c(x_bits_0, bits, beta, np.int8(0))
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

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [NUM_BITS], dtype=np.int8)
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


class SecureMSBClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBClient, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)
        self.private_compare = PrivateCompareClient(crypto_assets, network_assets)

    def forward(self, a_0):
        return self.forward_(a_0)

    def forward_(self, a_0):

        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=np.int8)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [NUM_BITS], dtype=np.int8)
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
            return torch.zeros_like(X_share)
        else:

            shape = X_share.shape
            # X_share = X_share.numpy()
            dtype = X_share.dtype
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)

            X_share = X_share.astype(self.dtype).flatten()
            MSB_0 = self.DReLU(X_share)
            relu_0 = self.mult(X_share, MSB_0).reshape(shape)
            ret = relu_0.astype(SIGNED_DTYPE)

            return ret + mu_0


class SecureBlockReLUClient(SecureModule):

    def __init__(self, crypto_assets, network_assets, block_sizes, dummy_relu=False):
        super(SecureBlockReLUClient, self).__init__(crypto_assets, network_assets)
        self.block_sizes = np.array(block_sizes)
        self.dummy_relu = dummy_relu
        self.DReLU = SecureDReLUClient(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if
                                   0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def forward(self, activation):
        if self.dummy_relu:
            network_assets.sender_01.put(activation)
            return torch.zeros_like(activation)

        assert activation.dtype == SIGNED_DTYPE
        reshaped_inputs = []
        mean_tensors = []
        channels = []
        orig_shapes = []

        for block_size in self.active_block_sizes:
            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = activation[:, cur_channels]
            reshaped_input = SpaceToDepth(block_size)(cur_input)
            assert reshaped_input.dtype == SIGNED_DTYPE
            mean_tensor = np.sum(reshaped_input, axis=-1, keepdims=True)

            channels.append(cur_channels)
            reshaped_inputs.append(reshaped_input)
            orig_shapes.append(mean_tensor.shape)
            mean_tensors.append(mean_tensor.flatten())

        cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
        mean_tensors = np.concatenate(mean_tensors)
        assert mean_tensors.dtype == SIGNED_DTYPE
        activation = activation.astype(self.dtype)
        sign_tensors = self.DReLU(mean_tensors.astype(self.dtype))

        relu_map = np.ones_like(activation)
        for i in range(len(self.active_block_sizes)):
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shapes[i])
            relu_map[:, channels[i]] = DepthToSpace(self.active_block_sizes[i])(
                sign_tensor.repeat(reshaped_inputs[i].shape[-1], axis=-1))

        # with Timer("Mult in BlockRelu"):
        activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels], activation[:, ~self.is_identity_channels])
        activation = activation.astype(SIGNED_DTYPE)

        return activation




def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False):

    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DClient

    if is_prf_fetcher:
        W = np.zeros(shape=conv_module.weight.shape, dtype=SIGNED_DTYPE)
    else:
        W = crypto_assets[CLIENT, SERVER].integers(low=MIN_VAL // 2,
                                                   high=MAX_VAL // 2,
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


class SecureModel(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(SecureModel, self).__init__( crypto_assets, network_assets)
        self.model = model

    def forward(self, img, img_meta):

        I = TypeConverter.f2i(img)
        I1 = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL // 2, high=MAX_VAL // 2, dtype=SIGNED_DTYPE, size=img.shape)
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
    results = []
    for sample_id in tqdm(range(num_images)):
        img = dataset[sample_id]['img'].data.unsqueeze(0)[:, :, :256, :256]
        img_meta = dataset[sample_id]['img_metas'].data

        img_meta['img_shape'] = (256, 256, 3)
        seg_map = dataset.get_gt_seg_map_by_idx(sample_id)
        seg_map = seg_map[:min(seg_map.shape), :min(seg_map.shape)]
        img_meta['ori_shape'] = (seg_map.shape[0], seg_map.shape[1], 3)

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
    model = get_model(
        config=Params.SECURE_CONFIG_PATH,
        gpu_id=None,
        checkpoint_path=None
    )

    crypto_assets, network_assets = get_assets(party, repeat=Params.NUM_IMAGES, simulated_bandwidth=Params.SIMULATED_BANDWIDTH)

    model = securify_mobilenetv2_model(
        model,
        build_secure_conv=build_secure_conv,
        build_secure_relu=build_secure_relu,
        secure_model_class=SecureModel,
        block_relu=SecureBlockReLUClient,
        relu_spec_file=Params.RELU_SPEC_FILE,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        dummy_relu=Params.DUMMY_RELU
    )

    if Params.PRF_PREFETCH:
        init_prf_fetcher(Params=Params,
                         build_secure_conv=build_secure_conv,
                         build_secure_relu=build_secure_relu,
                         prf_fetcher_secure_model=PRFFetcherSecureModel,
                         secure_block_relu=PRFFetcherBlockReLU,
                         crypto_assets=crypto_assets,
                         network_assets=network_assets)

    full_inference(model, Params.NUM_IMAGES)

    network_assets.done()
