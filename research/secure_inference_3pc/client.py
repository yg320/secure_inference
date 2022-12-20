import torch
import numpy as np
from tqdm import tqdm

from research.secure_inference_3pc.base import SecureModule, decompose, get_c, P, module_67, DepthToSpace, \
    SpaceToDepth, get_assets, TypeConverter
from research.secure_inference_3pc.conv2d import conv_2d, compile_numba_funcs
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
from research.secure_inference_3pc.modules.client import PRFFetcherConv2D, PRFFetcherReLU, PRFFetcherSecureModel


class SecureConv2DClient(SecureModule):
    def __init__(self, W, stride, dilation, padding, groups, crypto_assets, network_assets):
        super(SecureConv2DClient, self).__init__(crypto_assets, network_assets)

        self.W_share = W
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

    def forward_(self, X_share):

        X_share = X_share.numpy()
        assert X_share.dtype == self.signed_type
        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert (self.W_share.shape[1] == X_share.shape[1]) or self.groups > 1

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max,
                                                           size=X_share.shape, dtype=np.int64)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max,
                                                           size=self.W_share.shape, dtype=np.int64)
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

        return torch.from_numpy(out)

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
        # with Timer("ShareConvertClient"):
        return self.forward_(a_0)
    #TODO: should be like :@Timer("ShareConvertClient")
    def forward_(self, a_0):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=np.int8)

        r = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)

        alpha = (r < r_0).astype(self.dtype)

        a_tild_0 = a_0 + r_0
        beta_0 = (a_tild_0 < a_0).astype(self.dtype)
        self.network_assets.sender_02.put(a_tild_0)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [64], dtype=np.int8)
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

        return y_0


class SecureMultiplicationClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationClient, self).__init__(crypto_assets, network_assets)

    def forward(self, X_share, Y_share):
        # with Timer("SecureMultiplicationClient"):
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

        return out


class SecureMSBClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBClient, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)
        self.private_compare = PrivateCompareClient(crypto_assets, network_assets)

    def forward(self, a_0):
        # with Timer("SecureMSBClient"):
        return self.forward_(a_0)

    def forward_(self, a_0):

        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=np.int8)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [64], dtype=np.int8)
        x_0 = self.network_assets.receiver_02.get()
        x_bit_0_0 = self.network_assets.receiver_02.get()

        y_0 = self.add_mode_L_minus_one(a_0, a_0)
        r_0 = self.add_mode_L_minus_one(x_0, y_0)
        r_1 = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(r_0)
        r = self.add_mode_L_minus_one(r_0, r_1)

        r_mode_2 = r % 2
        self.private_compare(x_bits_0, r, beta)
        # execute_secure_compare
        beta = beta.astype(self.dtype)
        beta_p_0 = self.network_assets.receiver_02.get()

        gamma_0 = beta_p_0 + (0 * beta) - (2 * beta * beta_p_0)
        delta_0 = x_bit_0_0 - (2 * r_mode_2 * x_bit_0_0)

        theta_0 = self.mult(gamma_0, delta_0)
        alpha_0 = gamma_0 + delta_0 - 2 * theta_0

        return alpha_0


class SecureDReLUClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUClient, self).__init__(crypto_assets, network_assets)

        self.share_convert = ShareConvertClient(crypto_assets, network_assets)
        self.msb = SecureMSBClient(crypto_assets, network_assets)

    def forward(self, X_share):
        assert X_share.dtype == self.dtype
        X0_converted = self.share_convert(self.dtype(2) * X_share)
        MSB_0 = self.msb(X0_converted)
        return -MSB_0


class SecureReLUClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUClient, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUClient(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)

    def forward(self, X_share):
        # with Timer("SecureReLUClient"):
        return self.forward_(X_share)

    def forward_(self, X_share):
        # network_assets.sender_01.put(X_share)
        # return torch.zeros_like(X_share)
        shape = X_share.shape
        X_share = X_share.numpy()
        X_share = X_share.astype(self.dtype).flatten()
        MSB_0 = self.DReLU(X_share)
        relu_0 = self.mult(X_share, MSB_0).reshape(shape)
        ret = relu_0.astype(self.signed_type)
        return torch.from_numpy(ret)


class SecureBlockReLUClient(SecureModule):

    def __init__(self, crypto_assets, network_assets, block_sizes):
        super(SecureBlockReLUClient, self).__init__(crypto_assets, network_assets)
        self.block_sizes = np.array(block_sizes)
        self.DReLU = SecureDReLUClient(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if
                                   0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def forward(self, activation):
        # activation_server = network_assets.receiver_01.get()

        activation = activation.numpy()
        # desired_out = BlockReLU_V1(self.block_sizes)(activation + activation_server)
        assert activation.dtype == self.signed_type
        reshaped_inputs = []
        mean_tensors = []
        channels = []
        orig_shapes = []

        for block_size in self.active_block_sizes:
            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = activation[:, cur_channels]
            # reshaped_input[0,1,38,31]
            reshaped_input = SpaceToDepth(block_size)(cur_input)
            assert reshaped_input.dtype == self.signed_type
            mean_tensor = np.sum(reshaped_input, axis=-1, keepdims=True)

            channels.append(cur_channels)
            reshaped_inputs.append(reshaped_input)
            orig_shapes.append(mean_tensor.shape)
            mean_tensors.append(mean_tensor.flatten())

        cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
        mean_tensors = np.concatenate(mean_tensors)
        assert mean_tensors.dtype == self.signed_type
        activation = activation.astype(self.dtype)
        sign_tensors = self.DReLU(mean_tensors.astype(self.dtype))

        relu_map = np.ones_like(activation)
        for i in range(len(self.active_block_sizes)):
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shapes[i])
            relu_map[:, channels[i]] = DepthToSpace(self.active_block_sizes[i])(
                sign_tensor.repeat(reshaped_inputs[i].shape[-1], axis=-1))

        activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels],
                                                              activation[:, ~self.is_identity_channels])
        activation = activation.astype(self.signed_type)
        # real_out = network_assets.receiver_01.get() + activation
        # assert np.all(real_out == desired_out.numpy())
        return torch.from_numpy(activation)




def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False):

    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DClient

    if is_prf_fetcher:
        W = np.zeros(shape=conv_module.weight.shape, dtype=np.int64)
    else:
        dtype = np.int64
        W = crypto_assets[CLIENT, SERVER].integers(low=np.iinfo(dtype).min // 2,
                                                   high=np.iinfo(dtype).max // 2,
                                                   size=conv_module.weight.shape,
                                                   dtype=dtype)

    return conv_class(
        W=W,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        groups=conv_module.groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def build_secure_relu(crypto_assets, network_assets, is_prf_fetcher=False):
    relu_class = PRFFetcherReLU if is_prf_fetcher else SecureReLUClient
    return relu_class(crypto_assets=crypto_assets, network_assets=network_assets)


class SecureModel(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(SecureModel, self).__init__( crypto_assets, network_assets)
        self.model = model

    def forward(self, img, img_meta):
        dtype = np.int64

        I = TypeConverter.f2i(img)
        I1 = torch.from_numpy(self.prf_handler[CLIENT, SERVER].integers(low=np.iinfo(dtype).min // 2, high=np.iinfo(dtype).max // 2, dtype=dtype, size=img.shape))
        I0 = I - I1
        out_0 = self.model.decode_head(self.model.backbone(I0))
        out_1 = self.network_assets.receiver_01.get()
        out = (torch.from_numpy(out_1) + out_0)
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
            seg_pred = model(img, img_meta)

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



if __name__ == "__main__":

    config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py"
    secure_config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline_secure.py"
    model_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth"
    relu_spec_file = None
    image_shape = (1, 3, 256, 256)
    num_images = 50

    model = get_model(
        config=secure_config_path,
        gpu_id=None,
        checkpoint_path=None
    )

    compile_numba_funcs()
    crypto_assets, network_assets = get_assets(0, repeat=num_images)

    model = securify_mobilenetv2_model(
        model,
        build_secure_conv=partial(build_secure_conv, crypto_assets=crypto_assets, network_assets=network_assets),
        build_secure_relu=partial(build_secure_relu, crypto_assets=crypto_assets, network_assets=network_assets),
        secure_model_class=partial(SecureModel, crypto_assets=crypto_assets, network_assets=network_assets),
        block_relu=partial(SecureBlockReLUClient, crypto_assets=crypto_assets, network_assets=network_assets),
        relu_spec_file=relu_spec_file)

    prf_fetcher_model = get_model(
        config=secure_config_path,
        gpu_id=None,
        checkpoint_path=None
    )

    prf_fetcher_model = securify_mobilenetv2_model(
        prf_fetcher_model,
        build_secure_conv=partial(build_secure_conv, crypto_assets=crypto_assets, network_assets=network_assets, is_prf_fetcher=True),
        build_secure_relu=partial(build_secure_relu, crypto_assets=crypto_assets, network_assets=network_assets, is_prf_fetcher=True),
        secure_model_class=partial(PRFFetcherSecureModel, crypto_assets=crypto_assets, network_assets=network_assets),
        block_relu=partial(SecureBlockReLUClient, crypto_assets=crypto_assets, network_assets=network_assets),
        relu_spec_file=relu_spec_file)

    prf_fetcher_model.prf_handler.fetch(repeat=num_images, model=prf_fetcher_model, image=torch.zeros(size=image_shape, dtype=torch.int64))

    full_inference(model, num_images)

    network_assets.done()
    
# SecureReLUClient Elapsed time: 3.2032 seconds. Total elapsed time: 112.7030 seconds.
# Integers - fetch Elapsed time: 0.0000 seconds. Total elapsed time: 0.1586 seconds.
# SecureConv2DClient Elapsed time: 0.0916 seconds. Total elapsed time: 12.8702 seconds.
# Inference Elapsed time: 125.5989 seconds. Total elapsed time: 125.5989 seconds.