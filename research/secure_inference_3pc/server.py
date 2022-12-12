import torch
from research.communication.utils import Sender, Receiver
import time
import numpy as np
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets, fuse_conv_bn, pre_conv, post_conv, mat_mult, Addresses, decompose, P, get_c, get_c_case_2, module_67
from scipy.signal import fftconvolve

from research.communication.utils import Sender, Receiver
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets

class SecureConv2DServer(SecureModule):
    def __init__(self, W, bias, stride, dilation, padding, crypto_assets: CryptoAssets, network_assets: NetworkAssets):
        super(SecureConv2DServer, self).__init__(crypto_assets, network_assets)

        self.W_share = W.numpy()
        self.bias = bias
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):
        X_share = X_share.numpy()

        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert self.W_share.shape[1] == X_share.shape[1]
        assert self.W_share.shape[1] == X_share.shape[1]
        assert self.stride[0] == self.stride[1]

        A_share = self.crypto_assets.prf_12_numpy.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=X_share.shape, dtype=np.int64)
        B_share = self.crypto_assets.prf_12_numpy.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=self.W_share.shape, dtype=np.int64)

        E_share = X_share - A_share
        F_share = self.W_share - B_share

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        E = E_share_client + E_share
        F = F_share_client + F_share

        new_weight = self.W_share - F

        E_numpy, new_weight_numpy, batch_size, nb_channels_out, nb_rows_out, nb_cols_out = pre_conv(E, new_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
        X_share_numpy, F_numpy, _, _, _, _ = pre_conv(X_share, F, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

        out_numpy = mat_mult(E_numpy[0], new_weight_numpy, X_share_numpy[0], F_numpy)
        out_numpy = out_numpy[np.newaxis]
        out_numpy = post_conv(None, out_numpy, batch_size, nb_channels_out, nb_rows_out, nb_cols_out)

        C_share = self.crypto_assets.prf_12_numpy.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=out_numpy.shape, dtype=np.int64)

        out = out_numpy + C_share
        out = torch.from_numpy(out)

        out = out // self.trunc
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return out

#
# class SecureConv2DServer_V2(SecureModule):
#     def __init__(self, W, bias, stride, dilation, padding, crypto_assets: CryptoAssets, network_assets: NetworkAssets):
#         super(SecureConv2DServer, self).__init__(crypto_assets, network_assets)
#
#         self.W_share = W
#         self.bias = bias
#         self.stride = stride
#         self.dilation = dilation
#         self.padding = padding
#
#     def forward(self, X_share):
#
#         assert self.W_share.shape[2] == self.W_share.shape[3]
#         if self.W_share.shape[1] != X_share.shape[1]:
#             print('fds')
#         assert self.W_share.shape[1] == X_share.shape[1]
#         # assert X_share.shape[2] == X_share.shape[3]
#         assert self.stride[0] == self.stride[1]
#
#         A_share = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_12_torch)
#         B_share = self.crypto_assets.get_random_tensor_over_L(shape=self.W_share.shape, prf=self.crypto_assets.prf_12_torch)
#
#         E_share = X_share - A_share
#         F_share = self.W_share - B_share
#
#         self.network_assets.sender_01.put(E_share)
#         E_share_client = torch.from_numpy(self.network_assets.receiver_01.get())
#         self.network_assets.sender_01.put(F_share)
#         F_share_client = torch.from_numpy(self.network_assets.receiver_01.get())
#
#         E = E_share_client + E_share
#         F = F_share_client + F_share
#
#         new_weight = self.W_share - F
#         out = \
#             torch.conv2d(E, new_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1) + \
#             torch.conv2d(X_share, F, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
#
#         C_share = self.crypto_assets.get_random_tensor_over_L(shape=out.shape, prf=self.crypto_assets.prf_12_torch)
#
#         out = out + C_share
#         out = out // self.trunc
#         if self.bias is not None:
#             out = out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
#         return out



class PrivateCompareServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PrivateCompareServer, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_1, r, beta):
        t0 = time.time()

        s = self.crypto_assets.prf_01_numpy.integers(low=1, high=67, size=x_bits_1.shape, dtype=np.int32)
        # u = self.crypto_assets.prf_01_numpy.integers(low=1, high=67, size=x_bits_1.shape, dtype=self.crypto_assets.numpy_dtype)

        r[beta] += 1
        bits = decompose(r)

        c_bits_1 = get_c(x_bits_1, bits, beta, np.int8(1))

        np.multiply(s, c_bits_1, out=s)

        d_bits_1 = module_67(s)

        d_bits_1 = self.crypto_assets.prf_01_numpy.permutation(d_bits_1, axis=-1)
        t1 = time.time()

        print("**********PrivateCompareServer***************")
        print(t1 - t0)

        self.network_assets.sender_12.put(d_bits_1)


class ShareConvertServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertServer, self).__init__(crypto_assets, network_assets)
        self.private_compare = PrivateCompareServer(crypto_assets, network_assets)

    def forward(self, a_1):
        t0 = time.time()
        eta_pp = self.crypto_assets.prf_01_numpy.integers(0, 2, size=a_1.shape, dtype=np.int8)
        r = self.crypto_assets.prf_01_numpy.integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
        r_0 = self.crypto_assets.prf_01_numpy.integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
        r_1 = r - r_0
        a_tild_1 = a_1 + r_1
        beta_1 = (a_tild_1 < a_1).astype(self.dtype)
        t1 = time.time()

        self.network_assets.sender_12.put(a_tild_1)
        x_bits_1 = self.network_assets.receiver_12.get().astype(np.int8)

        t2 = time.time()
        delta_1 = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val, size=a_1.shape, dtype=self.dtype)
        t3 = time.time()

        self.private_compare(x_bits_1, r - 1, eta_pp)
        eta_p_1 = self.network_assets.receiver_12.get()

        t4 = time.time()
        eta_pp = eta_pp.astype(self.dtype)
        t00 = eta_pp * eta_p_1
        t11 = self.add_mode_L_minus_one(t00, t00)
        eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)
        t00 = self.add_mode_L_minus_one(delta_1, eta_1)
        theta_1 = self.add_mode_L_minus_one(beta_1, t00)
        y_1 = self.sub_mode_L_minus_one(a_1, theta_1)
        t5 = time.time()

        print("**********ShareConvertServer***************")
        print(t5 - t4 + t3 - t2 + t1 - t0)
        return y_1


class SecureMultiplicationServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationServer, self).__init__(crypto_assets, network_assets)

    def forward(self, X_share, Y_share):
        t0 = time.time()
        A_share = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        B_share = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        C_share = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        E_share = X_share - A_share
        F_share = Y_share - B_share
        t1 = time.time()

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        t2 = time.time()
        E = E_share_client + E_share
        F = F_share_client + F_share
        out = - E * F + X_share * F + Y_share * E + C_share
        t3 = time.time()

        print("**********SecureMultiplicationServer***************")
        print(t3 - t2 + t1 - t0)
        return out


class SecureMSBServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBServer, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)
        self.private_compare = PrivateCompareServer(crypto_assets, network_assets)

    def forward(self, a_1):
        t0 = time.time()
        beta = self.crypto_assets.prf_01_numpy.integers(0, 2, size=a_1.shape, dtype=np.int8)
        x_1 = self.crypto_assets.prf_12_numpy.integers(self.min_val, self.max_val, size=a_1.shape, dtype=self.dtype)
        t1 = time.time()

        x_bits_1 = self.network_assets.receiver_12.get()
        x_bit_0_1 = self.network_assets.receiver_12.get()

        t2 = time.time()
        y_1 = self.add_mode_L_minus_one(a_1, a_1)
        r_1 = self.add_mode_L_minus_one(x_1, y_1)
        t3 = time.time()

        self.network_assets.sender_01.put(r_1)
        r_0 = self.network_assets.receiver_01.get()

        t4 = time.time()
        r = self.add_mode_L_minus_one(r_0, r_1)
        t5 = time.time()
        r_mod_2 = r % 2
        self.private_compare(x_bits_1, r, beta)

        beta_p_1 = self.network_assets.receiver_12.get()

        t6 = time.time()
        beta = beta.astype(self.dtype)
        gamma_1 = beta_p_1 + (1 * beta) - (2 * beta * beta_p_1)
        delta_1 = x_bit_0_1 + r_mod_2 - (2 * r_mod_2 * x_bit_0_1)
        t7 = time.time()

        theta_1 = self.mult(gamma_1, delta_1)

        t8 = time.time()
        alpha_1 = gamma_1 + delta_1 - 2 * theta_1
        t9 = time.time()
        print("***************SecureMSBServer******************8")
        print(t1 - t0 + t3 - t2 + t5 - t4 + t7 - t6 + t9 - t8)
        return alpha_1


class SecureDReLUServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUServer, self).__init__(crypto_assets, network_assets)

        self.share_convert = ShareConvertServer(crypto_assets, network_assets)
        self.msb = SecureMSBServer(crypto_assets, network_assets)

    def forward(self, X_share):
        X1_converted = self.share_convert(X_share)
        MSB_1 = self.msb(X1_converted)
        return 1 - MSB_1

class SecureReLUServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUServer, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUServer(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)

    def forward(self, X_share):
        shape = X_share.shape
        X_share = X_share.numpy().astype(self.dtype).flatten()
        MSB_0 = self.DReLU(X_share)
        relu_0 = self.mult(X_share, MSB_0).reshape(shape)
        return torch.from_numpy(relu_0.astype(self.signed_type))

def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module):
    if bn_module:
        W, B = fuse_conv_bn(conv_module=conv_module, batch_norm_module=bn_module)
        W = (W * crypto_assets.trunc).to(crypto_assets.torch_dtype)
        B = (B * crypto_assets.trunc).to(crypto_assets.torch_dtype)
        W = W - crypto_assets.get_random_tensor_over_L(W.shape, prf=crypto_assets.prf_01_torch)
    else:
        W = conv_module.weight
        W = (W * crypto_assets.trunc).to(crypto_assets.torch_dtype)
        W = W - crypto_assets.get_random_tensor_over_L(W.shape, prf=crypto_assets.prf_01_torch)
        B = None

    return SecureConv2DServer(
        W=W,
        bias=B,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def aspp_head_secure_forward(self, inputs):
    """Forward function for feature maps before classifying each pixel with
    ``self.cls_seg`` fc.

    Args:
        inputs (list[Tensor]): List of multi-level img features.

    Returns:
        feats (Tensor): A tensor of shape (batch_size, self.channels,
            H, W) which is feature map for last layer of decoder head.
    """
    x = self._transform_inputs(inputs)

    aspp_outs = [
        torch.ones_like(x) * x.sum(dim=[2, 3], keepdims=True) // (x.shape[2] * x.shape[3])
    ]
    aspp_outs.extend(self.aspp_modules(x))
    aspp_outs = torch.cat(aspp_outs, dim=1)
    feats = self.bottleneck(aspp_outs)
    return feats

if __name__ == "__main__":
    from research.distortion.utils import get_model
    from research.pipeline.backbones.secure_resnet import AvgPoolResNet

    config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16.py"
    image_path = "/home/yakir/tmp/image_0.pt"
    model_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/iter_80000.pth"

    image_shape = (1, 3, 192, 256)

    model = get_model(
        config=config_path,
        gpu_id=None,
        checkpoint_path=model_path
    )
    addresses = Addresses()
    port_01 = addresses.port_01
    port_10 = addresses.port_10
    port_02 = addresses.port_02
    port_20 = addresses.port_20
    port_12 = addresses.port_12
    port_21 = addresses.port_21


    prf_01_seed = 0
    prf_02_seed = 1
    prf_12_seed = 2

    sender_01 = Sender(port_10)
    sender_02 = None
    sender_12 = Sender(port_12)
    receiver_01 = Receiver(port_01)
    receiver_02 = None
    receiver_12 = Receiver(port_21)

    prf_01_numpy = np.random.default_rng(seed=prf_01_seed)
    prf_02_numpy = None
    prf_12_numpy = np.random.default_rng(seed=prf_12_seed)
    prf_01_torch = torch.Generator().manual_seed(prf_01_seed)
    prf_02_torch = None
    prf_12_torch = torch.Generator().manual_seed(prf_12_seed)

    crypto_assets = CryptoAssets(
        prf_01_numpy=prf_01_numpy,
        prf_02_numpy=prf_02_numpy,
        prf_12_numpy=prf_12_numpy,
        prf_01_torch=prf_01_torch,
        prf_02_torch=prf_02_torch,
        prf_12_torch=prf_12_torch,
    )

    network_assets = NetworkAssets(
        sender_01=sender_01,
        sender_02=sender_02,
        sender_12=sender_12,
        receiver_01=receiver_01,
        receiver_02=receiver_02,
        receiver_12=receiver_12
    )

    model.backbone.stem[0] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[0], model.backbone.stem[1])
    model.backbone.stem[1] = torch.nn.Identity()
    model.backbone.stem[2] = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[3] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[3], model.backbone.stem[4])
    model.backbone.stem[4] = torch.nn.Identity()
    model.backbone.stem[5] = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[6] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[6], model.backbone.stem[7])
    model.backbone.stem[7] = torch.nn.Identity()
    model.backbone.stem[8] = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

    for layer in [1, 2, 3, 4]:
        for block in [0, 1]:
            cur_res_layer = getattr(model.backbone, f"layer{layer}")
            cur_res_layer[block].conv1 = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].conv1, cur_res_layer[block].bn1)
            cur_res_layer[block].bn1 = torch.nn.Identity()
            cur_res_layer[block].relu_1 = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

            cur_res_layer[block].conv2 = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].conv2, cur_res_layer[block].bn2)
            cur_res_layer[block].bn2 = torch.nn.Identity()
            cur_res_layer[block].relu_2 = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

            if cur_res_layer[block].downsample:
                cur_res_layer[block].downsample = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].downsample[0], cur_res_layer[block].downsample[1])

    model.decode_head.image_pool[1].conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.image_pool[1].conv, model.decode_head.image_pool[1].bn)
    model.decode_head.image_pool[1].bn = torch.nn.Identity()
    model.decode_head.image_pool[1].activate = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

    for i in range(4):
        model.decode_head.aspp_modules[i].conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.aspp_modules[i].conv, model.decode_head.aspp_modules[i].bn)
        model.decode_head.aspp_modules[i].bn = torch.nn.Identity()
        model.decode_head.aspp_modules[i].activate = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

    model.decode_head.bottleneck.conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.bottleneck.conv, model.decode_head.bottleneck.bn)
    model.decode_head.bottleneck.bn = torch.nn.Identity()
    model.decode_head.bottleneck.activate = SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets)

    model.decode_head.conv_seg = build_secure_conv(crypto_assets, network_assets, model.decode_head.conv_seg, None)
    model.decode_head.image_pool[0].forward = lambda x: x.sum(dim=[2, 3], keepdims=True) // (x.shape[2] * x.shape[3])



    I1 = crypto_assets.get_random_tensor_over_L(image_shape, prf=crypto_assets.prf_01_torch)
    import time
    time.sleep(5)
    print("Start")

    image = I1
    # out = model.decode_head(model.backbone(image))
    out = model.decode_head(model.backbone(image))
    # out = model.backbone.stem(image)

    network_assets.sender_01.put(out)

