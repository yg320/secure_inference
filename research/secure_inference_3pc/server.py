import torch
import numpy as np
from research.secure_inference_3pc.base import fuse_conv_bn, decompose, get_c_party_1, module_67,  get_assets, TypeConverter, get_c_party_1_torch, decompose_torch_1
from research.secure_inference_3pc.conv2d import conv_2d
from research.secure_inference_3pc.modules.conv2d import get_output_shape
from research.secure_inference_3pc.conv2d_torch import Conv2DHandler
from research.secure_inference_3pc.base import SecureModule, NetworkAssets
from research.secure_inference_3pc.timer import Timer
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.distortion.utils import get_model
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE
from research.pipeline.backbones.secure_resnet import AvgPoolResNet
from research.pipeline.backbones.secure_aspphead import SecureASPPHead
from research.secure_inference_3pc.resnet_converter import get_secure_model, init_prf_fetcher
from research.secure_inference_3pc.params import Params
from research.secure_inference_3pc.modules.server import PRFFetcherConv2D, PRFFetcherReLU, PRFFetcherMaxPool, PRFFetcherSecureModelSegmentation, PRFFetcherSecureModelClassification, PRFFetcherBlockReLU
from functools import partial
from research.bReLU import NumpySecureOptimizedBlockReLU
import mmcv
from research.mmlab_extension.resnet_cifar_v2 import ResNet_CIFAR_V2
from research.mmlab_extension.classification.resnet import AvgPoolResNet, MyResNet

class SecureConv2DServer(SecureModule):
    def __init__(self, W, bias, stride, dilation, padding, groups, crypto_assets, network_assets: NetworkAssets, device="cpu"):
        super(SecureConv2DServer, self).__init__(crypto_assets, network_assets)

        self.W_plaintext = W
        self.bias = bias
        if self.bias is not None:
            self.bias = self.bias[np.newaxis, :, np.newaxis, np.newaxis]
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.conv2d_handler = Conv2DHandler("cuda:1")
        self.device = device

    def forward(self, X_share):
        # return np.zeros(get_output_shape(X_share, self.W_share, self.padding, self.dilation, self.stride), dtype=X_share.dtype)
        W_client = crypto_assets[CLIENT, SERVER].integers(low=MIN_VAL,
                                                          high=MAX_VAL,
                                                          size=self.W_plaintext.shape,
                                                          dtype=SIGNED_DTYPE)

        self.W_share = self.W_plaintext - W_client
        assert X_share.dtype == SIGNED_DTYPE

        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert (self.W_share.shape[1] == X_share.shape[1]) or self.groups > 1
        assert self.stride[0] == self.stride[1]

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_share.shape, dtype=SIGNED_DTYPE)

        E_share = X_share - A_share
        F_share = self.W_share - B_share


        self.network_assets.sender_01.put(np.concatenate([E_share.flatten(), F_share.flatten()]))
        share_client = self.network_assets.receiver_01.get()

        E_share_client, F_share_client = \
            share_client[:E_share.size].reshape(E_share.shape), \
                share_client[E_share.size:].reshape(F_share.shape)

        E = E_share_client + E_share
        F = F_share_client + F_share

        new_weight = self.W_share - F
        # out_numpy =  np.zeros(get_output_shape(X_share, self.W_share, self.padding, self.dilation, self.stride), dtype=X_share.dtype)
        if self.device == "cpu":
            out_numpy = conv_2d(E, new_weight, X_share, F, self.padding, self.stride, self.dilation, self.groups)
        else:
            out_numpy = self.conv2d_handler.conv2d(E, new_weight, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
            out_numpy += self.conv2d_handler.conv2d(X_share, F, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=out_numpy.shape, dtype=SIGNED_DTYPE)

        out = out_numpy + C_share
        out = out // self.trunc
        # This is the proper way, but it's slower and takes more time
        # t = out.dtype
        # out = (out / self.trunc).round().astype(t)
        if self.bias is not None:
            out = out + self.bias

        mu_1 = -self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=out.dtype)
        return out + mu_1


class PrivateCompareServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PrivateCompareServer, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_1, r, beta):
        # with Timer("PrivateCompareServer"):
        return self._forward(x_bits_1, r, beta)
    def _forward(self, x_bits_1, r, beta):
        # r = r.astype(np.int64)

        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_1.shape, dtype=np.int32)
        # u = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_1.shape, dtype=self.crypto_assets.numpy_dtype)

        r[beta] += 1
        bits = decompose(r)

        c_bits_1 = get_c_party_1(x_bits_1, bits, beta, np.int8(1))

        np.multiply(s, c_bits_1, out=s)

        d_bits_1 = module_67(s)
        # r = torch.from_numpy(r).to("cuda:1")
        # s = torch.from_numpy(s).to("cuda:1")
        # beta = torch.from_numpy(beta).to("cuda:1")
        # x_bits_1 = torch.from_numpy(x_bits_1).to("cuda:1")
        #
        # r[beta.to(torch.int64)] += 1
        # bits = decompose_torch_1(r)
        # c_bits_0 = get_c_party_1_torch(x_bits_1, bits, beta)
        # torch.mul(s, c_bits_0, out=s)
        # d_bits_1 = s % 67

        d_bits_1 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_1, axis=-1)
        # d_bits_1 = d_bits_1.cpu().numpy().astype(np.uint8)

        self.network_assets.sender_12.put(d_bits_1)


class ShareConvertServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertServer, self).__init__(crypto_assets, network_assets)
        self.private_compare = PrivateCompareServer(crypto_assets, network_assets)

    def forward(self, a_1):
        # with Timer("ShareConvertServer"):
        return self._forward(a_1)
    def _forward(self, a_1):

        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=np.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=SIGNED_DTYPE)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=SIGNED_DTYPE)
        mu_1 = -self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)
        r_1 = r - r_0
        a_tild_1 = a_1 + r_1
        beta_1 = (0 < a_1 - a_tild_1).astype(SIGNED_DTYPE)

        self.network_assets.sender_12.put(a_tild_1)
        # x_bits_1 = torch.from_numpy(self.network_assets.receiver_12.get().astype(np.int8)).to("cuda:1")
        x_bits_1 = self.network_assets.receiver_12.get().astype(np.int8)

        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)

        self.private_compare(x_bits_1, r - 1, eta_pp)
        eta_p_1 = self.network_assets.receiver_12.get()

        eta_pp = eta_pp.astype(SIGNED_DTYPE)
        t00 = eta_pp * eta_p_1
        t11 = self.add_mode_L_minus_one(t00, t00)
        eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)
        t00 = self.add_mode_L_minus_one(delta_1, eta_1)
        theta_1 = self.add_mode_L_minus_one(beta_1, t00)
        y_1 = self.sub_mode_L_minus_one(a_1, theta_1)
        y_1 = self.add_mode_L_minus_one(y_1, mu_1)
        return y_1


class SecureMultiplicationServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationServer, self).__init__(crypto_assets, network_assets)

    def forward(self, X_share, Y_share):

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        E_share = X_share - A_share
        F_share = Y_share - B_share

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        E = E_share_client + E_share
        F = F_share_client + F_share
        out = - E * F + X_share * F + Y_share * E + C_share
        mu_1 = -self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=X_share.dtype)

        return out + mu_1


class SecureMSBServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBServer, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)
        self.private_compare = PrivateCompareServer(crypto_assets, network_assets)

    def forward(self, a_1):

        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=np.int8)
        x_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)
        mu_1 = -self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=a_1.dtype)

        x_bits_1 = self.network_assets.receiver_12.get()
        x_bit_0_1 = self.network_assets.receiver_12.get()

        y_1 = self.add_mode_L_minus_one(a_1, a_1)
        r_1 = self.add_mode_L_minus_one(x_1, y_1)

        self.network_assets.sender_01.put(r_1)
        r_0 = self.network_assets.receiver_01.get()

        r = self.add_mode_L_minus_one(r_0, r_1)
        r_mod_2 = r % 2

        self.private_compare(x_bits_1, r, beta)

        beta_p_1 = self.network_assets.receiver_12.get()

        beta = beta.astype(SIGNED_DTYPE)
        gamma_1 = beta_p_1 + (1 * beta) - (2 * beta * beta_p_1)
        delta_1 = x_bit_0_1 + r_mod_2 - (2 * r_mod_2 * x_bit_0_1)

        theta_1 = self.mult(gamma_1, delta_1)

        alpha_1 = gamma_1 + delta_1 - 2 * theta_1
        alpha_1 = alpha_1 + mu_1

        return alpha_1


class SecureDReLUServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUServer, self).__init__(crypto_assets, network_assets)

        self.share_convert = ShareConvertServer(crypto_assets, network_assets)
        self.msb = SecureMSBServer(crypto_assets, network_assets)

    def forward(self, X_share):

        mu_1 = -self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=X_share.dtype)

        X1_converted = self.share_convert(X_share)
        MSB_1 = self.msb(X1_converted)
        return 1 - MSB_1 + mu_1


class SecureReLUServer(SecureModule):
    def __init__(self, crypto_assets, network_assets, dummy_relu=False):
        super(SecureReLUServer, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUServer(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        if self.dummy_relu:
            share_client = self.network_assets.receiver_01.get()
            value = share_client + X_share
            value = value * ((value > 0).astype(value.dtype))
            return value
        else:

            shape = X_share.shape
            mu_1 = -self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

            X_share = X_share.flatten()
            MSB_0 = self.DReLU(X_share)
            relu_0 = self.mult(X_share, MSB_0).reshape(shape)
            ret = relu_0.astype(SIGNED_DTYPE)
            return ret + mu_1


class SecureBlockReLUServer(SecureModule, NumpySecureOptimizedBlockReLU):
    def __init__(self, block_sizes, crypto_assets, network_assets, dummy_relu=False):
        SecureModule.__init__(self, crypto_assets=crypto_assets, network_assets=network_assets)
        NumpySecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.secure_DReLU = SecureDReLUServer(crypto_assets, network_assets)
        self.secure_mult = SecureMultiplicationServer(crypto_assets, network_assets)

        self.dummy_relu = dummy_relu

    def mult(self, x, y):
        return self.secure_mult(x.astype(SIGNED_DTYPE), y.astype(SIGNED_DTYPE)).astype(x.dtype)

    def DReLU(self, activation):
        return self.secure_DReLU(activation)

    def forward(self, activation):
        if self.dummy_relu:
            activation_client = self.network_assets.receiver_01.get()
            activation = activation + activation_client

        activation = NumpySecureOptimizedBlockReLU.forward(self, activation)
        activation = activation.astype(SIGNED_DTYPE)

        return activation


class SecureSelectShareServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureSelectShareServer, self).__init__(crypto_assets, network_assets)
        self.secure_multiplication = SecureMultiplicationServer(crypto_assets, network_assets)

    def forward(self, alpha, x, y):
        dtype = alpha.dtype
        shape = alpha.shape
        mu_1 = -self.prf_handler[CLIENT, SERVER].integers(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)

        w = y - x
        c = self.secure_multiplication(alpha, w)
        z = x + c
        return z + mu_1


class SecureMaxPoolServer(SecureMaxPool):
    def __init__(self, kernel_size, stride, padding, crypto_assets, network_assets, dummy_max_pool):
        super(SecureMaxPoolServer, self).__init__(kernel_size, stride, padding, crypto_assets, network_assets, dummy_max_pool)
        self.select_share = SecureSelectShareServer(crypto_assets, network_assets)
        self.dReLU = SecureDReLUServer(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)

    def forward(self, x):
        if self.dummy_max_pool:
            x_client = self.network_assets.receiver_01.get()
            x_rec = x_client + x
            return torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)(torch.from_numpy(x_rec).to(torch.float64)).numpy().astype(x.dtype)

        ret = super(SecureMaxPoolServer, self).forward(x)
        mu_1 = -self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)
        ret = ret + mu_1
        return ret


def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False, device="cpu"):
    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DServer

    if bn_module:
        W, B = fuse_conv_bn(conv_module=conv_module, batch_norm_module=bn_module)
        W = TypeConverter.f2i(W)
        B = TypeConverter.f2i(B)

    else:
        W = conv_module.weight
        assert conv_module.bias is None
        W = TypeConverter.f2i(W)
        B = None

    return conv_class(
        W=W,
        bias=B,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        groups=conv_module.groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        device=device
    )

def build_secure_fully_connected(crypto_assets, network_assets, conv_module, bn_module, is_prf_fetcher=False):
    conv_class = PRFFetcherConv2D if is_prf_fetcher else SecureConv2DServer

    assert bn_module is None
    stride = (1, 1)
    dilation = (1, 1)
    padding = (0, 0)
    groups = 1

    W = TypeConverter.f2i(conv_module.weight.unsqueeze(2).unsqueeze(3))
    B = TypeConverter.f2i(conv_module.bias)

    return conv_class(
        W=W,
        bias=B,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def build_secure_relu(crypto_assets, network_assets, is_prf_fetcher=False, dummy_relu=False):
    relu_class = PRFFetcherReLU if is_prf_fetcher else SecureReLUServer
    return relu_class(crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu)



class SecureModelSegmentation(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(SecureModelSegmentation, self).__init__( crypto_assets, network_assets)
        self.model = model

    def forward(self, image_shape):

        image = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL,
                                                          high=MAX_VAL,
                                                          size=image_shape,
                                                          dtype=SIGNED_DTYPE)
        out = self.model.decode_head(self.model.backbone(image))
        self.network_assets.sender_01.put(out)

class SecureModelClassification(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(SecureModelClassification, self).__init__( crypto_assets, network_assets)
        self.model = model

    def forward(self, image_shape):

        image = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL,
                                                          high=MAX_VAL,
                                                          size=image_shape,
                                                          dtype=SIGNED_DTYPE)

        out = self.model.backbone(image)[0]
        out = self.model.neck(out)
        out = self.model.head.fc(out)
        self.network_assets.sender_01.put(out)


if __name__ == "__main__":
    party = 1
    cfg = mmcv.Config.fromfile(Params.SECURE_CONFIG_PATH)

    crypto_assets, network_assets = get_assets(party, repeat=Params.NUM_IMAGES, simulated_bandwidth=Params.SIMULATED_BANDWIDTH)

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
            dummy_max_pool=Params.DUMMY_MAX_POOL)
    else:
        prf_fetcher = None

    model = get_secure_model(
        cfg,
        checkpoint_path=Params.MODEL_PATH,
        build_secure_conv=build_secure_conv,
        build_secure_relu=build_secure_relu,
        build_secure_fully_connected=build_secure_fully_connected,
        max_pool=SecureMaxPoolServer,
        secure_model_class=SecureModelSegmentation if cfg.model.type == "EncoderDecoder" else SecureModelClassification,
        block_relu=SecureBlockReLUServer,
        relu_spec_file=Params.RELU_SPEC_FILE,
        crypto_assets=crypto_assets,
        network_assets=network_assets,
        dummy_relu=Params.DUMMY_RELU,
        dummy_max_pool=Params.DUMMY_MAX_POOL,
        prf_fetcher=prf_fetcher,
        device=Params.DEVICE

    )
    if model.prf_fetcher:
        model.prf_fetcher.prf_handler.fetch(repeat=Params.NUM_IMAGES, model=model.prf_fetcher,
                                            image=np.zeros(shape=Params.IMAGE_SHAPE, dtype=SIGNED_DTYPE))

    for _ in range(Params.NUM_IMAGES):

        out = model(Params.IMAGE_SHAPE)

    network_assets.done()

    print("Num of bytes sent 1 -> 0", network_assets.sender_01.num_of_bytes_sent)
    print("Num of bytes sent 1 -> 2", network_assets.sender_12.num_of_bytes_sent)