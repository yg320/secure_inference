import torch
import numpy as np
from research.secure_inference_3pc.base import fuse_conv_bn, decompose, get_c, module_67, DepthToSpace, SpaceToDepth, get_assets, TypeConverter
from research.secure_inference_3pc.conv2d import conv_2d

from research.secure_inference_3pc.base import SecureModule, NetworkAssets

from research.distortion.utils import get_model
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER
from research.pipeline.backbones.secure_resnet import AvgPoolResNet
from research.pipeline.backbones.secure_aspphead import SecureASPPHead
from research.secure_inference_3pc.resnet_converter import securify_mobilenetv2_model
from research.secure_inference_3pc.params import Params
from functools import partial

class SecureConv2DServer(SecureModule):
    def __init__(self, W, bias, stride, dilation, padding, groups, crypto_assets, network_assets: NetworkAssets):
        super(SecureConv2DServer, self).__init__(crypto_assets, network_assets)

        self.W_share = W.numpy()
        self.bias = bias
        if self.bias is not None:
            self.bias = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).numpy()
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

    def forward(self, X_share):

        X_share = X_share.numpy()
        assert X_share.dtype == self.signed_type

        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert (self.W_share.shape[1] == X_share.shape[1]) or self.groups > 1
        assert self.stride[0] == self.stride[1]

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=X_share.shape, dtype=np.int64)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=self.W_share.shape, dtype=np.int64)

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

        out_numpy = conv_2d(E, new_weight, X_share, F, self.padding, self.stride, self.dilation, self.groups)


        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=out_numpy.shape, dtype=np.int64)

        out = out_numpy + C_share

        out = out // self.trunc
        if self.bias is not None:
            out = out + self.bias
        return torch.from_numpy(out)


class PrivateCompareServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PrivateCompareServer, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_1, r, beta):

        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_1.shape, dtype=np.int32)
        # u = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_1.shape, dtype=self.crypto_assets.numpy_dtype)

        r[beta] += 1
        bits = decompose(r)

        c_bits_1 = get_c(x_bits_1, bits, beta, np.int8(1))

        np.multiply(s, c_bits_1, out=s)

        d_bits_1 = module_67(s)

        d_bits_1 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_1, axis=-1)

        self.network_assets.sender_12.put(d_bits_1)


class ShareConvertServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertServer, self).__init__(crypto_assets, network_assets)
        self.private_compare = PrivateCompareServer(crypto_assets, network_assets)

    def forward(self, a_1):

        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=np.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_1.shape, dtype=self.dtype)
        r_1 = r - r_0
        a_tild_1 = a_1 + r_1
        beta_1 = (a_tild_1 < a_1).astype(self.dtype)


        self.network_assets.sender_12.put(a_tild_1)
        x_bits_1 = self.network_assets.receiver_12.get().astype(np.int8)

        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(self.min_val, self.max_val, size=a_1.shape, dtype=self.dtype)

        self.private_compare(x_bits_1, r - 1, eta_pp)
        eta_p_1 = self.network_assets.receiver_12.get()

        eta_pp = eta_pp.astype(self.dtype)
        t00 = eta_pp * eta_p_1
        t11 = self.add_mode_L_minus_one(t00, t00)
        eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)
        t00 = self.add_mode_L_minus_one(delta_1, eta_1)
        theta_1 = self.add_mode_L_minus_one(beta_1, t00)
        y_1 = self.sub_mode_L_minus_one(a_1, theta_1)

        return y_1


class SecureMultiplicationServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationServer, self).__init__(crypto_assets, network_assets)

    def forward(self, X_share, Y_share):
        assert X_share.dtype == self.dtype
        assert Y_share.dtype == self.dtype

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        E_share = X_share - A_share
        F_share = Y_share - B_share

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        E = E_share_client + E_share
        F = F_share_client + F_share
        out = - E * F + X_share * F + Y_share * E + C_share

        return out


class SecureMSBServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBServer, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)
        self.private_compare = PrivateCompareServer(crypto_assets, network_assets)

    def forward(self, a_1):
        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=np.int8)
        x_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(self.min_val, self.max_val, size=a_1.shape, dtype=self.dtype)

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

        beta = beta.astype(self.dtype)
        gamma_1 = beta_p_1 + (1 * beta) - (2 * beta * beta_p_1)
        delta_1 = x_bit_0_1 + r_mod_2 - (2 * r_mod_2 * x_bit_0_1)

        theta_1 = self.mult(gamma_1, delta_1)

        alpha_1 = gamma_1 + delta_1 - 2 * theta_1

        return alpha_1


class SecureDReLUServer(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUServer, self).__init__(crypto_assets, network_assets)

        self.share_convert = ShareConvertServer(crypto_assets, network_assets)
        self.msb = SecureMSBServer(crypto_assets, network_assets)

    def forward(self, X_share):
        assert X_share.dtype == self.dtype
        X1_converted = self.share_convert(self.dtype(2) * X_share)
        MSB_1 = self.msb(X1_converted)
        return 1 - MSB_1


class SecureReLUServer(SecureModule):
    def __init__(self, crypto_assets, network_assets, dummy_relu=False):
        super(SecureReLUServer, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUServer(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        if self.dummy_relu:
            share_client = self.network_assets.receiver_01.get()
            value = share_client + X_share.numpy()
            value = value * ((value > 0).astype(value.dtype))
            return torch.from_numpy(value)
        else:
            shape = X_share.shape
            X_share = X_share.numpy()
            X_share = X_share.astype(self.dtype).flatten()
            MSB_0 = self.DReLU(X_share)
            relu_0 = self.mult(X_share, MSB_0).reshape(shape)
            ret = relu_0.astype(self.signed_type)
            return torch.from_numpy(ret)


class SecureBlockReLUServer(SecureModule):

    def __init__(self, crypto_assets, network_assets, block_sizes):
        super(SecureBlockReLUServer, self).__init__(crypto_assets, network_assets)
        self.block_sizes = np.array(block_sizes)
        self.DReLU = SecureDReLUServer(crypto_assets, network_assets)
        self.mult = SecureMultiplicationServer(crypto_assets, network_assets)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if 0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def forward(self, activation):
        assert False
        activation = activation.numpy()
        assert activation.dtype == self.signed_type

        reshaped_inputs = []
        mean_tensors = []
        channels = []
        orig_shapes = []

        for block_size in self.active_block_sizes:

            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = activation[:, cur_channels]

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
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i+1])].reshape(orig_shapes[i])
            relu_map[:, channels[i]] = DepthToSpace(self.active_block_sizes[i])(sign_tensor.repeat(reshaped_inputs[i].shape[-1], axis=-1))

        activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels], activation[:, ~self.is_identity_channels])
        activation = activation.astype(self.signed_type)

        return torch.from_numpy(activation)


def build_secure_conv(crypto_assets, network_assets, conv_module, bn_module):

    dtype = np.int64

    if bn_module:
        W, B = fuse_conv_bn(conv_module=conv_module, batch_norm_module=bn_module)
        W = TypeConverter.f2i(W)
        B = TypeConverter.f2i(B)

    else:
        W = conv_module.weight
        W = TypeConverter.f2i(W)
        B = None

    W_client = torch.from_numpy(crypto_assets[CLIENT, SERVER].integers(low=np.iinfo(dtype).min // 2,
                                           high=np.iinfo(dtype).max // 2,
                                           size=conv_module.weight.shape,
                                           dtype=dtype))
    W = W - W_client
    return SecureConv2DServer(
        W=W,
        bias=B,
        stride=conv_module.stride,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        groups=conv_module.groups,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )


def build_secure_relu(crypto_assets, network_assets, dummy_relu):
    return SecureReLUServer(crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=dummy_relu)



class SecureModel(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(SecureModel, self).__init__( crypto_assets, network_assets)
        self.model = model

    def forward(self, image_shape):
        dtype = np.int64

        image = torch.from_numpy(self.prf_handler[CLIENT, SERVER].integers(low=np.iinfo(dtype).min // 2,
                                                                        high=np.iinfo(dtype).max // 2,
                                                                               size=image_shape,
                                                                               dtype=dtype))
        out = self.model.decode_head(self.model.backbone(image))
        self.network_assets.sender_01.put(out)



if __name__ == "__main__":

    model = get_model(
        config=Params.SECURE_CONFIG_PATH,
        gpu_id=None,
        checkpoint_path=Params.MODEL_PATH
    )


    crypto_assets, network_assets = get_assets(1, repeat=Params.NUM_IMAGES)

    model = securify_mobilenetv2_model(
        model,
        build_secure_conv=partial(build_secure_conv, crypto_assets=crypto_assets, network_assets=network_assets),
        build_secure_relu=partial(build_secure_relu, crypto_assets=crypto_assets, network_assets=network_assets, dummy_relu=Params.DUMMY_RELU),
        secure_model_class=partial(SecureModel, crypto_assets=crypto_assets, network_assets=network_assets),
        block_relu=partial(SecureBlockReLUServer, crypto_assets=crypto_assets, network_assets=network_assets),
        relu_spec_file=Params.RELU_SPEC_FILE)

    for _ in range(Params.NUM_IMAGES):
        out = model(Params.IMAGE_SHAPE)

    network_assets.done()
