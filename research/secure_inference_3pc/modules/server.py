from research.secure_inference_3pc.backend import backend

import torch


from research.secure_inference_3pc.modules.base import PRFFetcherModule
from research.secure_inference_3pc.modules.conv2d import get_output_shape
from research.secure_inference_3pc.const import NUM_OF_COMPARE_BITS

from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.base import decompose, get_c_party_1, module_67, NetworkAssets, get_c_party_1_torch, decompose_torch_1
from research.secure_inference_3pc.conv2d import conv_2d
from research.secure_inference_3pc.conv2d_torch import Conv2DHandler
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE
from research.bReLU import NumpySecureOptimizedBlockReLU


class SecureConv2DServer(SecureModule):
    def __init__(self, W, bias, stride, dilation, padding, groups, device="cpu", **kwargs):
        super(SecureConv2DServer, self).__init__(**kwargs)

        self.W_plaintext = W
        self.bias = bias
        if self.bias is not None:
            self.bias = backend.reshape(self.bias, [1, -1, 1, 1])
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.conv2d_handler = Conv2DHandler("cuda:1")
        self.device = device

    def forward(self, X_share):

        W_client = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, size=self.W_plaintext.shape, dtype=SIGNED_DTYPE)

        self.W_share = self.W_plaintext - W_client

        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert (self.W_share.shape[1] == X_share.shape[1]) or self.groups > 1
        assert self.stride[0] == self.stride[1]

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_share.shape, dtype=SIGNED_DTYPE)

        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(self.W_share, B_share, out=B_share)

        self.network_assets.sender_01.put(backend.concatenate([E_share.reshape(-1), F_share.reshape(-1)]))
        share_client = self.network_assets.receiver_01.get()

        E_share_client, F_share_client = share_client[:backend.size(E_share)].reshape(E_share.shape), \
                share_client[backend.size(E_share):].reshape(F_share.shape)

        E = backend.add(E_share_client, E_share, out=E_share)
        F = backend.add(F_share_client, F_share, out=F_share)

        self.W_share = backend.subtract(self.W_share, F, out=self.W_share)

        if self.device == "cpu":
            out = conv_2d(E, self.W_share, X_share, F, self.padding, self.stride, self.dilation, self.groups)
        else:
            out = self.conv2d_handler.conv2d(E, self.W_share, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
            out += self.conv2d_handler.conv2d(X_share, F, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)

        out = backend.add(out, C_share, out=out)
        out = out // self.trunc  # TODO:
        # This is the proper way, but it's slower and takes more time
        # t = out.dtype
        # out = (out / self.trunc).round().astype(t)
        if self.bias is not None:
            out = backend.add(out, self.bias, out=out)

        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        return out + mu_1


class PrivateCompareServer(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareServer, self).__init__(**kwargs)

    def forward(self, x_bits_1, r, beta):

        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_1.shape, dtype=backend.int32)

        r[backend.astype(beta, backend.bool)] += 1

        bits = decompose(r)

        c_bits_1 = get_c_party_1(x_bits_1, bits, beta)

        s = backend.multiply(s, c_bits_1, out=s)

        d_bits_1 = module_67(s)

        d_bits_1 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_1, axis=-1)

        self.network_assets.sender_12.put(d_bits_1)


class ShareConvertServer(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertServer, self).__init__(**kwargs)
        self.private_compare = PrivateCompareServer(**kwargs)

    def forward(self, a_1):

        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=backend.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=SIGNED_DTYPE)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=SIGNED_DTYPE)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)

        r_1 = backend.subtract(r, r_0, out=r_0)
        a_tild_1 = backend.add(a_1, r_1, out=r_1)
        beta_1 = backend.astype(0 < a_1 - a_tild_1, SIGNED_DTYPE)  # TODO: Optimize this

        self.network_assets.sender_12.put(a_tild_1)

        x_bits_1 = self.network_assets.receiver_12.get()

        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)

        r_minus_1 = backend.subtract(r, 1, out=r)
        self.private_compare(x_bits_1, r_minus_1, eta_pp)
        eta_p_1 = self.network_assets.receiver_12.get()

        eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)  # TODO: Optimize this
        t00 = backend.multiply(eta_pp, eta_p_1, out=eta_pp)
        t11 = self.add_mode_L_minus_one(t00, t00)  # TODO: Optimize this
        eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)  # TODO: Optimize this
        t00 = self.add_mode_L_minus_one(delta_1, eta_1)  # TODO: Optimize this
        theta_1 = self.add_mode_L_minus_one(beta_1, t00)  # TODO: Optimize this
        y_1 = self.sub_mode_L_minus_one(a_1, theta_1)  # TODO: Optimize this
        y_1 = self.add_mode_L_minus_one(y_1, mu_1)  # TODO: Optimize this
        return y_1


class SecureMultiplicationServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationServer, self).__init__(**kwargs)

    def forward(self, X_share, Y_share):

        A_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        C_share = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(Y_share, B_share, out=B_share)

        self.network_assets.sender_01.put(E_share)
        E_share_client = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)
        F_share_client = self.network_assets.receiver_01.get()

        E = backend.add(E_share_client, E_share, out=E_share)
        F = backend.add(F_share_client, F_share, out=F_share)

        out = - E * F + X_share * F + Y_share * E + C_share  # TODO: Optimize this

        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=X_share.dtype)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)

        return out + mu_1


class SecureMSBServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBServer, self).__init__(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)
        self.private_compare = PrivateCompareServer(**kwargs)

    def forward(self, a_1):

        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_1.shape, dtype=backend.int8)
        x_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_1.shape, dtype=SIGNED_DTYPE)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_1.shape, dtype=a_1.dtype)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)

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

        beta = backend.astype(beta, SIGNED_DTYPE)  # TODO: Optimize this
        gamma_1 = beta_p_1 + beta - 2 * beta * beta_p_1  # TODO: Optimize this
        delta_1 = x_bit_0_1 + r_mod_2 - (2 * r_mod_2 * x_bit_0_1)  # TODO: Optimize this

        theta_1 = self.mult(gamma_1, delta_1)

        alpha_1 = gamma_1 + delta_1 - 2 * theta_1  # TODO: Optimize this
        alpha_1 = alpha_1 + mu_1  # TODO: Optimize this

        return alpha_1


class SecureDReLUServer(SecureModule):
    def __init__(self,  **kwargs):
        super(SecureDReLUServer, self).__init__( **kwargs)

        self.share_convert = ShareConvertServer( **kwargs)
        self.msb = SecureMSBServer(**kwargs)

    def forward(self, X_share):
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        backend.multiply(mu_1, -1, out=mu_1)

        X1_converted = self.share_convert(X_share)
        MSB_1 = self.msb(X1_converted)

        ret = backend.multiply(MSB_1, -1, out=MSB_1)
        ret = backend.add(ret, mu_1, out=ret)
        ret = backend.add(ret, 1, out=ret)
        return ret


class SecureReLUServer(SecureModule):
    def __init__(self, dummy_relu=False,  **kwargs):
        super(SecureReLUServer, self).__init__( **kwargs)

        self.DReLU = SecureDReLUServer( **kwargs)
        self.mult = SecureMultiplicationServer( **kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        if self.dummy_relu:
            assert False
            share_client = self.network_assets.receiver_01.get()
            value = share_client + X_share
            value = value * (backend.astype(value > 0, value.dtype))
            return value
        else:

            shape = X_share.shape
            mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
            backend.multiply(mu_1, -1, out=mu_1)

            X_share = X_share.reshape(-1)
            MSB_0 = self.DReLU(X_share)
            ret = self.mult(X_share, MSB_0).reshape(shape)
            backend.add(ret, mu_1, out=ret)
            return ret


class SecureBlockReLUServer(SecureModule, NumpySecureOptimizedBlockReLU):
    def __init__(self, block_sizes,  dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        NumpySecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUServer(**kwargs)
        self.mult = SecureMultiplicationServer(**kwargs)


class SecureSelectShareServer(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareServer, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationServer(**kwargs)

    def forward(self, alpha, x, y):
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=alpha.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        y = backend.subtract(y, x, out=y)

        c = self.secure_multiplication(alpha, y)
        x = backend.add(x, c, out=x)
        x = backend.add(x, mu_1, out=x)
        return x


class SecureMaxPoolServer(SecureMaxPool):
    def __init__(self, kernel_size, stride, padding, dummy_max_pool,  **kwargs):
        super(SecureMaxPoolServer, self).__init__(kernel_size, stride, padding, dummy_max_pool,  **kwargs)
        self.select_share = SecureSelectShareServer( **kwargs)
        self.dReLU = SecureDReLUServer( **kwargs)
        self.mult = SecureMultiplicationServer( **kwargs)

    def forward(self, x):
        if self.dummy_max_pool:
            x_client = self.network_assets.receiver_01.get()
            x_rec = x_client + x
            return torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)(torch.from_numpy(x_rec).to(torch.float64)).numpy().astype(x.dtype)

        ret = super(SecureMaxPoolServer, self).forward(x)
        mu_1 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)
        mu_1 = backend.multiply(mu_1, -1, out=mu_1)
        ret = backend.add(ret, mu_1, out=ret)
        return ret



# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors
class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W, bias, stride, dilation, padding, groups, device="cpu", **kwargs):
        super(PRFFetcherConv2D, self).__init__(**kwargs)

        self.W_shape = W.shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)

        out_shape = get_output_shape(X_share, self.W_shape, self.padding, self.dilation, self.stride)

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=X_share.dtype)

        return backend.zeros(shape=out_shape, dtype=X_share.dtype)


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherPrivateCompare, self).__init__(**kwargs)

    def forward(self, x_bits_0):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=1, high=67, size=[x_bits_0.shape[0]] + [NUM_OF_COMPARE_BITS], dtype=backend.int32)


class PRFFetcherShareConvert(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherShareConvert, self).__init__(**kwargs)
        self.private_compare = PRFFetcherPrivateCompare(**kwargs)

    def forward(self, dummy_tensor):
        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=dummy_tensor.shape, dtype=backend.int8)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)

        self.private_compare(dummy_tensor)

        return dummy_tensor


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherMultiplication, self).__init__(**kwargs)

    def forward(self, dummy_tensor):

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=dummy_tensor.dtype)

        return dummy_tensor

class PRFFetcherSelectShare(PRFFetcherModule):
    def __init__(self,  **kwargs):
        super(PRFFetcherSelectShare, self).__init__( **kwargs)
        self.mult = PRFFetcherMultiplication( **kwargs)


    def forward(self, dummy_tensor):
        dtype = dummy_tensor.dtype
        shape = dummy_tensor.shape

        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=dtype)
        self.mult(dummy_tensor)
        return dummy_tensor


class PRFFetcherMSB(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherMSB, self).__init__(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)
        self.private_compare = PRFFetcherPrivateCompare(**kwargs)

    def forward(self, dummy_tensor):

        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=dummy_tensor.shape, dtype=backend.int8)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=dummy_tensor.dtype)

        self.private_compare(dummy_tensor)
        self.mult(dummy_tensor)

        return dummy_tensor


class PRFFetcherDReLU(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherDReLU, self).__init__(**kwargs)

        self.share_convert = PRFFetcherShareConvert(**kwargs)
        self.msb = PRFFetcherMSB(**kwargs)

    def forward(self, dummy_tensor):
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=dummy_tensor.dtype)

        self.share_convert(dummy_tensor)
        self.msb(dummy_tensor)

        return dummy_tensor


class PRFFetcherReLU(PRFFetcherModule):
    def __init__(self, dummy_relu=False, **kwargs):
        super(PRFFetcherReLU, self).__init__(**kwargs)
        self.DReLU = PRFFetcherDReLU(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, dummy_tensor):
        if self.dummy_relu:
            return dummy_tensor
        else:

            dummy_numpy = dummy_tensor
            dtype = dummy_numpy.dtype
            shape = dummy_numpy.shape
            self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=dtype)

            dummy_arr = dummy_numpy.flatten()
            self.DReLU(dummy_arr)
            self.mult(dummy_arr)
            return dummy_tensor

class PRFFetcherMaxPool(PRFFetcherModule):
    def __init__(self, kernel_size=3, stride=2, padding=1, dummy_max_pool=False, **kwargs):
        super(PRFFetcherMaxPool, self).__init__( **kwargs)

        self.dummy_max_pool = dummy_max_pool
        self.select_share = PRFFetcherSelectShare( **kwargs)
        self.dReLU = PRFFetcherDReLU( **kwargs)
        self.mult = PRFFetcherMultiplication( **kwargs)

    def forward(self, x):
        if self.dummy_max_pool:
            return x[:,:,::2, ::2]
        assert x.shape[2] == 112
        assert x.shape[3] == 112

        x = backend.pad(x, ((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant')
        x = backend.stack([x[:, :, 0:-1:2, 0:-1:2],
                      x[:, :, 0:-1:2, 1:-1:2],
                      x[:, :, 0:-1:2, 2::2],
                      x[:, :, 1:-1:2, 0:-1:2],
                      x[:, :, 1:-1:2, 1:-1:2],
                      x[:, :, 1:-1:2, 2::2],
                      x[:, :, 2::2, 0:-1:2],
                      x[:, :, 2::2, 1:-1:2],
                      x[:, :, 2::2, 2::2]])

        out_shape = x.shape[1:]
        x = x.reshape((x.shape[0], -1))

        max_ = x[0]
        for i in range(1, 9):
            self.dReLU(max_)
            self.select_share(max_)

        ret = backend.astype(max_.reshape(out_shape), SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)

        return ret

class PRFFetcherBlockReLU(SecureModule, NumpySecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False,  **kwargs):
        SecureModule.__init__(self,  **kwargs)
        NumpySecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.secure_DReLU = PRFFetcherDReLU( **kwargs)
        self.secure_mult = PRFFetcherMultiplication( **kwargs)

        self.dummy_relu = dummy_relu

    def mult(self, x, y):
        return self.secure_mult(x)

    def DReLU(self, activation):
        return self.secure_DReLU(activation)

    def forward(self, activation):
        if self.dummy_relu:
            return torch.zeros_like(activation)

        activation = NumpySecureOptimizedBlockReLU.forward(self, activation)
        activation = backend.astype(activation, SIGNED_DTYPE)

        return activation

class PRFFetcherSecureModelSegmentation(SecureModule):
    def __init__(self, model,   **kwargs):
        super(PRFFetcherSecureModelSegmentation, self).__init__( **kwargs)
        self.model = model

    def forward(self, img):

        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=img.shape, dtype=SIGNED_DTYPE)
        out_0 = self.model.decode_head(self.model.backbone(backend.zeros(shape=img.shape, dtype=SIGNED_DTYPE)))


class PRFFetcherSecureModelClassification(SecureModule):
    def __init__(self, model,   **kwargs):
        super(PRFFetcherSecureModelClassification, self).__init__(  **kwargs)
        self.model = model

    def forward(self, img):

        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, dtype=SIGNED_DTYPE, size=img.shape)
        out = self.model.backbone(backend.zeros(shape=img.shape, dtype=SIGNED_DTYPE))[0]
        out = self.model.neck(out)
        out_0 = self.model.head.fc(out)