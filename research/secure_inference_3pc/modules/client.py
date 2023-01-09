from research.secure_inference_3pc.modules.base import PRFFetcherModule, SecureModule

# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors
from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.base import decompose, get_c_party_0, P, module_67, get_assets, TypeConverter, decompose_torch_0, get_c_party_0_torch
from research.secure_inference_3pc.conv2d import conv_2d
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, NUM_OF_COMPARE_BITS

from research.secure_inference_3pc.modules.conv2d import get_output_shape

from research.secure_inference_3pc.conv2d_torch import Conv2DHandler
from research.bReLU import SecureOptimizedBlockReLU



class SecureConv2DClient(SecureModule):

    def __init__(self, W_shape, stride, dilation, padding, groups, device="cpu", **kwargs):
        super(SecureConv2DClient, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.conv2d_handler = Conv2DHandler("cuda:0")
        self.device = device

    def forward_(self, X_share):
        self.W_share = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL,
                                                              high=MAX_VAL,
                                                              size=self.W_shape,
                                                              dtype=SIGNED_DTYPE)
        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert (self.W_share.shape[1] == X_share.shape[1]) or self.groups > 1

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_share.shape, dtype=SIGNED_DTYPE)
        C_share = self.network_assets.receiver_02.get()

        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(self.W_share, B_share, out=B_share)

        share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(backend.concatenate([E_share.flatten(), F_share.flatten()]))
        E_share_server, F_share_server = share_server[:backend.size(E_share)].reshape(E_share.shape), share_server[backend.size(E_share):].reshape(F_share.shape)

        E = backend.add(E_share_server, E_share, out=E_share)
        F = backend.add(F_share_server, F_share, out=F_share)

        if self.device == "cpu":
            out = conv_2d(X_share, F, E, self.W_share, self.padding, self.stride, self.dilation, self.groups)
        else:
            out = self.conv2d_handler.conv2d(X_share, F, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
            out += self.conv2d_handler.conv2d(E, self.W_share, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

        out = backend.add(out, C_share, out=out)
        # out = out // self.trunc
        out = backend.right_shift(out, 16)

        # out = backend.astype((out/self.trunc).round(), dtype=SIGNED_DTYPE)
        # TODO: This is the proper way, but it's slower and takes more time
        #  t = out_numpy.dtype
        #  out = (out_numpy / self.trunc).round().astype(t)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)

        out = backend.add(out, mu_0, out=out)

        return out

    def forward(self, X_share):
        # with Timer("SecureConv2DClient"):
        return self.forward_(X_share)

class PrivateCompareClient(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareClient, self).__init__(**kwargs)

    def forward(self, x_bits_0, r, beta):
        # with Timer("PrivateCompareClient"):
        return self.forward_(x_bits_0, r, beta)

    def forward_(self, x_bits_0, r, beta):
        # TODO: what about this piece of code??!
        # print(r.dtype)
        # if backend.any(r == backend.iinfo(r.dtype).max):  # HERE
        #     assert False
        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=P, size=x_bits_0.shape, dtype=backend.int32)
        r[backend.astype(beta, backend.bool)] += 1
        bits = decompose(r)
        c_bits_0 = get_c_party_0(x_bits_0, bits, beta)
        backend.multiply(s, c_bits_0, out=s)
        d_bits_0 = module_67(s)

        d_bits_0 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_0, axis=-1)

        self.network_assets.sender_02.put(d_bits_0)

        return


class ShareConvertClient(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertClient, self).__init__(**kwargs)
        self.private_compare = PrivateCompareClient(**kwargs)

    def forward(self, a_0):
        return self.forward_(a_0)

    def forward_(self, a_0):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=backend.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=a_0.shape, dtype=SIGNED_DTYPE)
        # alpha = backend.greater_than(r_0 - r, 0)
        alpha = backend.astype(0 < r_0 - r, SIGNED_DTYPE)

        a_tild_0 = a_0 + r_0
        beta_0 = backend.astype(0 < a_0 - a_tild_0, SIGNED_DTYPE)
        self.network_assets.sender_02.put(a_tild_0)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [NUM_OF_COMPARE_BITS], dtype=backend.int8)
        delta_0 = self.network_assets.receiver_02.get()

        self.private_compare(x_bits_0, r - 1, eta_pp)

        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_0.shape, dtype=SIGNED_DTYPE)
        eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)
        t0 = eta_pp * eta_p_0
        t1 = self.add_mode_L_minus_one(t0, t0)
        t2 = self.sub_mode_L_minus_one(eta_pp, t1)
        eta_0 = self.add_mode_L_minus_one(eta_p_0, t2)

        t0 = self.add_mode_L_minus_one(delta_0, eta_0)
        t1 = self.sub_mode_L_minus_one(t0, backend.ones_like(t0))
        t2 = self.sub_mode_L_minus_one(t1, alpha)
        theta_0 = self.add_mode_L_minus_one(beta_0, t2)

        y_0 = self.sub_mode_L_minus_one(a_0, theta_0)
        y_0 = self.add_mode_L_minus_one(y_0, mu_0)

        return y_0


class SecureMultiplicationClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationClient, self).__init__(**kwargs)

    def forward(self, X_share, Y_share):
        return self.forward_(X_share, Y_share)

    def forward_(self, X_share, Y_share):

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
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
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)

        return out + mu_0


class SecureSelectShareClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareClient, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationClient(**kwargs)

    def forward(self, alpha, x, y):
        # if alpha == 0: return x else return 1
        shape = alpha.shape
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        w = y - x
        c = self.secure_multiplication(alpha, w)
        z = x + c
        return z + mu_0


class SecureMSBClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBClient, self).__init__(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.private_compare = PrivateCompareClient(**kwargs)

    def forward(self, a_0):
        return self.forward_(a_0)

    def forward_(self, a_0):

        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=backend.int8)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [NUM_OF_COMPARE_BITS], dtype=backend.int8)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=a_0.dtype)

        x_0 = self.network_assets.receiver_02.get()
        x_bit_0_0 = self.network_assets.receiver_02.get()

        y_0 = self.add_mode_L_minus_one(a_0, a_0)
        r_0 = self.add_mode_L_minus_one(x_0, y_0)
        r_1 = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(r_0)
        r = self.add_mode_L_minus_one(r_0, r_1)

        r_mode_2 = r % 2

        self.private_compare(x_bits_0, r, beta)

        beta = backend.astype(beta, SIGNED_DTYPE)
        beta_p_0 = self.network_assets.receiver_02.get()

        gamma_0 = beta_p_0 + (0 * beta) - (2 * beta * beta_p_0)
        delta_0 = x_bit_0_0 - (2 * r_mode_2 * x_bit_0_0)

        theta_0 = self.mult(gamma_0, delta_0)
        alpha_0 = gamma_0 + delta_0 - 2 * theta_0
        alpha_0 = alpha_0 + mu_0

        return alpha_0


class SecureDReLUClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureDReLUClient, self).__init__(**kwargs)

        self.share_convert = ShareConvertClient(**kwargs)
        self.msb = SecureMSBClient(**kwargs)

    def forward(self, X_share):

        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)

        X0_converted = self.share_convert(X_share)
        MSB_0 = self.msb(X0_converted)

        return -MSB_0+mu_0


class SecureReLUClient(SecureModule):
    def __init__(self, dummy_relu=False, **kwargs, ):
        super(SecureReLUClient, self).__init__(**kwargs)

        self.DReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        return self.forward_(X_share)

    def forward_(self, X_share):
        if self.dummy_relu:
            self.network_assets.sender_01.put(X_share)
            return backend.zeros_like(X_share)
        else:

            shape = X_share.shape
            mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

            X_share = X_share.flatten()
            MSB_0 = self.DReLU(X_share)
            ret = self.mult(X_share, MSB_0).reshape(shape)

            return ret + mu_0


class SecureMaxPoolClient(SecureMaxPool):
    def __init__(self, kernel_size, stride, padding, dummy_max_pool, **kwargs):
        super(SecureMaxPoolClient, self).__init__(kernel_size, stride, padding, dummy_max_pool, **kwargs)
        self.select_share = SecureSelectShareClient(**kwargs)
        self.dReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)

    def forward(self, x):
        if self.dummy_max_pool:
            self.network_assets.sender_01.put(x)
            return backend.zeros_like(x[:, :, ::2, ::2])

        ret = super(SecureMaxPoolClient, self).forward(x)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)

        return ret + mu_0


class SecureBlockReLUClient(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)


class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W_shape, stride, dilation, padding, groups, device="cpu", **kwargs):
        super(PRFFetcherConv2D, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        out_shape = get_output_shape(X_share, self.W_shape, self.padding, self.dilation, self.stride)
        # print(f"PRFFetcherConv2D - {X_share.shape}, {self.W_share.shape}, {out_shape}")
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=X_share.dtype)

        return backend.zeros(shape=out_shape, dtype=X_share.dtype)


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherPrivateCompare, self).__init__(**kwars)

    def forward(self, x_bits_0):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=1, high=P, size=[x_bits_0.shape[0]] + [NUM_OF_COMPARE_BITS], dtype=backend.int32)


class PRFFetcherShareConvert(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherShareConvert, self).__init__(**kwars)
        self.private_compare = PRFFetcherPrivateCompare(**kwars)

    def forward(self, dummy_tensor):
        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=dummy_tensor.shape, dtype=backend.int8)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(dummy_tensor.shape) + [NUM_OF_COMPARE_BITS], dtype=backend.int8)

        self.private_compare(dummy_tensor)

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)

        return dummy_tensor


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherMultiplication, self).__init__(**kwars)


    def forward(self, dummy_tensor):

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)

        return dummy_tensor


class PRFFetcherSelectShare(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherSelectShare, self).__init__(**kwars)
        self.mult = PRFFetcherMultiplication(**kwars)


    def forward(self, dummy_tensor):
        dtype = dummy_tensor.dtype
        shape = dummy_tensor.shape

        self.prf_handler[CLIENT, SERVER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=shape, dtype=dtype)
        self.mult(dummy_tensor)
        return dummy_tensor



class PRFFetcherMSB(PRFFetcherModule):
    def __init__(self, **kwars):
        super(PRFFetcherMSB, self).__init__(**kwars)
        self.mult = PRFFetcherMultiplication(**kwars)
        self.private_compare = PRFFetcherPrivateCompare(**kwars)

    def forward(self, dummy_tensor):

        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=dummy_tensor.shape, dtype=backend.int8)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(dummy_tensor.shape) + [NUM_OF_COMPARE_BITS], dtype=backend.int8)
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
        super(PRFFetcherMaxPool, self).__init__(**kwargs)
        self.dummy_max_pool = dummy_max_pool
        self.select_share = PRFFetcherSelectShare(**kwargs)
        self.dReLU = PRFFetcherDReLU(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)

    def forward(self, x):
        if self.dummy_max_pool:
            return x[:,:,::2,::2]
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


class PRFFetcherBlockReLU(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.secure_DReLU = PRFFetcherDReLU(**kwargs)
        self.secure_mult = PRFFetcherMultiplication(**kwargs)

        self.dummy_relu = dummy_relu

    def mult(self, x, y):
        return self.secure_mult(x)

    def DReLU(self, activation):
        return self.secure_DReLU(activation)

    def forward(self, activation):
        if self.dummy_relu:
            return backend.zeros_like(activation)

        activation = SecureOptimizedBlockReLU.forward(self, activation)
        activation = backend.astype(activation, SIGNED_DTYPE)

        return activation

class PRFFetcherSecureModelSegmentation(SecureModule):
    def __init__(self, model,  **kwargs):
        super(PRFFetcherSecureModelSegmentation, self).__init__( **kwargs)
        self.model = model

    def forward(self, img):

        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, dtype=SIGNED_DTYPE, size=img.shape)
        out_0 = self.model.decode_head(self.model.backbone(backend.zeros(shape=img.shape, dtype=SIGNED_DTYPE)))


class PRFFetcherSecureModelClassification(SecureModule):
    def __init__(self, model, **kwargs):
        super(PRFFetcherSecureModelClassification, self).__init__(**kwargs)
        self.model = model

    def forward(self, img):
        # print(f"PRFFetcherSecureModelClassification - {img.shape}")
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=MIN_VAL, high=MAX_VAL, dtype=SIGNED_DTYPE, size=img.shape)
        out = self.model.backbone(backend.zeros(shape=img.shape, dtype=SIGNED_DTYPE))[0]
        out = self.model.neck(out)
        out_0 = self.model.head.fc(out)