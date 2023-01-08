from research.secure_inference_3pc.modules.base import PRFFetcherModule
from research.secure_inference_3pc.const import  NUM_OF_COMPARE_BITS

import torch
from research.secure_inference_3pc.backend import backend


from research.secure_inference_3pc.base import P, sub_mode_p, decompose
from research.secure_inference_3pc.conv2d import conv_2d
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE
from research.secure_inference_3pc.modules.conv2d import get_output_shape
from research.secure_inference_3pc.conv2d_torch import Conv2DHandler
from research.bReLU import NumpySecureOptimizedBlockReLU
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool


# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors
class SecureConv2DCryptoProvider(SecureModule):
    def __init__(self, W_shape, stride, dilation, padding, groups, device="cpu", **kwargs):
        super(SecureConv2DCryptoProvider, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

        self.conv2d_handler = Conv2DHandler("cuda:0")
        self.device = device

    def forward(self, X_share):

        # TODO: intergers should be called without all of these arguments
        A_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        A_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)

        A = backend.add(A_share_0, A_share_1, out=A_share_0)
        B = backend.add(B_share_0, B_share_1, out=B_share_0)

        if self.device == "cpu":
            C = conv_2d(A, B, None, None, self.padding, self.stride, self.dilation, self.groups)
        else:
            C = self.conv2d_handler.conv2d(A, B, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
        C_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=C.shape, dtype=SIGNED_DTYPE)
        C_share_0 = backend.subtract(C, C_share_1, out=C)

        self.network_assets.sender_02.put(C_share_0)

        return C_share_0


class PrivateCompareCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareCryptoProvider, self).__init__(**kwargs)

    def forward(self):
        d_bits_0 = self.network_assets.receiver_02.get()
        d_bits_1 = self.network_assets.receiver_12.get()

        d = (d_bits_0 + d_bits_1) % P
        beta_p = backend.astype((d == 0).any(axis=-1), SIGNED_DTYPE)

        return beta_p


class ShareConvertCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertCryptoProvider, self).__init__(**kwargs)
        self.private_compare = PrivateCompareCryptoProvider(**kwargs)

    def forward(self, size):
        a_tild_0 = self.network_assets.receiver_02.get()
        a_tild_1 = self.network_assets.receiver_12.get()

        x = (a_tild_0 + a_tild_1)

        x_bits = decompose(x)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=x_bits.shape, dtype=backend.int8)
        x_bits_1 = sub_mode_p(x_bits, x_bits_0)

        delta = backend.astype((0 < a_tild_0 - x), SIGNED_DTYPE)

        delta_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        delta_0 = self.sub_mode_L_minus_one(delta, delta_1)

        self.network_assets.sender_02.put(delta_0)
        self.network_assets.sender_12.put(backend.astype(x_bits_1, backend.int8))

        # r = self.network_assets.receiver_12.get()
        # eta_p = self.network_assets.receiver_12.get()
        # eta_p = eta_p ^ (x > r)

        eta_p = self.private_compare()

        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        eta_p_1 = self.sub_mode_L_minus_one(eta_p, eta_p_0)

        self.network_assets.sender_12.put(eta_p_1)

        return


class SecureMultiplicationCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationCryptoProvider, self).__init__(**kwargs)

    def forward(self, shape):
        A_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        B_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        C_share_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        A_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)
        B_share_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        A = A_share_0 + A_share_1
        B = B_share_0 + B_share_1

        C_share_0 = A * B - C_share_1

        self.network_assets.sender_02.put(C_share_0)


class SecureMSBCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBCryptoProvider, self).__init__(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)
        self.private_compare = PrivateCompareCryptoProvider(**kwargs)

    def forward(self, size):
        x = self.prf_handler[CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)

        x_bits = decompose(x)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=x_bits.shape, dtype=backend.int8)
        x_bits_1 = sub_mode_p(x_bits, x_bits_0)

        x_1 = self.prf_handler[SERVER, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        x_0 = self.sub_mode_L_minus_one(x, x_1)

        x_bit0 = x % 2
        x_bit_0_0 = self.prf_handler[CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=size, dtype=SIGNED_DTYPE)
        x_bit_0_1 = x_bit0 - x_bit_0_0

        self.network_assets.sender_02.put(x_0)
        self.network_assets.sender_02.put(x_bit_0_0)

        self.network_assets.sender_12.put(backend.astype(x_bits_1, backend.int8))
        self.network_assets.sender_12.put(x_bit_0_1)

        # r = self.network_assets.receiver_12.get()
        # beta = self.network_assets.receiver_12.get()
        # beta_p = beta ^ (x > r)
        beta_p = self.private_compare()

        beta_p_0 = self.prf_handler[CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=size, dtype=SIGNED_DTYPE)
        beta_p_1 = beta_p - beta_p_0

        self.network_assets.sender_02.put(beta_p_0)
        self.network_assets.sender_12.put(beta_p_1)

        self.mult(size)
        return


class SecureDReLUCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureDReLUCryptoProvider, self).__init__(**kwargs)

        self.share_convert = ShareConvertCryptoProvider(**kwargs)
        self.msb = SecureMSBCryptoProvider(**kwargs)

    def forward(self, X_share):

        self.share_convert(X_share.shape)
        self.msb(X_share.shape)
        return X_share


class SecureReLUCryptoProvider(SecureModule):
    def __init__(self, dummy_relu=False, **kwargs):
        super(SecureReLUCryptoProvider, self).__init__(**kwargs)

        self.DReLU = SecureDReLUCryptoProvider(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)
        self.dummy_relu = dummy_relu

    def forward(self, X_share):
        if self.dummy_relu:
            return X_share
        else:
            orig_shape = X_share.shape
            X_share = X_share.flatten()
            X_share = self.DReLU(X_share)
            self.mult(X_share.shape)
            return X_share.reshape(orig_shape)

class SecureBlockReLUCryptoProvider(SecureModule, NumpySecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        NumpySecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUCryptoProvider(**kwargs)
        self.secure_mult = SecureMultiplicationCryptoProvider(**kwargs)

    def mult(self, x, y):
        self.secure_mult(x.shape)
        return x



class SecureSelectShareCryptoProvider(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareCryptoProvider, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationCryptoProvider(**kwargs)

    def forward(self, share, dummy0=None, dummy1=None):

        self.secure_multiplication(share.shape)
        return share

class SecureMaxPoolCryptoProvider(SecureMaxPool):
    def __init__(self, kernel_size, stride, padding, dummy_max_pool, **kwargs):
        super(SecureMaxPoolCryptoProvider, self).__init__(kernel_size, stride, padding, dummy_max_pool, **kwargs)
        self.select_share = SecureSelectShareCryptoProvider(**kwargs)
        self.dReLU = SecureDReLUCryptoProvider(**kwargs)
        self.mult = SecureMultiplicationCryptoProvider(**kwargs)

    def forward(self, x):
        if self.dummy_max_pool:
            return x[:, :, ::2, ::2]
        return super(SecureMaxPoolCryptoProvider, self).forward(x)


class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W_shape, stride, dilation, padding, groups, device="cpu", **kwargs):
        super(PRFFetcherConv2D, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):

        out_shape = get_output_shape(X_share, self.W_shape, self.padding, self.dilation, self.stride)

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=out_shape, dtype=SIGNED_DTYPE)

        return backend.zeros(shape=out_shape, dtype=X_share.dtype)


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherPrivateCompare, self).__init__(**kwargs)

    def forward(self, x_bits_0):
        return 


class PRFFetcherShareConvert(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherShareConvert, self).__init__(**kwargs)
        self.private_compare = PRFFetcherPrivateCompare(**kwargs)

    def forward(self, dummy_tensor):
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(dummy_tensor.shape) + [NUM_OF_COMPARE_BITS], dtype=backend.int8)
        # self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)

        self.private_compare(dummy_tensor)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)

        return


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherMultiplication, self).__init__(**kwargs)

    def forward(self, dummy_tensor):

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=dummy_tensor.shape, dtype=SIGNED_DTYPE)
        return dummy_tensor

class PRFFetcherSelectShare(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherSelectShare, self).__init__(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)


    def forward(self, dummy_tensor):

        self.mult(dummy_tensor)
        return dummy_tensor

class PRFFetcherMSB(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherMSB, self).__init__(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)
        self.private_compare = PRFFetcherPrivateCompare(**kwargs)

    def forward(self, dummy_tensor):
        size = dummy_tensor.shape
        
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(size) + [NUM_OF_COMPARE_BITS], dtype=backend.int8)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL, size=size, dtype=SIGNED_DTYPE)
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=size,dtype=SIGNED_DTYPE)
        self.private_compare(dummy_tensor)
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(MIN_VAL, MAX_VAL + 1, size=size, dtype=SIGNED_DTYPE)
        self.mult(dummy_tensor)

        return dummy_tensor


class PRFFetcherDReLU(PRFFetcherModule):
    def __init__(self, **kwargs):
        super(PRFFetcherDReLU, self).__init__(**kwargs)

        self.share_convert = PRFFetcherShareConvert(**kwargs)
        self.msb = PRFFetcherMSB(**kwargs)

    def forward(self, dummy_tensor):

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
            dummy_arr = backend.astype(dummy_tensor, SIGNED_DTYPE).flatten()
            self.DReLU(dummy_arr)
            self.mult(dummy_arr)
            return dummy_tensor


class PRFFetcherMaxPool(PRFFetcherModule):
    def __init__(self, kernel_size=3, stride=2, padding=1, dummy_max_pool=False, **kwargs):
        super(PRFFetcherMaxPool, self).__init__(**kwargs)

        self.select_share = PRFFetcherSelectShare(**kwargs)
        self.dReLU = PRFFetcherDReLU(**kwargs)
        self.mult = PRFFetcherMultiplication(**kwargs)
        self.dummy_max_pool = dummy_max_pool

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
        x = backend.astype(x.reshape((x.shape[0], -1)), SIGNED_DTYPE)

        max_ = x[0]
        for i in range(1, 9):
            self.dReLU(max_)
            self.select_share(max_)

        ret = backend.astype(max_.reshape(out_shape), SIGNED_DTYPE)

        return ret


class PRFFetcherBlockReLU(SecureModule, NumpySecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        NumpySecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.secure_DReLU = PRFFetcherDReLU(**kwargs)
        self.secure_mult = PRFFetcherMultiplication(**kwargs)

        self.dummy_relu = dummy_relu

    def mult(self, x, y):
        return self.secure_mult(backend.astype(x, SIGNED_DTYPE))

    def DReLU(self, activation):
        return self.secure_DReLU(backend.astype(activation, SIGNED_DTYPE))

    def forward(self, activation):
        if self.dummy_relu:
            return torch.zeros_like(activation)

        activation = NumpySecureOptimizedBlockReLU.forward(self, activation)
        activation = backend.astype(activation, SIGNED_DTYPE)

        return activation
class PRFFetcherSecureModelSegmentation(SecureModule):
    def __init__(self, model,  **kwargs):
        super(PRFFetcherSecureModelSegmentation, self).__init__( **kwargs)
        self.model = model

    def forward(self, img):
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=img.shape, dtype=SIGNED_DTYPE)
        out_0 = self.model.decode_head(self.model.backbone(backend.zeros(shape=img.shape, dtype=SIGNED_DTYPE)))


class PRFFetcherSecureModelClassification(SecureModule):
    def __init__(self, model,  **kwargs):
        super(PRFFetcherSecureModelClassification, self).__init__( **kwargs)
        self.model = model

    def forward(self, img):
        self.prf_handler[CRYPTO_PROVIDER].integers_fetch(low=MIN_VAL, high=MAX_VAL, size=img.shape, dtype=SIGNED_DTYPE)
        out = self.model.backbone(backend.zeros(shape=img.shape, dtype=SIGNED_DTYPE))[0]
        out = self.model.neck(out)
        out_0 = self.model.head.fc(out)


