import numpy as np
import torch

from research.secure_inference_3pc.modules.base import PRFFetcherModule, SecureModule
from research.secure_inference_3pc.modules.conv2d import get_output_shape, conv_2d
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, P
from research.secure_inference_3pc.timer import Timer
from research.secure_inference_3pc.base import decompose, get_c, module_67, TypeConverter

# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors
class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W, bias, stride, dilation, padding, groups, crypto_assets, network_assets):
        super(PRFFetcherConv2D, self).__init__(crypto_assets, network_assets)

        self.W_share = W
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):

        X_share = X_share.numpy()
        out_shape = get_output_shape(X_share, self.W_share, self.padding, self.dilation, self.stride)

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=X_share.shape, dtype=np.int64)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=self.W_share.shape, dtype=np.int64)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=out_shape, dtype=np.int64)
        self.prf_handler[CLIENT, SERVER].integers_fetch(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=out_shape, dtype=X_share.dtype)

        return torch.from_numpy(np.zeros(shape=out_shape, dtype=X_share.dtype))


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherPrivateCompare, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_0):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=1, high=67, size=[x_bits_0.shape[0]] + [64], dtype=np.int32)


class PRFFetcherShareConvert(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherShareConvert, self).__init__(crypto_assets, network_assets)
        self.private_compare = PRFFetcherPrivateCompare(crypto_assets, network_assets)

    def forward(self, dummy_tensor):
        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=dummy_tensor.shape, dtype=np.int8)
        self.prf_handler[CLIENT, SERVER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[CLIENT, SERVER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[CLIENT, SERVER].integers_fetch(self.min_val, self.max_val, size=dummy_tensor.shape, dtype=self.dtype)

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val, size=dummy_tensor.shape, dtype=self.dtype)

        self.private_compare(dummy_tensor)

        return dummy_tensor


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherMultiplication, self).__init__(crypto_assets, network_assets)

    def forward(self, dummy_tensor):

        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[CLIENT, SERVER].integers_fetch(np.iinfo(dummy_tensor.dtype).min, np.iinfo(dummy_tensor.dtype).max, size=dummy_tensor.shape, dtype=dummy_tensor.dtype)

        return dummy_tensor


class PRFFetcherMSB(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherMSB, self).__init__(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)
        self.private_compare = PRFFetcherPrivateCompare(crypto_assets, network_assets)

    def forward(self, dummy_tensor):

        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=dummy_tensor.shape, dtype=np.int8)
        self.prf_handler[SERVER, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[CLIENT, SERVER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=dummy_tensor.dtype)

        self.private_compare(dummy_tensor)
        self.mult(dummy_tensor)

        return dummy_tensor


class PRFFetcherDReLU(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherDReLU, self).__init__(crypto_assets, network_assets)

        self.share_convert = PRFFetcherShareConvert(crypto_assets, network_assets)
        self.msb = PRFFetcherMSB(crypto_assets, network_assets)

    def forward(self, dummy_tensor):
        self.prf_handler[CLIENT, SERVER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=dummy_tensor.dtype)

        self.share_convert(dummy_tensor)
        self.msb(dummy_tensor)

        return dummy_tensor


class PRFFetcherReLU(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets, dummy_relu=False):
        super(PRFFetcherReLU, self).__init__(crypto_assets, network_assets)

        self.DReLU = PRFFetcherDReLU(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)
        self.dummy_relu = dummy_relu

    def forward(self, dummy_tensor):
        if self.dummy_relu:
            return dummy_tensor
        else:

            dummy_numpy = dummy_tensor.numpy()
            dtype = dummy_numpy.dtype
            shape = dummy_numpy.shape
            self.prf_handler[CLIENT, SERVER].integers_fetch(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)

            dummy_arr = dummy_numpy.astype(self.dtype).flatten()
            self.DReLU(dummy_arr)
            self.mult(dummy_arr)
            return dummy_tensor



class PRFFetcherSecureModel(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(PRFFetcherSecureModel, self).__init__( crypto_assets, network_assets)
        self.model = model

    def forward(self, img):

        dtype = np.int64
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=np.iinfo(dtype).min // 2, high=np.iinfo(dtype).max // 2, size=img.shape, dtype=dtype)
        out_0 = self.model.decode_head(self.model.backbone(torch.zeros(size=img.shape, dtype=torch.int64)))


