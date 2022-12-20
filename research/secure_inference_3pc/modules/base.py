from research.secure_inference_3pc.const import TRUNC, NUM_BITS
import torch
import numpy as np


class SecureModule(torch.nn.Module):
    def __init__(self, crypto_assets, network_assets):

        super(SecureModule, self).__init__()

        self.prf_handler = crypto_assets
        self.network_assets = network_assets

        self.trunc = TRUNC
        self.torch_dtype = torch.int64
        self.dtype = np.ulonglong

        self.min_val = np.iinfo(self.dtype).min
        self.max_val = np.iinfo(self.dtype).max
        self.L_minus_1 = 2 ** NUM_BITS - 1
        self.signed_type = np.int64

    def add_mode_L_minus_one(self, a, b):
        ret = a + b
        ret[ret < a] += self.dtype(1)
        ret[ret == self.L_minus_1] = self.dtype(0)
        return ret

    def sub_mode_L_minus_one(self, a, b):
        ret = a - b
        ret[b > a] -= self.dtype(1)
        return ret


class PRFFetcherModule(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherModule, self).__init__(crypto_assets, network_assets)


