from research.secure_inference_3pc.const import TRUNC, NUM_BITS
import torch
import numpy as np


class SecureModule(torch.nn.Module):
    def __init__(self, crypto_assets, network_assets, is_prf_fetcher=False):
        torch.nn.Module.__init__(self)

        self.is_prf_fetcher = is_prf_fetcher
        self.prf_handler = crypto_assets
        self.network_assets = network_assets

        self.trunc = TRUNC

    def add_mode_L_minus_one(self, a, b):
        ret = a + b
        ret[ret.astype(np.uint64, copy=False) < a.astype(np.uint64, copy=False)] += 1
        ret[ret == - 1] = 0   # If ret were uint64, then the condition would be ret == 2**64 - 1
        return ret

    def sub_mode_L_minus_one(self, a, b):
        ret = a - b
        ret[b.astype(np.uint64, copy=False) > a.astype(np.uint64, copy=False)] -= 1
        return ret

    # def forward(self):
    #     if self.is_prf_fetcher:
    #         self.forward_prf_fetcher()
    #     else:
    #         self.forward()
    #
class PRFFetcherModule(SecureModule):
    def __init__(self, **kwargs):
        super(PRFFetcherModule, self).__init__(**kwargs)


