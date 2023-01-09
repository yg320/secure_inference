from research.secure_inference_3pc.const import TRUNC, NUM_BITS
import torch
import numpy as np
from research.secure_inference_3pc.backend import backend


class SecureModule(torch.nn.Module):
    def __init__(self, crypto_assets, network_assets, device, is_prf_fetcher=False):
        torch.nn.Module.__init__(self)

        self.is_prf_fetcher = is_prf_fetcher
        self.prf_handler = crypto_assets
        self.network_assets = network_assets
        self.device = device
        self.trunc = TRUNC

    def add_mode_L_minus_one(self, a, b):
        ret = a + b
        ret[backend.unsigned_gt(a, ret)] += 1
        ret[ret == - 1] = 0   # If ret were uint64, then the condition would be ret == 2**64 - 1
        return ret

    def sub_mode_L_minus_one(self, a, b):
        ret = a - b
        ret[backend.unsigned_gt(b, a)] -= 1
        return ret

    # def forward(self):
    #     if self.is_prf_fetcher:
    #         self.forward_prf_fetcher()
    #     else:
    #         self.forward()
    #
class PRFFetcherModule(SecureModule):
    def __init__(self, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = "cpu"
        super(PRFFetcherModule, self).__init__(**kwargs)


class Decompose(SecureModule):
    def __init__(self, ignore_msb_bits, num_of_compare_bits, dtype, **kwargs):
        super(Decompose, self).__init__(**kwargs)
        self.ignore_msb_bits = ignore_msb_bits
        self.num_of_compare_bits = num_of_compare_bits
        self.powers = backend.unsqueeze(backend.arange(NUM_BITS, dtype=dtype), 0)
        end = None if self.ignore_msb_bits == 0 else -self.ignore_msb_bits
        self.powers = backend.put_on_device(backend.flip(self.powers, axis=-1)[:, NUM_BITS - self.num_of_compare_bits - self.ignore_msb_bits:end], self.device)

    def forward(self, value):
        orig_shape = list(value.shape)
        value = value.reshape(-1, 1)
        r_shift = value >> self.powers
        value_bits = backend.zeros(shape=(value.shape[0], self.num_of_compare_bits), dtype=backend.int8)
        value_bits = backend.bitwise_and(r_shift, 1, out=value_bits)  # TODO: backend.int8(1) instead of 1
        ret = value_bits.reshape(orig_shape + [self.num_of_compare_bits])
        return ret