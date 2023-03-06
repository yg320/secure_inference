from research.secure_inference_3pc.const import NUM_BITS
import torch
from research.secure_inference_3pc.backend import backend


class SecureModule(torch.nn.Module):
    def __init__(self, crypto_assets, network_assets, device, is_prf_fetcher=False):
        torch.nn.Module.__init__(self)

        self.is_prf_fetcher = is_prf_fetcher
        self.prf_handler = crypto_assets
        self.network_assets = network_assets
        self.device = device

    def add_mode_L_minus_one(self, a, b):
        ret = a + b
        ret[backend.unsigned_gt(a, ret)] += 1
        ret[ret == - 1] = 0  # If ret were uint64, then the condition would be ret == 2**64 - 1
        return ret

    def sub_mode_L_minus_one(self, a, b):
        ret = a - b
        ret[backend.unsigned_gt(b, a)] -= 1
        return ret


class PRFFetcherModule(SecureModule):
    def __init__(self, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = "cpu"
        super(PRFFetcherModule, self).__init__(**kwargs)


class Decompose(SecureModule):
    def __init__(self, num_bits_ignored, dtype, **kwargs):
        super(Decompose, self).__init__(**kwargs)
        self.num_bits_ignored = num_bits_ignored
        self.powers = backend.unsqueeze(backend.arange(NUM_BITS, dtype=dtype), 0)
        end = None if self.num_bits_ignored == 0 else -self.num_bits_ignored
        self.powers = backend.put_on_device(backend.flip(self.powers, axis=-1)[:, :end], self.device)

    def forward(self, value):
        orig_shape = list(value.shape)
        value = value.reshape(-1, 1)
        value_bits = backend.zeros(shape=(value.shape[0], NUM_BITS - self.num_bits_ignored), dtype=backend.int8)
        value_bits = backend.right_shift(value, self.powers, out=value_bits)
        value_bits = backend.bitwise_and(value_bits, 1, out=value_bits)
        ret = value_bits.reshape(orig_shape + [NUM_BITS - self.num_bits_ignored])
        return ret


class DummyShapeTensor(tuple):
    def __add__(self, other):
        return self
