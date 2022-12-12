import torch
import numpy as np

from research.communication.utils import Sender, Receiver
import time
from numba import njit, prange
num_bit_to_dtype = {
    8: np.ubyte,
    16: np.ushort,
    32: np.uintc,
    64: np.ulonglong
}

num_bit_to_sign_dtype = {
    32: np.int32,
    64: np.int64
}

num_bit_to_torch_dtype = {
    32: torch.int32,
    64: torch.int64
}


NUM_BITS = 64
TRUNC = 10000
dtype = num_bit_to_dtype[NUM_BITS]
powers = np.arange(NUM_BITS, dtype=num_bit_to_dtype[NUM_BITS])[np.newaxis]
moduli = (2 ** powers)
P = 67

def decompose(value):
    orig_shape = list(value.shape)
    value_bits = (value.reshape(-1, 1) >> powers) & 1
    return value_bits.reshape(orig_shape + [NUM_BITS])

private_prf_numpy = np.random.default_rng(seed=31243)
dtype = np.uint64
value_0 = private_prf_numpy.integers(low=np.iinfo(dtype).min, high=np.iinfo(dtype).max, dtype=dtype, size=(10000000,))
value_1 = private_prf_numpy.integers(low=np.iinfo(dtype).min, high=np.iinfo(dtype).max, dtype=dtype, size=(10000000,))

t0 = time.time()

x, y = decompose(np.stack([value_0, value_1]))

print(time.time() - t0)

t0 = time.time()
x = decompose(value_0)
y = decompose(value_1)
print(time.time() - t0)