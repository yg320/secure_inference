import numpy as np
import time
import torch
from research.secure_inference_3pc.timer import Timer



IGNORE_MSB_BITS = 0
NUM_BITS = 64
NUM_OF_COMPARE_BITS = 64
UNSIGNED_DTYPE = np.uint64

powers = np.arange(NUM_BITS, dtype=UNSIGNED_DTYPE)[np.newaxis][:,::-1]

def decompose(value, out=None, out_mask=None):
    orig_shape = list(value.shape)
    value = value.reshape(-1, 1)
    end = None if IGNORE_MSB_BITS == 0 else -IGNORE_MSB_BITS
    r_shift = value >> powers[:,NUM_BITS - NUM_OF_COMPARE_BITS-IGNORE_MSB_BITS:end]
    value_bits = np.zeros(shape=(value.shape[0], NUM_OF_COMPARE_BITS), dtype=np.int8)
    np.bitwise_and(r_shift, np.int8(1), out=value_bits)
    ret = value_bits.reshape(orig_shape + [NUM_OF_COMPARE_BITS])
    return ret

def get_c_party_0(x_bits, multiplexer_bits, beta, j):
    beta = beta[..., np.newaxis]
    beta = 2 * beta  # Not allowed to change beta inplace
    np.subtract(beta, 1, out=beta)
    np.multiply(multiplexer_bits, x_bits, out=multiplexer_bits)
    np.multiply(multiplexer_bits, -2, out=multiplexer_bits)
    np.add(multiplexer_bits, x_bits, out=multiplexer_bits)

    w_cumsum = multiplexer_bits.astype(np.int32)
    np.cumsum(w_cumsum, axis=-1, out=w_cumsum)
    np.subtract(w_cumsum, multiplexer_bits, out=w_cumsum)
    np.multiply(x_bits, beta, out=x_bits)
    np.add(w_cumsum, x_bits, out=w_cumsum)

    return w_cumsum

N = 10000
x_bits_0 = np.random.randint(low=0, high=67, shape=(N, NUM_BITS), dtype=np.int8)
r = np.random.randint(low=np.iinfo(np.uint64).min, high=np.iinfo(np.uint64).max, size=(N,), dtype=np.uint64)
beta = np.random.randint(low=0, high=2, size=(N,), dtype=np.int8)


bits = decompose(r)
c_bits_0 = get_c_party_0(x_bits_0, bits, beta, np.int8(0))







# NUM_BITS = 64
# UNSIGNED_DTYPE = np.uint64
# NUM_OF_COMPARE_BITS = 64
#
# power_numpy = np.arange(NUM_BITS, dtype=UNSIGNED_DTYPE)[np.newaxis][:,::-1]
# powers_torch = torch.from_numpy(power_numpy.astype(np.int64)).to("cuda:0")
#
#
# def decompose(value):
#     orig_shape = list(value.shape)
#     value = value.reshape(-1, 1)
#     r_shift = value >> power_numpy[:,NUM_BITS - NUM_OF_COMPARE_BITS:]
#     value_bits = np.zeros(shape=(value.shape[0], NUM_OF_COMPARE_BITS), dtype=np.int8)
#     np.bitwise_and(r_shift, np.int8(1), out=value_bits)
#     ret =  value_bits.reshape(orig_shape + [NUM_OF_COMPARE_BITS])
#     return ret
#
# def decompose_torch(value):
#     orig_shape = list(value.shape)
#     value = value.reshape(-1, 1)
#
#     r_shift = value >> powers_torch[:, NUM_BITS - NUM_OF_COMPARE_BITS:]
#     value_bits = r_shift & 1
#
#     ret =  value_bits.reshape(orig_shape + [NUM_OF_COMPARE_BITS])
#     return ret
#
# value = np.random.randint(low=np.iinfo(np.uint64).min, high=np.iinfo(np.uint64).max, size=(1000000,), dtype=np.uint64)
#
# with Timer("decompose - torch"):
#     torch_tensor = torch.from_numpy(value.astype(np.int64)).to("cuda:0")
#     out = decompose_torch(torch_tensor)
#     out = out.to("cpu").numpy().astype(np.uint64)
#
# with Timer("decompose - numpy"):
#     decompose(value)