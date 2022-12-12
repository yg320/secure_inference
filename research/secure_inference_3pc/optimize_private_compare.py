import numpy as np
import time

prf = np.random.default_rng(seed=31243)
dtype = np.uint64

min_val = np.iinfo(dtype).min
max_val = np.iinfo(dtype).max

P = 67

NUM_BITS = 64
powers = np.arange(NUM_BITS, dtype=dtype)[np.newaxis][:,::-1]


def decompose(value):
    orig_shape = list(value.shape)
    value_bits = (value.reshape(-1, 1) >> powers) & 1
    return value_bits.reshape(orig_shape + [NUM_BITS])




def get_c(x_bits, r_bits, t_bits, beta, j):
    x_bits = x_bits.astype(np.int32)
    r_bits = r_bits.astype(np.int32)
    t_bits = t_bits.astype(np.int32)
    beta = beta.astype(np.int32)
    j = j.astype(np.int32)
    beta = beta[..., np.newaxis]

    multiplexer_bits = r_bits * (1 - beta) + t_bits * beta
    w = x_bits + j * multiplexer_bits - 2 * multiplexer_bits * x_bits
    rrr = w.cumsum(axis=-1) - w
    zzz = j + (1 - 2 * beta) * (j * multiplexer_bits - x_bits)
    ret = rrr + zzz

    return ret


r = prf.integers(min_val, max_val, dtype=dtype, size=(1000000,))
x_bits_1 = prf.integers(0, 67, dtype=dtype, size=(1000000, 64))
beta = prf.integers(0, 2, dtype=dtype, size=(1000000,))



t0 = time.time()
s = prf.integers(low=1, high=67, size=x_bits_1.shape, dtype=dtype)
# r_beta_0 = r[beta == 0]
# r_beta_1 = r[beta == 1]
t = r + dtype(1)
party = dtype(1)

r_bits = decompose(r)
t_bits = decompose(t)

c_bits_1 = get_c(x_bits_1, r_bits, t_bits, beta, party)

xxx = (s * c_bits_1).astype(np.int32)

d_bits_1_ = (xxx % P).astype(np.uint8)

d_bits_1 = prf.permutation(d_bits_1_, axis=-1)
print(time.time() - t0)
