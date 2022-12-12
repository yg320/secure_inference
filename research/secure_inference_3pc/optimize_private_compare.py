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
    t0 = time.time()
    value = value.reshape(-1, 1)
    print(time.time() - t0)
    value_bits = ((value >> powers) & 1).astype(np.int8)
    t0 = time.time()

    ret = value_bits.reshape(orig_shape + [NUM_BITS])
    print(time.time() - t0)

    return ret


def get_c(x_bits, r_bits, t_bits, beta, j):

    t0 = time.time()
    beta = beta[..., np.newaxis]
    t1 = time.time()
    # np.multiply(t_bits, beta, out=r_bits)
    # np.multiply(r_bits, (1-beta), out=r_bits)
    # np.add(r_bits, t_bits, out=r_bits)
    # multiplexer_bits = r_bits
    multiplexer_bits = r_bits * (1 - beta) + t_bits * beta
    t2 = time.time()

    w = x_bits + j * multiplexer_bits - 2 * multiplexer_bits * x_bits
    t3 = time.time()
    w_cumsum = w.astype(np.int32)
    t4 = time.time()
    np.cumsum(w_cumsum, axis=-1, out=w_cumsum)
    np.subtract(w_cumsum, w, out=w_cumsum)
    rrr = w_cumsum
    t5 = time.time()
    zzz = j + (1 - 2 * beta) * (j * multiplexer_bits - x_bits)
    t6 = time.time()
    ret = rrr + zzz.astype(np.int32)
    t7 = time.time()
    # print(t1-t0)
    # print(t2-t1)
    # print(t3-t2)
    # print(t4-t3)
    # print(t5-t4)
    # print(t6-t5)
    # print(t7-t6)
    return ret


r = prf.integers(min_val, max_val, dtype=dtype, size=(1000000,))
x_bits_1 = prf.integers(0, 67, dtype=np.int8, size=(1000000, 64))
beta = prf.integers(0, 2, dtype=np.int8, size=(1000000,))



t0 = time.time()
s = prf.integers(low=1, high=67, size=x_bits_1.shape, dtype=dtype)
t = r + np.int8(1)
party = np.int8(1)

r_bits = decompose(r)
# r_bits[beta == 1] = decompose(r[beta == 1]+1)
t_bits = decompose(t)

c_bits_1 = get_c(x_bits_1, r_bits, r_bits, beta, party)

xxx = (s * c_bits_1).astype(np.int32)

d_bits_1_ = (xxx % P).astype(np.uint8)

d_bits_1 = prf.permutation(d_bits_1_, axis=-1)
print(time.time() - t0)
