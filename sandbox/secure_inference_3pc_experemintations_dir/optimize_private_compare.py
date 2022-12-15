import numpy as np
import time

prf = np.random.default_rng(seed=31243)
dtype = np.uint64

min_val = np.iinfo(dtype).min
max_val = np.iinfo(dtype).max

P = 67

NUM_BITS = 64
powers = np.arange(NUM_BITS, dtype=dtype)[np.newaxis][:,::-1]

min_org_shit = -283206
max_org_shit = 287469
org_shit = (np.arange(min_org_shit, max_org_shit + 1) % P).astype(np.uint8)

def decompose(value, out=None, out_mask=None):
    orig_shape = list(value.shape)
    value = value.reshape(-1, 1)
    r_shift = value >> powers

    value_bits = np.zeros(shape=(value.shape[0], 64), dtype=np.int8)
    np.bitwise_and(r_shift, np.int8(1), out=value_bits)

    # value_bits = np.bitwise_and(r_shift, np.int8(1)).astype(np.int8)


    return value_bits.reshape(orig_shape + [NUM_BITS])


def get_c(x_bits, multiplexer_bits, beta, j):

    t0 = time.time()
    beta = beta[..., np.newaxis]
    t1 = time.time()
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
s = prf.integers(low=1, high=67, size=x_bits_1.shape, dtype=np.int32)

t1 = time.time()
r[beta] += 1
bits = decompose(r)

t2 = time.time()
c_bits_1 = get_c(x_bits_1, bits, beta, np.int8(1))
t3 = time.time()

s = np.multiply(s, c_bits_1, out=s)

t4 = time.time()

aaaa = s.reshape(-1) - min_org_shit
s = org_shit[aaaa].reshape(s.shape)

t5 = time.time()
# d_bits_1_ = (xxx % P).astype(np.uint8)
# prf.permuted(s, axis=-1, out=s)
a = prf.permutation(np.arange(0,64)[np.newaxis].repeat(s.shape[0], axis=0), axis=-1)
b = (64 * np.arange(s.shape[0]))[:,np.newaxis]
c = a + b
d = c.flatten()

t6 = time.time()
d_bits_1 = (s.reshape(-1)[d]).reshape(s.shape)
t7 = time.time()
d_bits_1 = prf.permutation(s, axis=-1)
t8 = time.time()
print("random s", t1 - t0)
print("get bits", t2 - t1)
print("get_c", t3 - t2)
print("mult", t4 - t3)
print("modul", t5 - t4)
print("permut", t6 - t5)
print("permut", t7 - t6)
print("permut", t8 - t7)
print(t6-t0)