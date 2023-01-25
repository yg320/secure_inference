# import numpy as np
# import time
# powers = np.array([[27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8]], dtype=np.int64)
# P = 67
#
# def get_c_party_1(x_bits, multiplexer_bits, beta):
#     beta = beta[..., np.newaxis]
#     beta = -2 * beta  # Not allowed to change beta inplace
#     np.add(beta, 1, out=beta)
#
#     w = multiplexer_bits * x_bits
#     np.multiply(w, -2, out=w)
#     np.add(w, x_bits, out=w)
#     np.add(w, multiplexer_bits, out=w)
#
#     w_cumsum = w.astype(np.int32)
#     np.cumsum(w_cumsum, axis=-1, out=w_cumsum)
#     np.subtract(w_cumsum, w, out=w_cumsum)
#
#     np.subtract(multiplexer_bits, x_bits, out=multiplexer_bits)
#     np.multiply(multiplexer_bits, beta, out=multiplexer_bits)
#     np.add(multiplexer_bits, 1, out=multiplexer_bits)
#     np.add(w_cumsum, multiplexer_bits, out=w_cumsum)
#
#     return w_cumsum
#
# def decompose(value):
#
#     orig_shape = list(value.shape)
#     value = value.reshape(-1, 1)
#     value_bits = np.zeros(shape=(value.shape[0], 20), dtype=np.int8)
#     value_bits = np.right_shift(value, powers, out=value_bits)
#     value_bits = np.bitwise_and(value_bits, 1, out=value_bits)
#     ret = value_bits.reshape(orig_shape + [20])
#     return ret
#
# min_org_shit = -283206
# max_org_shit = 287469
# org_shit = (np.arange(min_org_shit, max_org_shit + 1) % P).astype(np.int8)
#
# def module_67(xxx):
#
#     orig_shape = xxx.shape
#     xxx = xxx.reshape(-1)
#     np.subtract(xxx, min_org_shit, out=xxx)
#     return org_shit[xxx.astype(np.int64)].reshape(orig_shape)
#
#
# np.random.seed(123)
# x_bits_1 = np.random.randint(low=0, high=67, dtype=np.int8, size=(100000, 20))
# r = np.random.randint(low=-9223129350866284269, high=9221000347554954030, dtype=np.int64, size=(100000,))
# beta = np.random.randint(low=0, high=2, dtype=np.int8, size=(100000,))
# s = np.random.randint(low=0, high=67, dtype=np.int32, size=(100000, 20))
#
#
# from numba import njit, prange
# @njit('(int32[:,:])(int32[:,:], int64[:], int8[:,:], int8[:],  uint8)', parallel=True,  nogil=True, cache=True)
# def decompose_numba(s, r, x_bits_1, beta, bits):
#
#     for i in prange(x_bits_1.shape[0]):
#         r[i] = r[i] + beta[i]
#         counter = 0
#         for j in range(20):
#             multiplexer_bit = (r[i] >> (bits - 1 - j)) & 1
#
#             w = -2 * multiplexer_bit * x_bits_1[i, j] + x_bits_1[i, j] + multiplexer_bit
#
#             counter = counter + w
#             w_cumsum = counter - w
#
#             multiplexer_bit = multiplexer_bit - x_bits_1[i, j]
#             multiplexer_bit = multiplexer_bit * (-2 * beta[i] + 1)
#             multiplexer_bit = multiplexer_bit + 1
#             w_cumsum = w_cumsum + multiplexer_bit
#
#             s[i, j] = (s[i, j] * w_cumsum) % 67
#     return s
#
# t0 = time.time()
#
# decompose_numba(s, r, x_bits_1, beta, 28)
#
# d_bits_1 = s
# t1 = time.time()
# print(t1 - t0)
# print(d_bits_1.mean())
#
import numpy as np
P = 67
np.random.seed(123)
d_bits_0 = np.random.randint(low=0, high=67, dtype=np.int32, size=(100000, 20))
d_bits_1 = np.random.randint(low=0, high=67, dtype=np.int32, size=(100000, 20))

from numba import njit, prange
@njit('(int64[:])(int32[:,:], int32[:,:])', parallel=True,  nogil=True, cache=True)
def private(d_bits_0, d_bits_1):
    out = np.zeros(shape=(100000,), dtype=np.int64)
    for i in prange(d_bits_0.shape[0]):
        r = 0
        for j in range(d_bits_0.shape[1]):
            a = (d_bits_0[i, j] + d_bits_1[i, j])
            if a == 0 or a == 67:
                r = 1
                break

        out[i] = r
    return out
import time
t0 = time.time()
# out = np.zeros(shape=(100000, ), dtype=np.int64)
out = private(d_bits_0, d_bits_1)
t1 = time.time()
d = np.add(d_bits_0, d_bits_1, out=d_bits_0)
d = d % P
beta_p = (d == 0).any(axis=-1).astype(np.int64)
t2 = time.time()

print((out == beta_p).all())
print(t1 - t0)
print(t2 - t1)

# x_bits = self.decompose(x)
# x_bits_1 = backend.subtract_module(x_bits, x_bits_0, P)
# x_0 = self.sub_mode_L_minus_one(x, x_1)
# x_bit0 = np.bitwise_and(x, 1, out=x)  # x_bit0 = x % 2
# x_bit_0_1 = backend.subtract(x_bit0, x_bit_0_0, out=x_bit0)
