import numpy as np
import time
powers = np.array([[27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8]], dtype=np.int64)
P = 67

from numba import njit, prange
@njit('(int32[:,:], int64[:], int8[:,:], int8[:],  uint8, uint8)', parallel=True,  nogil=True, cache=True)
def decompose_numba(s, r, x_bits_0, beta, bits, ignore_msb_bits):

    for i in prange(x_bits_0.shape[0]):
        r[i] = r[i] + beta[i]

        counter = 0

        for j in range(bits - ignore_msb_bits):
            decompose_bit = (r[i] >> (bits - 1 - j)) & 1
            decompose_bit = -2 * decompose_bit * x_bits_0[i, j] + x_bits_0[i, j]
            counter = counter + decompose_bit

            tmp = (counter - decompose_bit) + x_bits_0[i, j] * (2 * beta[i] - 1)
            s[i, j] = (tmp * s[i, j]) % 67

# 32.487791

np.random.seed(123)
x_bits_0 = np.random.randint(low=0, high=67, dtype=np.int8, size=(100000, 20))
r = np.random.randint(low=-9223129350866284269, high=9221000347554954030, dtype=np.int64, size=(100000,))
beta = np.random.randint(low=0, high=2, dtype=np.int8, size=(100000,))
s = np.random.randint(low=0, high=67, dtype=np.int32, size=(100000, 20))

t0 = time.time()

decompose_numba(s, r, x_bits_0, beta, 28, 8)
d_bits_0 =  s
t1 = time.time()
print(t1 - t0)
print(d_bits_0.mean())























import numpy as np
import time
powers = np.array([[27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8]], dtype=np.int64)
P = 67

def decompose(value):

    orig_shape = list(value.shape)
    value = value.reshape(-1, 1)
    value_bits = np.zeros(shape=(value.shape[0], 20), dtype=np.int8)
    value_bits = np.right_shift(value, powers, out=value_bits)
    value_bits = np.bitwise_and(value_bits, 1, out=value_bits)
    ret = value_bits.reshape(orig_shape + [20])
    return ret

def get_c_party_0(x_bits, multiplexer_bits, beta):
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

min_org_shit = -283206
max_org_shit = 287469
org_shit = (np.arange(min_org_shit, max_org_shit + 1) % P).astype(np.int8)

def module_67(xxx):

    orig_shape = xxx.shape
    xxx = xxx.reshape(-1)
    np.subtract(xxx, min_org_shit, out=xxx)
    return org_shit[xxx.astype(np.int64)].reshape(orig_shape)


t0 = time.time()
r[beta.astype(bool)] += 1
bits = decompose(r)
c_bits_0 = get_c_party_0(x_bits_0, bits, beta)
s = np.multiply(s, c_bits_0, out=s)
d_bits_0 = module_67(s)
t1 = time.time()

print(t1 - t0)



















#
# eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)  # TODO: Optimize this
# t00 = backend.multiply(eta_pp, eta_p_1, out=eta_pp)
# t11 = self.add_mode_L_minus_one(t00, t00)  # TODO: Optimize this
# eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)  # TODO: Optimize this
# t00 = self.add_mode_L_minus_one(delta_1, eta_1)  # TODO: Optimize this
# theta_1 = self.add_mode_L_minus_one(beta_1, t00)  # TODO: Optimize this
# y_1 = self.sub_mode_L_minus_one(a_1, theta_1)  # TODO: Optimize this
# y_1 = self.add_mode_L_minus_one(y_1, mu_1)  # TODO: Optimize this
# return y_1