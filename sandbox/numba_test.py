import numpy as np
from numba import njit, prange, int64, int32
import numba as nb
import time
from research.secure_inference_3pc.const import SIGNED_DTYPE, NUM_BITS
NUMBA_MATMUL = "NUMBA_MATMUL"
NUMBA_CONV = "NUMBA_CONV"

NUMBA_DTYPE = int64 if NUM_BITS == 64 else int32

def subtract_module(x, y, P):
    ret = np.subtract(x, y, out=x)
    ret[ret < 0] += P
    return ret


def sub_mode_L_minus_one(a, b):
    ret = a - b
    ret[a.astype(np.uint64, copy=False) > b.astype(np.uint64, copy=False)] -= 1
    return ret

num_of_compare_bits = 28
ignore_msb_bits = 8
end = None if ignore_msb_bits == 0 else -ignore_msb_bits
powers = np.arange(NUM_BITS, dtype=np.int64)[np.newaxis][:,::-1][:, NUM_BITS - num_of_compare_bits:end]


def decompose(x):
    orig_shape = list(x.shape)
    value = x.reshape(-1, 1)

    value_bits = np.zeros(shape=(value.shape[0], num_of_compare_bits - ignore_msb_bits), dtype=np.int8)
    value_bits = np.right_shift(value, powers, out=value_bits)
    value_bits = np.bitwise_and(value_bits, 1, out=value_bits)
    ret = value_bits.reshape(orig_shape + [num_of_compare_bits - ignore_msb_bits])
    return ret




P = 67


def processing(x, x_1, x_bit_0_0, x_bits, x_bits_0):
    x_bits_1 = subtract_module(x_bits, x_bits_0, P)
    x_0 = sub_mode_L_minus_one(x, x_1)
    x_bit0 = np.bitwise_and(x, 1, out=x)  # x_bit0 = x % 2
    x_bit_0_1 = np.subtract(x_bit0, x_bit_0_0, out=x_bit0)

    return x_bits_1, x_0,  x_bit_0_1

@njit('Tuple((int8[:, :], int64[:], int64[:]))(int64[:], int64[:], int64[:], int8[:,:], uint64[:], uint64[:], uint8, uint8)', parallel=True,  nogil=True, cache=True)
def processing_numba(x, x_1, x_bit_0_0, x_bits_0, x_uint64, x_1_uint64, bits, ignore_msb_bits):
    x_bits_1 = x_bits_0
    x_0 = x_1
    x_bit_0_1 = x_bit_0_0
    # bits = bits - 1
    for i in prange(x_bits_1.shape[0]):
        for j in range(bits - ignore_msb_bits):
            a = (x[i] >> (bits - 1 - j)) & 1

            if a >= x_bits_0[i, j]:
                x_bits_1[i][j] = a - x_bits_0[i, j]
            else:
                x_bits_1[i][j] = a - x_bits_0[i, j] + P

        if x_uint64[i] > x_1_uint64[i]:
            x_0[i] = x[i] - x_1[i] - 1
        else:
            x_0[i] = x[i] - x_1[i]

        x_bit_0_1[i] = x[i] % 2 - x_bit_0_0[i]

    return x_bits_1, x_0, x_bit_0_1




numba_time = 0
non_numba_time = 0
for i in range(10):
    x_bits_0 = np.random.randint(0, P, size=(10000, num_of_compare_bits - ignore_msb_bits), dtype=np.int8)
    x = np.random.randint(-100000, 100000, size=(10000,), dtype=np.int64)
    x_1 = np.random.randint(-100000, 100000, size=(10000,), dtype=np.int64)
    x_bit_0_0 = np.random.randint(-100000, 100000, size=(10000,), dtype=np.int64)

    x_bits_0_nb = x_bits_0.copy()
    x_nb = x.copy()
    x_1_nb = x_1.copy()
    x_bit_0_0_nb = x_bit_0_0.copy()

    t0 = time.time()
    x_bits = decompose(x)
    x_bits_1, x_0,  x_bit_0_1 = processing(x, x_1, x_bit_0_0, x_bits, x_bits_0)
    t1 = time.time()
    x_bits_1_, x_0_,  x_bit_0_1_ = processing_numba(x_nb, x_1_nb, x_bit_0_0_nb, x_bits_0_nb, x_nb.astype(np.uint64, copy=False), x_1_nb.astype(np.uint64, copy=False), num_of_compare_bits, ignore_msb_bits)
    t2 = time.time()
    numba_time += t2 - t1
    non_numba_time += t1 - t0
    print((x_bits_1 == x_bits_1_).all())
    print((x_0 == x_0_).all())
    print((x_bit_0_1 == x_bit_0_1_).all())
print("Numba", numba_time / non_numba_time)
# print("Reg", t1 - t0)