from research.secure_inference_3pc.const import SIGNED_DTYPE, P, NUM_BITS

import numpy as np

from numba import njit, prange, int64, uint64, int8, uint8, int32, uint32

NUMBA_INT_DTYPE = int64 if NUM_BITS == 64 else int32
NUMBA_UINT_DTYPE = uint64 if NUM_BITS == 64 else uint32


@njit((NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], int8[:, :], NUMBA_UINT_DTYPE[:], NUMBA_UINT_DTYPE[:],
       uint8), parallel=True, nogil=True, cache=True)
def processing_numba(x, x_1, x_bit_0_0, x_bits_0, x_uint64, x_1_uint64, num_bits_ignored):
    x_bits_1 = x_bits_0
    x_0 = x_1
    x_bit_0_1 = x

    # bits = bits - 1
    for i in prange(x_bits_1.shape[0]):
        for j in range(64 - num_bits_ignored):
            x_bit = (x[i] >> (64 - 1 - j)) & 1  # x_bits

            if x_bit >= x_bits_0[i, j]:
                x_bits_1[i][j] = x_bit - x_bits_0[i, j]
            else:
                x_bits_1[i][j] = x_bit - x_bits_0[i, j] + P

        if x_uint64[i] < x_1_uint64[i]:
            x_0[i] = x[i] - x_1[i] - 1
        else:
            x_0[i] = x[i] - x_1[i]
        x_bit0 = x[i] % 2
        x_bit_0_1[i] = x_bit0 - x_bit_0_0[i]


@njit((NUMBA_INT_DTYPE[:])(int8[:, :], int8[:, :]), parallel=True, nogil=True, cache=True)
def numba_private_compare(d_bits_0, d_bits_1):
    out = np.zeros(shape=(d_bits_0.shape[0],), dtype=SIGNED_DTYPE)
    for i in prange(d_bits_0.shape[0]):
        for j in range(d_bits_0.shape[1]):
            a = (d_bits_0[i, j] + d_bits_1[i, j])
            if a == 0 or a == 67:
                out[i] = 1
                break

    return out
