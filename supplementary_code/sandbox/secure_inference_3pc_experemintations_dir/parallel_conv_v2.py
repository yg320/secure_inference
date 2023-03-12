import numpy as np

import time
from numba import njit, prange

@njit(parallel=True)
def conv(A, B):

    res = np.zeros((B.shape[0], A.shape[2] - 2, A.shape[3] - 2), dtype=np.int64)

    for out_channel in prange(B.shape[0]):
        for in_channel in range(A.shape[1]):
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    for k in range(3):
                        for l in range(3):
                            res[out_channel, i, j] += B[out_channel, in_channel, k, l] * A[0,in_channel, i+k, j+l]

    return res


# ================== (1, 640, 24, 24) (128, 640, 3, 3)
# ================== (1, 576, 5760) (5760, 128)


for i in range(10):
    A = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(1, 640, 24, 24))
    A = np.pad(A, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

    B = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(128, 640, 3, 3))
    t0 = time.time()
    res = conv(A, B)
    print(time.time() - t0)
