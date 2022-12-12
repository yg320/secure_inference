import numpy as np
import torch

import time
from numba import njit, prange

@njit(parallel=True)
def conv(A, B):
    k0, k1 = B.shape[2:]

    assert k0 % 2 == 1
    assert k1 % 2 == 1


    res = np.zeros((B.shape[0], A.shape[2], A.shape[3]), dtype=A.dtype)


    for m in range(1, A.shape[2] - 1):
        for l in range(1, A.shape[3] - 1):
            a = A[:, :, m-1:m+2, l-1:l+2]
            b = B
            tt = (a*b).sum(axis=(1,2,3))
            res[:, m, l] += tt
    return res

A = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(1, 512, 64, 64))
B = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(512, 512, 3, 3))

# a = A[:, :3, :3][np.newaxis]
# b = B
# (a*b).sum(axis=(1,2,3))
# C = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(512, 64, 64))
# D = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(512, 512, 3, 3))
t0 = time.time()
res = conv(A, B)
print(time.time() - t0)

# torch.Size([1, 4096, 4608])
# torch.Size([4608, 512])
