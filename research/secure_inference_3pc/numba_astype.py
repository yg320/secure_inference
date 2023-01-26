from numba import njit, prange, uint64
import numpy as np


@njit('(uint64[:])(int64[:])', parallel=True,  nogil=True, cache=True)
def convert(x):
    y = np.zeros(x.shape[0], dtype=np.uint64)
    z = uint64(0)
    r = uint64(0)
    for i in prange(x.shape[0],):
        z = x[i]
    return y

x = np.random.randint(low=-10000, high=10000, dtype=np.int64, size=(100000,))
y = convert(x)

print('fsd')