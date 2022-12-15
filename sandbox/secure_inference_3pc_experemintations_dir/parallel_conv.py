import numpy as np
import time
from numba import njit, prange

@njit(parallel=True)
def mat_mult(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]), )
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i,j] += A[i,k] * B[k,j]
    return res

# torch.Size([1, 4096, 4608])
# torch.Size([4608, 512])

m, n, c = 4096, 4608, 512
A = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(m, n))
B = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(n, c))

t0 = time.time()
res = A @ B
print(time.time() - t0)

print(A.dtype)
t0 = time.time()
res = mat_mult(A, B)
print(time.time() - t0)

print("Start")
print(A.shape, A.dtype)
print(B.shape, B.dtype)
t0 = time.time()
res = A @ B
print(time.time() - t0)