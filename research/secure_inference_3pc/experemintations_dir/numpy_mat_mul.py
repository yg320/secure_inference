import numpy as np
import time



m, n, c = 4096, 4608, 512
A_int64 = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(m, n))
B_int64 = np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=(n, c))

A_float64 = np.random.random(size=(m, n))
B_float64 = np.random.random(size=(n, c))

t0 = time.time()
C_float64 = A_float64 @ B_float64
print(time.time() - t0)

t0 = time.time()
C_int64 = A_int64 @ B_int64
print(time.time() - t0)

