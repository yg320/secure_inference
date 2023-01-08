from research.secure_inference_3pc.conv2d_torch import Conv2DHandler
import numpy as np
import torch
import time

conv2d_handler = Conv2DHandler("cuda:0")
dtype = np.int64
low = np.iinfo(dtype).min
high = np.iinfo(dtype).max
repetitions = 10
np.random.seed(1)
x = [np.random.randint(low, high, (1, 256, 14, 14), dtype=dtype) for _ in range(repetitions)]
w = [np.random.randint(low, high, (256, 256, 3, 3), dtype=dtype) for _ in range(repetitions)]

t0 = time.time()
for i in range(repetitions):
    conv2d_handler.conv2d(x[i], w[i], (1, 1), (1, 1), (1, 1), 1)
print(time.time() - t0)