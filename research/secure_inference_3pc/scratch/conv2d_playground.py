# import numpy as np
# dtype = np.int64
# low = np.iinfo(dtype).min
# high = np.iinfo(dtype).max
# a = np.random.randint(low, high, dtype=dtype)
# b = np.random.randint(low, high, dtype=dtype)
#
# mask0 = np.uint64(sum(2**i for i in range(32))).astype(np.int64)
# mask1 = np.uint64(sum(2**i for i in range(32, 64))).astype(np.int64)
# # np.binary_repr(mask0, width=64)
# # np.binary_repr(mask1, width=64)
# x0 = a & mask0
# x1 = (a & mask1) >> 32
# y0 = b & mask0
# y1 = (b & mask1) >> 32
#
# x0_float = np.float64(x0)
# x1_float = np.float64(x1)
# y0_float = np.float64(y0)
# y1_float = np.float64(y1)
#
# z0 = np.int64(x0*y0)
# z1 = np.int64(x0*y1)
# z2 = np.int64(x1*y0)
# z3 = np.int64(x1*y1)
#
# z0_float = np.int64(x0_float*y0_float)
# z1_float = np.int64(x0_float*y1_float)
# z2_float = np.int64(x1_float*y0_float)
# z3_float = np.int64(x1_float*y1_float)
#
# res = z0_float + (z1_float << 32) + (z2_float << 32)
#
# res_real = a * b
#
# np.binary_repr((a & mask0).astype(np.int32), width=32)
# np.binary_repr(((a & mask1)>>32).astype(np.int32), width=32)
# np.binary_repr(a, width=64)
# np.binary_repr(b, width=64)
# a >> 32



# # Divide to 4 parts
# import numpy as np
# dtype = np.int64
# low = np.iinfo(dtype).min
# high = np.iinfo(dtype).max
# a = np.random.randint(low, high, dtype=dtype)
# b = np.random.randint(low, high, dtype=dtype)
#
# mask0 = np.uint64(sum(2**i for i in range(16))).astype(np.int64)
# mask1 = np.uint64(sum(2**i for i in range(16, 32))).astype(np.int64)
# mask2 = np.uint64(sum(2**i for i in range(32, 48))).astype(np.int64)
# mask3 = np.uint64(sum(2**i for i in range(48, 64))).astype(np.int64)
#
# x0 = a & mask0
# x1 = (a & mask1) >> 16
# x2 = (a & mask2) >> 32
# x3 = (a & mask3) >> 48
#
# y0 = b & mask0
# y1 = (b & mask1) >> 16
# y2 = (b & mask2) >> 32
# y3 = (b & mask3) >> 48
#
# x0_float = np.float64(x0)
# x1_float = np.float64(x1)
# x2_float = np.float64(x2)
# x3_float = np.float64(x3)
# y0_float = np.float64(y0)
# y1_float = np.float64(y1)
# y2_float = np.float64(y2)
# y3_float = np.float64(y3)
#
# res = x0*y0 + (x0*y1 << 16) + (x0*y2 << 32) + (x0*y3 << 48) + (x1*y0 << 16) + (x1*y1 << 32) + (x1*y2 << 48) + (x1*y3 << 64) + (x2*y0 << 32) + (x2*y1 << 48) #+ (x2*y2 << 64) + (x2*y3 << 80) + (x3*y0 << 48) + (x3*y1 << 64) + (x3*y2 << 80) + (x3*y3 << 96)
# res_float = np.int64(x0_float*y0_float) + (np.int64(x0_float * y1_float) << 16) + (np.int64(x0_float * y2_float) << 32) + (np.int64(x0_float * y3_float) << 48) + (np.int64(x1_float * y0_float) << 16) + (np.int64(x1_float * y1_float) << 32) + (np.int64(x1_float * y2_float) << 48) + (np.int64(x1_float * y3_float) << 64) + (np.int64(x2_float * y0_float) << 32) + (np.int64(x2_float * y1_float) << 48) #+ (np.int64(x2_float * y2_float) << 64) + (np.int64(x2_float * y3_float) << 80) + (np.int64(x3_float * y0_float) << 48) + (np.int64(x3_float * y1_float) << 64) + (np.int64(x3_float * y2_float) << 80) + (np.int64(x3_float * y3_float) << 96)






# Divide to 4 parts
import numpy as np
import torch
from tqdm import tqdm


dtype = np.int64
low = np.iinfo(dtype).min
high = np.iinfo(dtype).max
a = torch.from_numpy(np.random.randint(low, high, size=(100000,), dtype=dtype))
b = torch.from_numpy(np.random.randint(low, high, size=(100000,), dtype=dtype))

mask0 = torch.from_numpy(np.uint64([sum(2**i for i in range(22))]).astype(np.int64))[0]
mask1 = torch.from_numpy(np.uint64([sum(2**i for i in range(22, 44))]).astype(np.int64))[0]
mask2 = torch.from_numpy(np.uint64([sum(2**i for i in range(44, 64))]).astype(np.int64))[0]

x0 = ((a & mask0) >> 0).to(torch.float64)
x1 = ((a & mask1) >> 22).to(torch.float64)
x2 = ((a & mask2) >> 44).to(torch.float64)
y0 = ((b & mask0) >> 0).to(torch.float64)
y1 = ((b & mask1) >> 22).to(torch.float64)
y2 = ((b & mask2) >> 44).to(torch.float64)

res_float = \
    ((x0*y0).to(torch.int64) << 0) + \
    ((x0*y1).to(torch.int64) << 22) + \
    ((x0*y2).to(torch.int64) << 44) + \
    ((x1*y0).to(torch.int64) << 22) + \
    ((x1*y1).to(torch.int64) << 44) + \
    ((x2*y0).to(torch.int64) << 44)

res_real = a * b

print(torch.all(res_float == res_real))

