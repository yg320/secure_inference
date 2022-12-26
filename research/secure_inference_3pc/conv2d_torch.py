import torch
import numpy as np
#
# mask0 = torch.from_numpy(np.uint64([sum(2**i for i in range(22))]).astype(np.int64))[0]
# mask1 = torch.from_numpy(np.uint64([sum(2**i for i in range(22, 44))]).astype(np.int64))[0]
# mask2 = torch.from_numpy(np.uint64([sum(2**i for i in range(44, 64))]).astype(np.int64))[0]
#
#
# def conv2d_torch(a, b, padding, stride, dilation, groups):
#     a = torch.from_numpy(a)
#     b = torch.from_numpy(b)
#
#     x0 = ((a & mask0) >> 0).to(torch.float64)
#     x1 = ((a & mask1) >> 22).to(torch.float64)
#     x2 = ((a & mask2) >> 44).to(torch.float64)
#     y0 = ((b & mask0) >> 0).to(torch.float64)
#     y1 = ((b & mask1) >> 22).to(torch.float64)
#     y2 = ((b & mask2) >> 44).to(torch.float64)
#
#     res_float = \
#         (torch.conv2d(x0, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 0) + \
#         (torch.conv2d(x0, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 22) + \
#         (torch.conv2d(x0, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 44) + \
#         (torch.conv2d(x1, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 22) + \
#         (torch.conv2d(x1, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 44) + \
#         (torch.conv2d(x2, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 44)
#
#     res_real = torch.conv2d(a, b, stride=stride, padding=padding, dilation=dilation, groups=groups).numpy()
#     ret_float = res_float.numpy()
#     if not np.all(ret_float == res_real):
#         print('fdsfds')
#     return ret_float


mask0 = torch.from_numpy(np.uint64([sum(2**i for i in range(16))]).astype(np.int64))[0]
mask1 = torch.from_numpy(np.uint64([sum(2**i for i in range(16, 32))]).astype(np.int64))[0]
mask2 = torch.from_numpy(np.uint64([sum(2**i for i in range(32, 48))]).astype(np.int64))[0]
mask3 = torch.from_numpy(np.uint64([sum(2**i for i in range(48, 64))]).astype(np.int64))[0]


def conv2d_torch(a, b, padding, stride, dilation, groups):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)

    x0 = (a & mask0).to(torch.float64)
    x1 = ((a & mask1) >> 16).to(torch.float64)
    x2 = ((a & mask2) >> 32).to(torch.float64)
    x3 = ((a & mask3) >> 48).to(torch.float64)
    y0 = (b & mask0).to(torch.float64)
    y1 = ((b & mask1) >> 16).to(torch.float64)
    y2 = ((b & mask2) >> 32).to(torch.float64)
    y3 = ((b & mask3) >> 48).to(torch.float64)

    res_float = \
        (torch.conv2d(x0, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64)) + \
        (torch.conv2d(x0, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 16) + \
        (torch.conv2d(x0, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 32) + \
        (torch.conv2d(x0, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 48) + \
        (torch.conv2d(x1, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 16) + \
        (torch.conv2d(x1, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 32) + \
        (torch.conv2d(x1, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 48) + \
        (torch.conv2d(x2, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 32) + \
        (torch.conv2d(x2, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 48) + \
        (torch.conv2d(x3, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 48)

    return res_float.numpy()