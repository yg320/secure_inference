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

    x0 = a & mask0
    x1 = (a & mask1) >> 16
    x2 = (a & mask2) >> 32
    x3 = (a & mask3) >> 48
    y0 = b & mask0
    y1 = (b & mask1) >> 16
    y2 = (b & mask2) >> 32
    y3 = (b & mask3) >> 48

    x0_float = x0.to(torch.float64)
    x1_float = x1.to(torch.float64)
    x2_float = x2.to(torch.float64)
    x3_float = x3.to(torch.float64)
    y0_float = y0.to(torch.float64)
    y1_float = y1.to(torch.float64)
    y2_float = y2.to(torch.float64)
    y3_float = y3.to(torch.float64)

    # x_float = torch.cat([x0_float, x1_float, x2_float, x3_float], dim=0)
    #
    # z0, z1, z2, z3 = torch.conv2d(x_float, y0_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64)
    # t0, t1, t2 = torch.conv2d(x_float[:3], y1_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64)
    # r0, r1 = torch.conv2d(x_float[:2], y2_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64)
    # s0 = torch.conv2d(x_float[:1], y3_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64)
    #
    # res_float = \
    #     (z0 << 0) + \
    #     (t0 << 16) + \
    #     (r0 << 32) + \
    #     (s0 << 48) + \
    #     (z1 << 16) + \
    #     (t1 << 32) + \
    #     (r1 << 48) + \
    #     (z2 << 32) + \
    #     (t2 << 48) + \
    #     (z3 << 48)

    res_float = \
        (torch.conv2d(x0_float, y0_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 0) + \
        (torch.conv2d(x0_float, y1_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 16) + \
        (torch.conv2d(x0_float, y2_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 32) + \
        (torch.conv2d(x0_float, y3_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 48) + \
        (torch.conv2d(x1_float, y0_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 16) + \
        (torch.conv2d(x1_float, y1_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 32) + \
        (torch.conv2d(x1_float, y2_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 48) + \
        (torch.conv2d(x2_float, y0_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 32) + \
        (torch.conv2d(x2_float, y1_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 48) + \
        (torch.conv2d(x3_float, y0_float, stride=stride, padding=padding, dilation=dilation, groups=groups).to(torch.int64) << 48)
    # res_real = torch.conv2d(a, b, stride=stride, padding=padding, dilation=dilation, groups=groups).numpy()
    # ret_float = res_float.numpy()
    # if not np.all(ret_float == res_real):
    #     print('fdsfds')
    return res_float.numpy()