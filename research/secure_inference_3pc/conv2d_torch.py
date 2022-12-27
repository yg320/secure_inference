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

mask0_cuda = mask0.to("cuda:0")
mask1_cuda = mask1.to("cuda:0")
mask2_cuda = mask2.to("cuda:0")
mask3_cuda = mask3.to("cuda:0")

def conv2d_torch_cpu(a, b, padding, stride, dilation, groups):
    a_cpu = torch.from_numpy(a)
    b_cpu = torch.from_numpy(b)

    x0_cpu = ( a_cpu & mask0).to(torch.float64)
    x1_cpu = ((a_cpu & mask1) >> 16).to(torch.float64)
    x2_cpu = ((a_cpu & mask2) >> 32).to(torch.float64)
    x3_cpu = ((a_cpu & mask3) >> 48).to(torch.float64)
    y0_cpu = ( b_cpu & mask0).to(torch.float64)
    y1_cpu = ((b_cpu & mask1) >> 16).to(torch.float64)
    y2_cpu = ((b_cpu & mask2) >> 32).to(torch.float64)
    y3_cpu = ((b_cpu & mask3) >> 48).to(torch.float64)

    z0_cpu = torch.conv2d(x0_cpu, y0_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z1_cpu = torch.conv2d(x0_cpu, y1_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z2_cpu = torch.conv2d(x0_cpu, y2_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z3_cpu = torch.conv2d(x0_cpu, y3_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z4_cpu = torch.conv2d(x1_cpu, y0_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z5_cpu = torch.conv2d(x1_cpu, y1_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z6_cpu = torch.conv2d(x1_cpu, y2_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z7_cpu = torch.conv2d(x2_cpu, y0_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z8_cpu = torch.conv2d(x2_cpu, y1_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z9_cpu = torch.conv2d(x3_cpu, y0_cpu, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)

    res_float = \
        (z0_cpu) + \
        (z1_cpu << 16) + \
        (z2_cpu << 32) + \
        (z3_cpu << 48) + \
        (z4_cpu << 16) + \
        (z5_cpu << 32) + \
        (z6_cpu << 48) + \
        (z7_cpu << 32) + \
        (z8_cpu << 48) + \
        (z9_cpu << 48)

    return res_float.numpy()

def conv2d_torch_cuda(a, b, padding, stride, dilation, groups):
    a_cuda = torch.from_numpy(a).to("cuda:0")
    b_cuda = torch.from_numpy(b).to("cuda:0")

    x0_cuda = (a_cuda & mask0_cuda).to(torch.float64)
    x1_cuda = ((a_cuda & mask1_cuda) >> 16).to(torch.float64)
    x2_cuda = ((a_cuda & mask2_cuda) >> 32).to(torch.float64)
    x3_cuda = ((a_cuda & mask3_cuda) >> 48).to(torch.float64)
    y0_cuda = (b_cuda & mask0_cuda).to(torch.float64)
    y1_cuda = ((b_cuda & mask1_cuda) >> 16).to(torch.float64)
    y2_cuda = ((b_cuda & mask2_cuda) >> 32).to(torch.float64)
    y3_cuda = ((b_cuda & mask3_cuda) >> 48).to(torch.float64)

    z0_cuda = torch.conv2d(x0_cuda, y0_cuda, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z1_cuda = torch.conv2d(x0_cuda, y1_cuda, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z2_cuda = torch.conv2d(x0_cuda, y2_cuda, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z3_cuda = torch.conv2d(x0_cuda, y3_cuda, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z4_cuda = torch.conv2d(x1_cuda, y0_cuda, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z5_cuda = torch.conv2d(x1_cuda, y1_cuda, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z6_cuda = torch.conv2d(x1_cuda, y2_cuda, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z7_cuda = torch.conv2d(x2_cuda, y0_cuda, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z8_cuda = torch.conv2d(x2_cuda, y1_cuda, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)
    z9_cuda = torch.conv2d(x3_cuda, y0_cuda, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64)

    res_float_cuda = \
        (z0_cuda) + \
        (z1_cuda << 16) + \
        (z2_cuda << 32) + \
        (z3_cuda << 48) + \
        (z4_cuda << 16) + \
        (z5_cuda << 32) + \
        (z6_cuda << 48) + \
        (z7_cuda << 32) + \
        (z8_cuda << 48) + \
        (z9_cuda << 48)

    return res_float_cuda.cpu().numpy()

def conv2d_torch(a, b, padding, stride, dilation, groups):
    out_cuda = conv2d_torch_cuda(a, b, padding, stride, dilation, groups)
    # out_cpu = conv2d_torch_cpu(a, b, padding, stride, dilation, groups)
    # if not np.all(out_cpu == out_cuda):
    #     print('fds')
    return out_cuda