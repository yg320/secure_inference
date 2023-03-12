import numpy as np
import torch

device = "cuda:0"

mask0 = torch.from_numpy(np.uint64([sum(2**i for i in range(0,  4))]).astype(np.int64))[0].to(device)
mask1 = torch.from_numpy(np.uint64([sum(2**i for i in range(4,  8))]).astype(np.int64))[0].to(device)
mask2 = torch.from_numpy(np.uint64([sum(2**i for i in range(8, 12))]).astype(np.int64))[0].to(device)
mask3 = torch.from_numpy(np.uint64([sum(2**i for i in range(12, 16))]).astype(np.int64))[0].to(device)
mask4 = torch.from_numpy(np.uint64([sum(2**i for i in range(16, 20))]).astype(np.int64))[0].to(device)
mask5 = torch.from_numpy(np.uint64([sum(2**i for i in range(20, 24))]).astype(np.int64))[0].to(device)
mask6 = torch.from_numpy(np.uint64([sum(2**i for i in range(24, 28))]).astype(np.int64))[0].to(device)
mask7 = torch.from_numpy(np.uint64([sum(2**i for i in range(28, 32))]).astype(np.int64))[0].to(device)
mask8 = torch.from_numpy(np.uint64([sum(2**i for i in range(32, 36))]).astype(np.int64))[0].to(device)
mask9 = torch.from_numpy(np.uint64([sum(2**i for i in range(36, 40))]).astype(np.int64))[0].to(device)
mask10 = torch.from_numpy(np.uint64([sum(2**i for i in range(40, 44))]).astype(np.int64))[0].to(device)
mask11 = torch.from_numpy(np.uint64([sum(2**i for i in range(44, 48))]).astype(np.int64))[0].to(device)
mask12 = torch.from_numpy(np.uint64([sum(2**i for i in range(48, 52))]).astype(np.int64))[0].to(device)
mask13 = torch.from_numpy(np.uint64([sum(2**i for i in range(52, 56))]).astype(np.int64))[0].to(device)
mask14 = torch.from_numpy(np.uint64([sum(2**i for i in range(56, 60))]).astype(np.int64))[0].to(device)
mask15 = torch.from_numpy(np.uint64([sum(2**i for i in range(60, 64))]).astype(np.int64))[0].to(device)

def conv2d_torch(a, b, stride, padding, dilation, groups):
    a = torch.from_numpy(a).to(device)
    b = torch.from_numpy(b).to(device)
    x0 =  ((a & mask0) >> 0  ).to(torch.float32)
    x1 =  ((a & mask1) >> 4  ).to(torch.float32)
    x2 =  ((a & mask2) >> 8  ).to(torch.float32)
    x3 =  ((a & mask3) >> 12 ).to(torch.float32)
    x4 =  ((a & mask4) >> 16 ).to(torch.float32)
    x5 =  ((a & mask5) >> 20 ).to(torch.float32)
    x6 =  ((a & mask6) >> 24 ).to(torch.float32)
    x7 =  ((a & mask7) >> 28 ).to(torch.float32)
    x8 =  ((a & mask8) >> 32 ).to(torch.float32)
    x9 =  ((a & mask9) >> 36 ).to(torch.float32)
    x10 = ((a & mask10) >> 40).to(torch.float32)
    x11 = ((a & mask11) >> 44).to(torch.float32)
    x12 = ((a & mask12) >> 48).to(torch.float32)
    x13 = ((a & mask13) >> 52).to(torch.float32)
    x14 = ((a & mask14) >> 56).to(torch.float32)
    x15 = ((a & mask15) >> 60).to(torch.float32)

    y0 =  ((b & mask0) >> 0  ).to(torch.float32)
    y1 =  ((b & mask1) >> 4  ).to(torch.float32)
    y2 =  ((b & mask2) >> 8  ).to(torch.float32)
    y3 =  ((b & mask3) >> 12 ).to(torch.float32)
    y4 =  ((b & mask4) >> 16 ).to(torch.float32)
    y5 =  ((b & mask5) >> 20 ).to(torch.float32)
    y6 =  ((b & mask6) >> 24 ).to(torch.float32)
    y7 =  ((b & mask7) >> 28 ).to(torch.float32)
    y8 =  ((b & mask8) >> 32 ).to(torch.float32)
    y9 =  ((b & mask9) >> 36 ).to(torch.float32)
    y10 = ((b & mask10) >> 40).to(torch.float32)
    y11 = ((b & mask11) >> 44).to(torch.float32)
    y12 = ((b & mask12) >> 48).to(torch.float32)
    y13 = ((b & mask13) >> 52).to(torch.float32)
    y14 = ((b & mask14) >> 56).to(torch.float32)
    y15 = ((b & mask15) >> 60).to(torch.float32)

    res_float = \
        (torch.conv2d(x0, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 0) + \
        (torch.conv2d(x0, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 4) + \
        (torch.conv2d(x0, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 8) + \
        (torch.conv2d(x0, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 12) + \
        (torch.conv2d(x0, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 16) + \
        (torch.conv2d(x0, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 20) + \
        (torch.conv2d(x0, y6, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 24) + \
        (torch.conv2d(x0, y7, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 28) + \
        (torch.conv2d(x0, y8, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 32) + \
        (torch.conv2d(x0, y9, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 36) + \
        (torch.conv2d(x0, y10, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x0, y11, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x0, y12, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x0, y13, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x0, y14, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x0, y15, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x1, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 4) + \
        (torch.conv2d(x1, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 8) + \
        (torch.conv2d(x1, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 12) + \
        (torch.conv2d(x1, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 16) + \
        (torch.conv2d(x1, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 20) + \
        (torch.conv2d(x1, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 24) + \
        (torch.conv2d(x1, y6, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 28) + \
        (torch.conv2d(x1, y7, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 32) + \
        (torch.conv2d(x1, y8, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 36) + \
        (torch.conv2d(x1, y9, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x1, y10, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x1, y11, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x1, y12, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x1, y13, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x1, y14, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x2, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 8) + \
        (torch.conv2d(x2, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 12) + \
        (torch.conv2d(x2, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 16) + \
        (torch.conv2d(x2, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 20) + \
        (torch.conv2d(x2, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 24) + \
        (torch.conv2d(x2, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 28) + \
        (torch.conv2d(x2, y6, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 32) + \
        (torch.conv2d(x2, y7, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 36) + \
        (torch.conv2d(x2, y8, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x2, y9, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x2, y10, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x2, y11, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x2, y12, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x2, y13, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x3, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 12) + \
        (torch.conv2d(x3, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 16) + \
        (torch.conv2d(x3, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 20) + \
        (torch.conv2d(x3, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 24) + \
        (torch.conv2d(x3, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 28) + \
        (torch.conv2d(x3, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 32) + \
        (torch.conv2d(x3, y6, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 36) + \
        (torch.conv2d(x3, y7, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x3, y8, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x3, y9, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x3, y10, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x3, y11, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x3, y12, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x4, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 16) + \
        (torch.conv2d(x4, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 20) + \
        (torch.conv2d(x4, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 24) + \
        (torch.conv2d(x4, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 28) + \
        (torch.conv2d(x4, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 32) + \
        (torch.conv2d(x4, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 36) + \
        (torch.conv2d(x4, y6, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x4, y7, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x4, y8, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x4, y9, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x4, y10, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x4, y11, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x5, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 20) + \
        (torch.conv2d(x5, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 24) + \
        (torch.conv2d(x5, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 28) + \
        (torch.conv2d(x5, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 32) + \
        (torch.conv2d(x5, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 36) + \
        (torch.conv2d(x5, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x5, y6, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x5, y7, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x5, y8, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x5, y9, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x5, y10, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x6, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 24) + \
        (torch.conv2d(x6, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 28) + \
        (torch.conv2d(x6, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 32) + \
        (torch.conv2d(x6, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 36) + \
        (torch.conv2d(x6, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x6, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x6, y6, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x6, y7, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x6, y8, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x6, y9, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x7, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 28) + \
        (torch.conv2d(x7, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 32) + \
        (torch.conv2d(x7, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 36) + \
        (torch.conv2d(x7, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x7, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x7, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x7, y6, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x7, y7, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x7, y8, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x8, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 32) + \
        (torch.conv2d(x8, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 36) + \
        (torch.conv2d(x8, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x8, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x8, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x8, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x8, y6, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x8, y7, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x9, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 36) + \
        (torch.conv2d(x9, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x9, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x9, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x9, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x9, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x9, y6, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x10, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 40) + \
        (torch.conv2d(x10, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x10, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x10, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x10, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x10, y5, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x11, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 44) + \
        (torch.conv2d(x11, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x11, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x11, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x11, y4, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x12, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 48) + \
        (torch.conv2d(x12, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x12, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x12, y3, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x13, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 52) + \
        (torch.conv2d(x13, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x13, y2, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x14, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 56) + \
        (torch.conv2d(x14, y1, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60) + \
        (torch.conv2d(x15, y0, stride=stride, padding=padding, dilation=dilation, groups=groups).round().to(torch.int64) << 60)

    return res_float.cpu().numpy()


