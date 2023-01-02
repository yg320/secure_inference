import torch
import numpy as np


class Conv2DHandler:
    def __init__(self, device):
        self.device = device

        self.mask0_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(0, 4))]).astype(np.int64))[0].to(device)
        self.mask1_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(4, 8))]).astype(np.int64))[0].to(device)
        self.mask2_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(8, 12))]).astype(np.int64))[0].to(device)
        self.mask3_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(12, 16))]).astype(np.int64))[0].to(device)
        self.mask4_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(16, 20))]).astype(np.int64))[0].to(device)
        self.mask5_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(20, 24))]).astype(np.int64))[0].to(device)
        self.mask6_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(24, 28))]).astype(np.int64))[0].to(device)
        self.mask7_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(28, 32))]).astype(np.int64))[0].to(device)
        self.mask8_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(32, 36))]).astype(np.int64))[0].to(device)
        self.mask9_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(36, 40))]).astype(np.int64))[0].to(device)
        self.mask10_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(40, 44))]).astype(np.int64))[0].to(device)
        self.mask11_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(44, 48))]).astype(np.int64))[0].to(device)
        self.mask12_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(48, 52))]).astype(np.int64))[0].to(device)
        self.mask13_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(52, 56))]).astype(np.int64))[0].to(device)
        self.mask14_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(56, 60))]).astype(np.int64))[0].to(device)
        self.mask15_4 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(60, 64))]).astype(np.int64))[0].to(device)

        self.mask0_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(0, 8))]).astype(np.int64))[0].to(device)
        self.mask1_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(8, 16))]).astype(np.int64))[0].to(device)
        self.mask2_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(16, 24))]).astype(np.int64))[0].to(device)
        self.mask3_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(24, 32))]).astype(np.int64))[0].to(device)
        self.mask4_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(32, 40))]).astype(np.int64))[0].to(device)
        self.mask5_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(40, 48))]).astype(np.int64))[0].to(device)
        self.mask6_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(48, 56))]).astype(np.int64))[0].to(device)
        self.mask7_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(56, 64))]).astype(np.int64))[0].to(device)

    def conv2d_torch_4(self, a, b, stride, padding, dilation, groups, dtype=torch.float32):
        
        kwargs = {'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}
        
        a = torch.from_numpy(a).to(self.device)
        b = torch.from_numpy(b).to(self.device)

        x0 = ((a & self.mask0_4) >> 0).to(dtype)
        x1 = ((a & self.mask1_4) >> 4).to(dtype)
        x2 = ((a & self.mask2_4) >> 8).to(dtype)
        x3 = ((a & self.mask3_4) >> 12).to(dtype)
        x4 = ((a & self.mask4_4) >> 16).to(dtype)
        x5 = ((a & self.mask5_4) >> 20).to(dtype)
        x6 = ((a & self.mask6_4) >> 24).to(dtype)
        x7 = ((a & self.mask7_4) >> 28).to(dtype)
        x8 = ((a & self.mask8_4) >> 32).to(dtype)
        x9 = ((a & self.mask9_4) >> 36).to(dtype)
        x10 = ((a & self.mask10_4) >> 40).to(dtype)
        x11 = ((a & self.mask11_4) >> 44).to(dtype)
        x12 = ((a & self.mask12_4) >> 48).to(dtype)
        x13 = ((a & self.mask13_4) >> 52).to(dtype)
        x14 = ((a & self.mask14_4) >> 56).to(dtype)
        x15 = ((a & self.mask15_4) >> 60).to(dtype)

        y0 = ((b & self.mask0_4) >> 0).to(dtype)
        y1 = ((b & self.mask1_4) >> 4).to(dtype)
        y2 = ((b & self.mask2_4) >> 8).to(dtype)
        y3 = ((b & self.mask3_4) >> 12).to(dtype)
        y4 = ((b & self.mask4_4) >> 16).to(dtype)
        y5 = ((b & self.mask5_4) >> 20).to(dtype)
        y6 = ((b & self.mask6_4) >> 24).to(dtype)
        y7 = ((b & self.mask7_4) >> 28).to(dtype)
        y8 = ((b & self.mask8_4) >> 32).to(dtype)
        y9 = ((b & self.mask9_4) >> 36).to(dtype)
        y10 = ((b & self.mask10_4) >> 40).to(dtype)
        y11 = ((b & self.mask11_4) >> 44).to(dtype)
        y12 = ((b & self.mask12_4) >> 48).to(dtype)
        y13 = ((b & self.mask13_4) >> 52).to(dtype)
        y14 = ((b & self.mask14_4) >> 56).to(dtype)
        y15 = ((b & self.mask15_4) >> 60).to(dtype)

        res_float = \
            (torch.conv2d(x0, y0, **kwargs).round().to(torch.int64) << 0) + \
            (torch.conv2d(x0, y1, **kwargs).round().to(torch.int64) << 4) + \
            (torch.conv2d(x0, y2, **kwargs).round().to(torch.int64) << 8) + \
            (torch.conv2d(x0, y3, **kwargs).round().to(torch.int64) << 12) + \
            (torch.conv2d(x0, y4, **kwargs).round().to(torch.int64) << 16) + \
            (torch.conv2d(x0, y5, **kwargs).round().to(torch.int64) << 20) + \
            (torch.conv2d(x0, y6, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x0, y7, **kwargs).round().to(torch.int64) << 28) + \
            (torch.conv2d(x0, y8, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x0, y9, **kwargs).round().to(torch.int64) << 36) + \
            (torch.conv2d(x0, y10, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x0, y11, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x0, y12, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x0, y13, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x0, y14, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x0, y15, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x1, y0, **kwargs).round().to(torch.int64) << 4) + \
            (torch.conv2d(x1, y1, **kwargs).round().to(torch.int64) << 8) + \
            (torch.conv2d(x1, y2, **kwargs).round().to(torch.int64) << 12) + \
            (torch.conv2d(x1, y3, **kwargs).round().to(torch.int64) << 16) + \
            (torch.conv2d(x1, y4, **kwargs).round().to(torch.int64) << 20) + \
            (torch.conv2d(x1, y5, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x1, y6, **kwargs).round().to(torch.int64) << 28) + \
            (torch.conv2d(x1, y7, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x1, y8, **kwargs).round().to(torch.int64) << 36) + \
            (torch.conv2d(x1, y9, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x1, y10, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x1, y11, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x1, y12, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x1, y13, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x1, y14, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x2, y0, **kwargs).round().to(torch.int64) << 8) + \
            (torch.conv2d(x2, y1, **kwargs).round().to(torch.int64) << 12) + \
            (torch.conv2d(x2, y2, **kwargs).round().to(torch.int64) << 16) + \
            (torch.conv2d(x2, y3, **kwargs).round().to(torch.int64) << 20) + \
            (torch.conv2d(x2, y4, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x2, y5, **kwargs).round().to(torch.int64) << 28) + \
            (torch.conv2d(x2, y6, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x2, y7, **kwargs).round().to(torch.int64) << 36) + \
            (torch.conv2d(x2, y8, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x2, y9, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x2, y10, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x2, y11, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x2, y12, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x2, y13, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x3, y0, **kwargs).round().to(torch.int64) << 12) + \
            (torch.conv2d(x3, y1, **kwargs).round().to(torch.int64) << 16) + \
            (torch.conv2d(x3, y2, **kwargs).round().to(torch.int64) << 20) + \
            (torch.conv2d(x3, y3, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x3, y4, **kwargs).round().to(torch.int64) << 28) + \
            (torch.conv2d(x3, y5, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x3, y6, **kwargs).round().to(torch.int64) << 36) + \
            (torch.conv2d(x3, y7, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x3, y8, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x3, y9, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x3, y10, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x3, y11, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x3, y12, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x4, y0, **kwargs).round().to(torch.int64) << 16) + \
            (torch.conv2d(x4, y1, **kwargs).round().to(torch.int64) << 20) + \
            (torch.conv2d(x4, y2, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x4, y3, **kwargs).round().to(torch.int64) << 28) + \
            (torch.conv2d(x4, y4, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x4, y5, **kwargs).round().to(torch.int64) << 36) + \
            (torch.conv2d(x4, y6, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x4, y7, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x4, y8, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x4, y9, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x4, y10, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x4, y11, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x5, y0, **kwargs).round().to(torch.int64) << 20) + \
            (torch.conv2d(x5, y1, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x5, y2, **kwargs).round().to(torch.int64) << 28) + \
            (torch.conv2d(x5, y3, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x5, y4, **kwargs).round().to(torch.int64) << 36) + \
            (torch.conv2d(x5, y5, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x5, y6, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x5, y7, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x5, y8, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x5, y9, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x5, y10, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x6, y0, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x6, y1, **kwargs).round().to(torch.int64) << 28) + \
            (torch.conv2d(x6, y2, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x6, y3, **kwargs).round().to(torch.int64) << 36) + \
            (torch.conv2d(x6, y4, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x6, y5, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x6, y6, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x6, y7, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x6, y8, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x6, y9, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x7, y0, **kwargs).round().to(torch.int64) << 28) + \
            (torch.conv2d(x7, y1, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x7, y2, **kwargs).round().to(torch.int64) << 36) + \
            (torch.conv2d(x7, y3, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x7, y4, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x7, y5, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x7, y6, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x7, y7, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x7, y8, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x8, y0, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x8, y1, **kwargs).round().to(torch.int64) << 36) + \
            (torch.conv2d(x8, y2, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x8, y3, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x8, y4, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x8, y5, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x8, y6, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x8, y7, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x9, y0, **kwargs).round().to(torch.int64) << 36) + \
            (torch.conv2d(x9, y1, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x9, y2, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x9, y3, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x9, y4, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x9, y5, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x9, y6, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x10, y0, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x10, y1, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x10, y2, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x10, y3, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x10, y4, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x10, y5, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x11, y0, **kwargs).round().to(torch.int64) << 44) + \
            (torch.conv2d(x11, y1, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x11, y2, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x11, y3, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x11, y4, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x12, y0, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x12, y1, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x12, y2, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x12, y3, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x13, y0, **kwargs).round().to(torch.int64) << 52) + \
            (torch.conv2d(x13, y1, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x13, y2, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x14, y0, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x14, y1, **kwargs).round().to(torch.int64) << 60) + \
            (torch.conv2d(x15, y0, **kwargs).round().to(torch.int64) << 60)

        return res_float.cpu().numpy()

    def conv2d_torch_8(self, a, b, stride, padding, dilation, groups, dtype):
        kwargs = {'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}
        a = torch.from_numpy(a).to(self.device)
        b = torch.from_numpy(b).to(self.device)

        x0 = ((a & self.mask0_8) >> 0).to(dtype)
        x1 = ((a & self.mask1_8) >> 8).to(dtype)
        x2 = ((a & self.mask2_8) >> 16).to(dtype)
        x3 = ((a & self.mask3_8) >> 24).to(dtype)
        x4 = ((a & self.mask4_8) >> 32).to(dtype)
        x5 = ((a & self.mask5_8) >> 40).to(dtype)
        x6 = ((a & self.mask6_8) >> 48).to(dtype)
        x7 = ((a & self.mask7_8) >> 56).to(dtype)
        y0 = ((b & self.mask0_8) >> 0).to(dtype)
        y1 = ((b & self.mask1_8) >> 8).to(dtype)
        y2 = ((b & self.mask2_8) >> 16).to(dtype)
        y3 = ((b & self.mask3_8) >> 24).to(dtype)
        y4 = ((b & self.mask4_8) >> 32).to(dtype)
        y5 = ((b & self.mask5_8) >> 40).to(dtype)
        y6 = ((b & self.mask6_8) >> 48).to(dtype)
        y7 = ((b & self.mask7_8) >> 56).to(dtype)


        res_float = \
            (torch.conv2d(x0, y0, **kwargs).round().to(torch.int64) << 0) + \
            (torch.conv2d(x0, y1, **kwargs).round().to(torch.int64) << 8) + \
            (torch.conv2d(x0, y2, **kwargs).round().to(torch.int64) << 16) + \
            (torch.conv2d(x0, y3, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x0, y4, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x0, y5, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x0, y6, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x0, y7, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x1, y0, **kwargs).round().to(torch.int64) << 8) + \
            (torch.conv2d(x1, y1, **kwargs).round().to(torch.int64) << 16) + \
            (torch.conv2d(x1, y2, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x1, y3, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x1, y4, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x1, y5, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x1, y6, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x2, y0, **kwargs).round().to(torch.int64) << 16) + \
            (torch.conv2d(x2, y1, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x2, y2, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x2, y3, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x2, y4, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x2, y5, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x3, y0, **kwargs).round().to(torch.int64) << 24) + \
            (torch.conv2d(x3, y1, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x3, y2, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x3, y3, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x3, y4, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x4, y0, **kwargs).round().to(torch.int64) << 32) + \
            (torch.conv2d(x4, y1, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x4, y2, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x4, y3, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x5, y0, **kwargs).round().to(torch.int64) << 40) + \
            (torch.conv2d(x5, y1, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x5, y2, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x6, y0, **kwargs).round().to(torch.int64) << 48) + \
            (torch.conv2d(x6, y1, **kwargs).round().to(torch.int64) << 56) + \
            (torch.conv2d(x7, y0, **kwargs).round().to(torch.int64) << 56)

        return res_float.cpu().numpy()

    def conv2d(self, a, b, stride, padding, dilation, groups):
        num_mult = b.shape[1] * b.shape[2] * b.shape[3]
        # TODO: clean up
        if groups > 1:
            return self.conv2d_torch_8(a, b, stride, padding, dilation, groups, dtype=torch.float32)
        if num_mult > 255:
            return self.conv2d_torch_4(a, b, stride, padding, dilation, groups, dtype=torch.float32)
        else:
            return self.conv2d_torch_8(a, b, stride, padding, dilation, groups, dtype=torch.float32)

    # def conv2d(self, a, b, stride, padding, dilation, groups):
    #     num_mult = b.shape[1] * b.shape[2] * b.shape[3]
    #
    #     # TODO: clean up
    #     if num_mult < 16:
    #         return self.conv2d_torch_4(a, b, stride, padding, dilation, groups, dtype=torch.float32)
    #     if num_mult < 256:
    #         return self.conv2d_torch_4(a, b, stride, padding, dilation, groups, dtype=torch.float32)
    #
    #     return self.conv2d_torch_8(a, b, stride, padding, dilation, groups, dtype=torch.float32)
    #     #     # 2^8 + 2^8
    #     # if num_mult > 255:
    #     #     return self.conv2d_torch_4(a, b, stride, padding, dilation, groups)
    #     # else:
    #     #     return self.conv2d_torch_8(a, b, stride, padding, dilation, groups)
    #
