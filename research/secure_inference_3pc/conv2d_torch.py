import torch
import numpy as np
from research.secure_inference_3pc.const import IS_TORCH_BACKEND

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
        self.mask4_x = torch.stack([self.mask0_4, self.mask1_4, self.mask2_4, self.mask3_4, self.mask4_4, self.mask5_4, self.mask6_4, self.mask7_4, self.mask8_4, self.mask9_4, self.mask10_4, self.mask11_4, self.mask12_4, self.mask13_4, self.mask14_4, self.mask15_4]).unsqueeze_(1).unsqueeze_(2).unsqueeze_(3)
        self.mask4_y = torch.stack([self.mask15_4, self.mask14_4, self.mask13_4, self.mask12_4, self.mask11_4, self.mask10_4, self.mask9_4, self.mask8_4, self.mask7_4, self.mask6_4, self.mask5_4, self.mask4_4, self.mask3_4, self.mask2_4, self.mask1_4, self.mask0_4]).unsqueeze_(1).unsqueeze_(2).unsqueeze_(3).unsqueeze_(1)
        self.shift_x = torch.arange(0, 64, 4).unsqueeze_(1).unsqueeze_(2).unsqueeze_(3).to(self.device)
        self.shift_y = torch.arange(60, -4, -4).unsqueeze_(1).unsqueeze_(2).unsqueeze_(3).to(self.device).unsqueeze(1)
        self.mask0_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(0, 8))]).astype(np.int64))[0].to(device)
        self.mask1_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(8, 16))]).astype(np.int64))[0].to(device)
        self.mask2_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(16, 24))]).astype(np.int64))[0].to(device)
        self.mask3_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(24, 32))]).astype(np.int64))[0].to(device)
        self.mask4_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(32, 40))]).astype(np.int64))[0].to(device)
        self.mask5_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(40, 48))]).astype(np.int64))[0].to(device)
        self.mask6_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(48, 56))]).astype(np.int64))[0].to(device)
        self.mask7_8 = torch.from_numpy(np.uint64([sum(2 ** i for i in range(56, 64))]).astype(np.int64))[0].to(device)

    def conv2d_torch_4_type_1(self, a, b, stride, padding, dilation, groups, dtype=torch.float32):
        kwargs = dict(stride=stride, padding=padding, dilation=dilation)
        if not IS_TORCH_BACKEND:
            a = torch.from_numpy(a)
            b = torch.from_numpy(b)
        a = a.to(self.device)
        b = b.to(self.device)

        x = (a & self.mask4_x) >> self.shift_x
        y = (b.unsqueeze(0) & self.mask4_y) >> self.shift_y

        x = x.reshape(1, x.shape[0] * x.shape[1], x.shape[2], x.shape[3]).to(dtype)
        y = y.reshape(y.shape[0] * y.shape[1], y.shape[2], y.shape[3], y.shape[4]).to(dtype)

        res_float = (torch.conv2d(x[:, :-15*a.shape[1]], y[15*b.shape[0]:], **kwargs, groups=1).sum(axis=0, keepdim=True)).to(torch.int64) << 0
        res_float += (torch.conv2d(x, y, **kwargs, groups=16).reshape(16,  res_float.shape[1], res_float.shape[2], res_float.shape[3]).sum(axis=0, keepdim=True)).to(torch.int64) << 60
        res_float += (torch.conv2d(x[:, :-1 * a.shape[1]], y[1* b.shape[0]:], **kwargs, groups=15).reshape(15, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 56
        res_float += (torch.conv2d(x[:, :-2 * a.shape[1]], y[2* b.shape[0]:], **kwargs, groups=14).reshape(14, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 52
        res_float += (torch.conv2d(x[:, :-3 * a.shape[1]], y[3* b.shape[0]:], **kwargs, groups=13).reshape(13, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 48
        res_float += (torch.conv2d(x[:, :-4 * a.shape[1]], y[4* b.shape[0]:], **kwargs, groups=12).reshape(12, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 44
        res_float += (torch.conv2d(x[:, :-5 * a.shape[1]], y[5* b.shape[0]:], **kwargs, groups=11).reshape(11, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 40
        res_float += (torch.conv2d(x[:, :-6 * a.shape[1]], y[6* b.shape[0]:], **kwargs, groups=10).reshape(10, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 36
        res_float += (torch.conv2d(x[:, :-7 * a.shape[1]], y[7* b.shape[0]:], **kwargs, groups=9).reshape(  9, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 32
        res_float += (torch.conv2d(x[:, :-8 * a.shape[1]], y[8* b.shape[0]:], **kwargs, groups=8).reshape(  8, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 28
        res_float += (torch.conv2d(x[:, :-9 * a.shape[1]], y[9* b.shape[0]:], **kwargs, groups=7).reshape(  7, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 24
        res_float += (torch.conv2d(x[:, :-10 * a.shape[1]], y[10*b.shape[0]:], **kwargs, groups=6).reshape(  6, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 20
        res_float += (torch.conv2d(x[:, :-11 * a.shape[1]], y[11*b.shape[0]:], **kwargs, groups=5).reshape(  5, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 16
        res_float += (torch.conv2d(x[:, :-12 * a.shape[1]], y[12*b.shape[0]:], **kwargs, groups=4).reshape(  4, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 12
        res_float += (torch.conv2d(x[:, :-13 * a.shape[1]], y[13*b.shape[0]:], **kwargs, groups=3).reshape(  3, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 8
        res_float += (torch.conv2d(x[:, :-14 * a.shape[1]], y[14*b.shape[0]:], **kwargs, groups=2).reshape(  2, *res_float.shape[1:]).sum(axis=0, keepdim=True)).to(torch.int64) << 4

        return res_float.cpu().numpy()


    def conv2d_torch_4(self, a, b, stride, padding, dilation, groups, dtype=torch.float32):
        
        kwargs = {'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}
        
        if not IS_TORCH_BACKEND:
            a = torch.from_numpy(a)
            b = torch.from_numpy(b)
        a = a.to(self.device)
        b = b.to(self.device)

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

        # res_float = \
        # (torch.conv2d(x0, y1, **kwargs) + torch.conv2d(x1, y0, **kwargs) << 4).round().to(torch.int64) + \
        # (torch.conv2d(x0, y2, **kwargs) + torch.conv2d(x1, y1, **kwargs) + torch.conv2d(x2, y0, **kwargs) << 8).round().to(torch.int64) + \
        # (torch.conv2d(x0, y3, **kwargs) + torch.conv2d(x1, y2, **kwargs) + torch.conv2d(x2, y1, **kwargs) + torch.conv2d(x3, y0, **kwargs) << 12).round().to(torch.int64) + \
        # (torch.conv2d(x0, y4, **kwargs) + torch.conv2d(x1, y3, **kwargs) + torch.conv2d(x2, y2, **kwargs) + torch.conv2d(x3, y1, **kwargs) + torch.conv2d(x4, y0, **kwargs) << 16).round().to(torch.int64) + \
        # (torch.conv2d(x0, y5, **kwargs) + torch.conv2d(x1, y4, **kwargs) + torch.conv2d(x2, y3, **kwargs) + torch.conv2d(x3, y2, **kwargs) + torch.conv2d(x4, y1, **kwargs) + torch.conv2d(x5, y0, **kwargs) << 20).round().to(torch.int64) + \
        # (torch.conv2d(x0, y6, **kwargs) + torch.conv2d(x1, y5, **kwargs) + torch.conv2d(x2, y4, **kwargs) + torch.conv2d(x3, y3, **kwargs) + torch.conv2d(x4, y2, **kwargs) + torch.conv2d(x5, y1, **kwargs) + torch.conv2d(x6, y0, **kwargs) << 24).round().to(torch.int64) + \
        # (torch.conv2d(x0, y7, **kwargs) + torch.conv2d(x1, y6, **kwargs) + torch.conv2d(x2, y5, **kwargs) + torch.conv2d(x3, y4, **kwargs) + torch.conv2d(x4, y3, **kwargs) + torch.conv2d(x5, y2, **kwargs) + torch.conv2d(x6, y1, **kwargs) + torch.conv2d(x7, y0, **kwargs) << 28).round().to(torch.int64) + \
        # (torch.conv2d(x0, y8, **kwargs) + torch.conv2d(x1, y7, **kwargs) + torch.conv2d(x2, y6, **kwargs) + torch.conv2d(x3, y5, **kwargs) + torch.conv2d(x4, y4, **kwargs) + torch.conv2d(x5, y3, **kwargs) + torch.conv2d(x6, y2, **kwargs) + torch.conv2d(x7, y1, **kwargs) + torch.conv2d(x8, y0, **kwargs) << 32).round().to(torch.int64) + \
        # (torch.conv2d(x0, y9, **kwargs) + torch.conv2d(x1, y8, **kwargs) + torch.conv2d(x2, y7, **kwargs) + torch.conv2d(x3, y6, **kwargs) + torch.conv2d(x4, y5, **kwargs) + torch.conv2d(x5, y4, **kwargs) + torch.conv2d(x6, y3, **kwargs) + torch.conv2d(x7, y2, **kwargs) + torch.conv2d(x8, y1, **kwargs) + torch.conv2d(x9, y0, **kwargs) << 36).round().to(torch.int64) + \
        # (torch.conv2d(x0, y10, **kwargs) + torch.conv2d(x1, y9, **kwargs) + torch.conv2d(x2, y8, **kwargs) + torch.conv2d(x3, y7, **kwargs) + torch.conv2d(x4, y6, **kwargs) + torch.conv2d(x5, y5, **kwargs) + torch.conv2d(x6, y4, **kwargs) + torch.conv2d(x7, y3, **kwargs) + torch.conv2d(x8, y2, **kwargs) + torch.conv2d(x9, y1, **kwargs) + torch.conv2d(x10, y0, **kwargs) << 40).round().to(torch.int64) + \
        # (torch.conv2d(x0, y11, **kwargs) + torch.conv2d(x1, y10, **kwargs) + torch.conv2d(x2, y9, **kwargs) + torch.conv2d(x3, y8, **kwargs) + torch.conv2d(x4, y7, **kwargs) + torch.conv2d(x5, y6, **kwargs) + torch.conv2d(x6, y5, **kwargs) + torch.conv2d(x7, y4, **kwargs) + torch.conv2d(x8, y3, **kwargs) + torch.conv2d(x9, y2, **kwargs) + torch.conv2d(x10, y1, **kwargs) + torch.conv2d(x11, y0, **kwargs) << 44).round().to(torch.int64) + \
        # (torch.conv2d(x0, y12, **kwargs) + torch.conv2d(x1, y11, **kwargs) + torch.conv2d(x2, y10, **kwargs) + torch.conv2d(x3, y9, **kwargs) + torch.conv2d(x4, y8, **kwargs) + torch.conv2d(x5, y7, **kwargs) + torch.conv2d(x6, y6, **kwargs) + torch.conv2d(x7, y5, **kwargs) + torch.conv2d(x8, y4, **kwargs) + torch.conv2d(x9, y3, **kwargs) + torch.conv2d(x10, y2, **kwargs) + torch.conv2d(x11, y1, **kwargs) + torch.conv2d(x12, y0, **kwargs) << 48).round().to(torch.int64) + \
        # (torch.conv2d(x0, y13, **kwargs) + torch.conv2d(x1, y12, **kwargs) + torch.conv2d(x2, y11, **kwargs) + torch.conv2d(x3, y10, **kwargs) + torch.conv2d(x4, y9, **kwargs) + torch.conv2d(x5, y8, **kwargs) + torch.conv2d(x6, y7, **kwargs) + torch.conv2d(x7, y6, **kwargs) + torch.conv2d(x8, y5, **kwargs) + torch.conv2d(x9, y4, **kwargs) + torch.conv2d(x10, y3, **kwargs) + torch.conv2d(x11, y2, **kwargs) + torch.conv2d(x12, y1, **kwargs) + torch.conv2d(x13, y0, **kwargs) << 52).round().to(torch.int64) + \
        # (torch.conv2d(x0, y14, **kwargs) + torch.conv2d(x1, y13, **kwargs) + torch.conv2d(x2, y12, **kwargs) + torch.conv2d(x3, y11, **kwargs) + torch.conv2d(x4, y10, **kwargs) + torch.conv2d(x5, y9, **kwargs) + torch.conv2d(x6, y8, **kwargs) + torch.conv2d(x7, y7, **kwargs) + torch.conv2d(x8, y6, **kwargs) + torch.conv2d(x9, y5, **kwargs) + torch.conv2d(x10, y4, **kwargs) + torch.conv2d(x11, y3, **kwargs) + torch.conv2d(x12, y2, **kwargs) + torch.conv2d(x13, y1, **kwargs) + torch.conv2d(x14, y0, **kwargs) << 56).round().to(torch.int64) + \
        # (torch.conv2d(x0, y15, **kwargs) + torch.conv2d(x1, y14, **kwargs) + torch.conv2d(x2, y13, **kwargs) + torch.conv2d(x3, y12, **kwargs) + torch.conv2d(x4, y11, **kwargs) + torch.conv2d(x5, y10, **kwargs) + torch.conv2d(x6, y9, **kwargs) + torch.conv2d(x7, y8, **kwargs) + torch.conv2d(x8, y7, **kwargs) + torch.conv2d(x9, y6, **kwargs) + torch.conv2d(x10, y5, **kwargs) + torch.conv2d(x11, y4, **kwargs) + torch.conv2d(x12, y3, **kwargs) + torch.conv2d(x13, y2, **kwargs) + torch.conv2d(x14, y1, **kwargs) + torch.conv2d(x15, y0, **kwargs) << 60).round().to(torch.int64)
        # return res_float.cpu().numpy()


        res_float = (torch.conv2d(x0, y0, **kwargs).to(torch.int64) << 0)
        res_float += (torch.conv2d(x0, y1, **kwargs).to(torch.int64) << 4)
        res_float += (torch.conv2d(x0, y2, **kwargs).to(torch.int64) << 8)
        res_float += (torch.conv2d(x0, y3, **kwargs).to(torch.int64) << 12)
        res_float += (torch.conv2d(x0, y4, **kwargs).to(torch.int64) << 16)
        res_float += (torch.conv2d(x0, y5, **kwargs).to(torch.int64) << 20)
        res_float += (torch.conv2d(x0, y6, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x0, y7, **kwargs).to(torch.int64) << 28)
        res_float += (torch.conv2d(x0, y8, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x0, y9, **kwargs).to(torch.int64) << 36)
        res_float += (torch.conv2d(x0, y10, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x0, y11, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x0, y12, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x0, y13, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x0, y14, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x0, y15, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x1, y0, **kwargs).to(torch.int64) << 4)
        res_float += (torch.conv2d(x1, y1, **kwargs).to(torch.int64) << 8)
        res_float += (torch.conv2d(x1, y2, **kwargs).to(torch.int64) << 12)
        res_float += (torch.conv2d(x1, y3, **kwargs).to(torch.int64) << 16)
        res_float += (torch.conv2d(x1, y4, **kwargs).to(torch.int64) << 20)
        res_float += (torch.conv2d(x1, y5, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x1, y6, **kwargs).to(torch.int64) << 28)
        res_float += (torch.conv2d(x1, y7, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x1, y8, **kwargs).to(torch.int64) << 36)
        res_float += (torch.conv2d(x1, y9, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x1, y10, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x1, y11, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x1, y12, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x1, y13, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x1, y14, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x2, y0, **kwargs).to(torch.int64) << 8)
        res_float += (torch.conv2d(x2, y1, **kwargs).to(torch.int64) << 12)
        res_float += (torch.conv2d(x2, y2, **kwargs).to(torch.int64) << 16)
        res_float += (torch.conv2d(x2, y3, **kwargs).to(torch.int64) << 20)
        res_float += (torch.conv2d(x2, y4, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x2, y5, **kwargs).to(torch.int64) << 28)
        res_float += (torch.conv2d(x2, y6, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x2, y7, **kwargs).to(torch.int64) << 36)
        res_float += (torch.conv2d(x2, y8, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x2, y9, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x2, y10, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x2, y11, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x2, y12, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x2, y13, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x3, y0, **kwargs).to(torch.int64) << 12)
        res_float += (torch.conv2d(x3, y1, **kwargs).to(torch.int64) << 16)
        res_float += (torch.conv2d(x3, y2, **kwargs).to(torch.int64) << 20)
        res_float += (torch.conv2d(x3, y3, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x3, y4, **kwargs).to(torch.int64) << 28)
        res_float += (torch.conv2d(x3, y5, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x3, y6, **kwargs).to(torch.int64) << 36)
        res_float += (torch.conv2d(x3, y7, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x3, y8, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x3, y9, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x3, y10, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x3, y11, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x3, y12, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x4, y0, **kwargs).to(torch.int64) << 16)
        res_float += (torch.conv2d(x4, y1, **kwargs).to(torch.int64) << 20)
        res_float += (torch.conv2d(x4, y2, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x4, y3, **kwargs).to(torch.int64) << 28)
        res_float += (torch.conv2d(x4, y4, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x4, y5, **kwargs).to(torch.int64) << 36)
        res_float += (torch.conv2d(x4, y6, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x4, y7, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x4, y8, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x4, y9, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x4, y10, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x4, y11, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x5, y0, **kwargs).to(torch.int64) << 20)
        res_float += (torch.conv2d(x5, y1, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x5, y2, **kwargs).to(torch.int64) << 28)
        res_float += (torch.conv2d(x5, y3, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x5, y4, **kwargs).to(torch.int64) << 36)
        res_float += (torch.conv2d(x5, y5, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x5, y6, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x5, y7, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x5, y8, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x5, y9, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x5, y10, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x6, y0, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x6, y1, **kwargs).to(torch.int64) << 28)
        res_float += (torch.conv2d(x6, y2, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x6, y3, **kwargs).to(torch.int64) << 36)
        res_float += (torch.conv2d(x6, y4, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x6, y5, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x6, y6, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x6, y7, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x6, y8, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x6, y9, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x7, y0, **kwargs).to(torch.int64) << 28)
        res_float += (torch.conv2d(x7, y1, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x7, y2, **kwargs).to(torch.int64) << 36)
        res_float += (torch.conv2d(x7, y3, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x7, y4, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x7, y5, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x7, y6, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x7, y7, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x7, y8, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x8, y0, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x8, y1, **kwargs).to(torch.int64) << 36)
        res_float += (torch.conv2d(x8, y2, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x8, y3, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x8, y4, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x8, y5, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x8, y6, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x8, y7, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x9, y0, **kwargs).to(torch.int64) << 36)
        res_float += (torch.conv2d(x9, y1, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x9, y2, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x9, y3, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x9, y4, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x9, y5, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x9, y6, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x10, y0, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x10, y1, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x10, y2, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x10, y3, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x10, y4, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x10, y5, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x11, y0, **kwargs).to(torch.int64) << 44)
        res_float += (torch.conv2d(x11, y1, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x11, y2, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x11, y3, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x11, y4, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x12, y0, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x12, y1, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x12, y2, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x12, y3, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x13, y0, **kwargs).to(torch.int64) << 52)
        res_float += (torch.conv2d(x13, y1, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x13, y2, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x14, y0, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x14, y1, **kwargs).to(torch.int64) << 60)
        res_float += (torch.conv2d(x15, y0, **kwargs).to(torch.int64) << 60)

        return res_float.cpu().numpy()

    def conv2d_torch_8(self, a, b, stride, padding, dilation, groups, dtype):
        kwargs = {'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}

        if not IS_TORCH_BACKEND:
            a = torch.from_numpy(a)
            b = torch.from_numpy(b)
        a = a.to(self.device)
        b = b.to(self.device)

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

        # x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=0)
        # res_float = \
        # (torch.conv2d(x0, y0, **kwargs).round().to(torch.int64) << 0) + \
        # ((torch.conv2d(x0, y1, **kwargs) + torch.conv2d(x1, y0, **kwargs)).round().to(torch.int64) << 8) + \
        # ((torch.conv2d(x0, y2, **kwargs) + torch.conv2d(x1, y1, **kwargs) + torch.conv2d(x2, y0, **kwargs)).round().to(torch.int64) << 16) + \
        # ((torch.conv2d(x0, y3, **kwargs) + torch.conv2d(x1, y2, **kwargs) + torch.conv2d(x2, y1, **kwargs) + torch.conv2d(x3, y0, **kwargs)).round().to(torch.int64) << 24) + \
        # ((torch.conv2d(x0, y4, **kwargs) + torch.conv2d(x1, y3, **kwargs) + torch.conv2d(x2, y2, **kwargs) + torch.conv2d(x3, y1, **kwargs) + torch.conv2d(x4, y0, **kwargs)).round().to(torch.int64) << 32) + \
        # ((torch.conv2d(x0, y5, **kwargs) + torch.conv2d(x1, y4, **kwargs) + torch.conv2d(x2, y3, **kwargs) + torch.conv2d(x3, y2, **kwargs) + torch.conv2d(x4, y1, **kwargs) + torch.conv2d(x5, y0, **kwargs)).round().to(torch.int64) << 40) + \
        # ((torch.conv2d(x0, y6, **kwargs) + torch.conv2d(x1, y5, **kwargs) + torch.conv2d(x2, y4, **kwargs) + torch.conv2d(x3, y3, **kwargs) + torch.conv2d(x4, y2, **kwargs) + torch.conv2d(x5, y1, **kwargs) + torch.conv2d(x6, y0, **kwargs)).round().to(torch.int64) << 48) + \
        # ((torch.conv2d(x0, y7, **kwargs) + torch.conv2d(x1, y6, **kwargs) + torch.conv2d(x2, y5, **kwargs) + torch.conv2d(x3, y4, **kwargs) + torch.conv2d(x4, y3, **kwargs) + torch.conv2d(x5, y2, **kwargs) + torch.conv2d(x6, y1, **kwargs) + torch.conv2d(x7, y0, **kwargs)).round().to(torch.int64) << 56)
        #
        # return None

        res_float =  (torch.conv2d(x0, y0, **kwargs).to(torch.int64) << 0)
        res_float += (torch.conv2d(x0, y1, **kwargs).to(torch.int64) << 8)
        res_float += (torch.conv2d(x0, y2, **kwargs).to(torch.int64) << 16)
        res_float += (torch.conv2d(x0, y3, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x0, y4, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x0, y5, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x0, y6, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x0, y7, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x1, y0, **kwargs).to(torch.int64) << 8)
        res_float += (torch.conv2d(x1, y1, **kwargs).to(torch.int64) << 16)
        res_float += (torch.conv2d(x1, y2, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x1, y3, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x1, y4, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x1, y5, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x1, y6, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x2, y0, **kwargs).to(torch.int64) << 16)
        res_float += (torch.conv2d(x2, y1, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x2, y2, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x2, y3, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x2, y4, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x2, y5, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x3, y0, **kwargs).to(torch.int64) << 24)
        res_float += (torch.conv2d(x3, y1, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x3, y2, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x3, y3, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x3, y4, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x4, y0, **kwargs).to(torch.int64) << 32)
        res_float += (torch.conv2d(x4, y1, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x4, y2, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x4, y3, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x5, y0, **kwargs).to(torch.int64) << 40)
        res_float += (torch.conv2d(x5, y1, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x5, y2, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x6, y0, **kwargs).to(torch.int64) << 48)
        res_float += (torch.conv2d(x6, y1, **kwargs).to(torch.int64) << 56)
        res_float += (torch.conv2d(x7, y0, **kwargs).to(torch.int64) << 56)

        return res_float.cpu().numpy()

    def conv2d(self, a, b, stride, padding, dilation, groups):
        num_mult = b.shape[1] * b.shape[2] * b.shape[3]
        # print(num_mult)
        # print(a.shape, b.shape, stride, padding, dilation, groups)
        # TODO: clean up
        if groups > 1:
            out = self.conv2d_torch_8(a, b, stride, padding, dilation, groups, dtype=torch.float32)
        elif num_mult >= 4096:
            out = self.conv2d_torch_4(a, b, stride, padding, dilation, groups, dtype=torch.float32)
        elif num_mult >= 256:
            out = self.conv2d_torch_4_type_1(a, b, stride, padding, dilation, groups, dtype=torch.float32)
        else:
            out = self.conv2d_torch_8(a, b, stride, padding, dilation, groups, dtype=torch.float32)

        if IS_TORCH_BACKEND:
            return torch.from_numpy(out)
        else:
            return out
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
