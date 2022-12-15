import torch
base = 10
precision_fractional = 4
W = torch.load("/home/yakir/tmp/weight.pt")
I = torch.load("/home/yakir/tmp/data.pt")

W_i = (W * 10 ** 4).long()
I_i = (I * 10 ** 4).long()
out_f = torch.conv2d(I, W, stride=2, padding=3)
out_i = torch.conv2d(I_i, W_i, stride=2, padding=3)

out_i / out_f