import torch
import time
from research.secure_inference_3pc.conv2d_torch import Conv2DHandler

image_int = torch.randint(low=-1000000, high=100000000, size=(1, 128, 224, 224), dtype=torch.int64).to("cuda:0")
weight_int = torch.randint(low=-1000000, high=100000000, size=(256, 64, 3, 3), dtype=torch.int64).to("cuda:0")

image_float = torch.rand(size=(1, 128, 224, 224), dtype=torch.float16).to("cuda:1")
weight_float = torch.rand(size=(256, 64, 7, 7), dtype=torch.float16).to("cuda:1")
conv_handler = Conv2DHandler("cuda:0")

t0 = time.time()
out_1 = torch.nn.functional.conv2d(image_float, weight_float, stride=2, padding=5, dilation=1, groups=2)
t1 = time.time()
out_2 = conv_handler.conv2d(image_int, weight_int, stride=2, padding=5, dilation=1, groups=2)
t2 = time.time()

print(t1-t0)
print(t2-t1)