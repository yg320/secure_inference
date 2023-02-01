from research.secure_inference_3pc.conv2d.conv2d_handler_factory import conv2d_handler_factory
import torch
import time

image_0_int = torch.randint(low=-1000000, high=100000000, size=(1, 256, 56, 56), dtype=torch.int64)
image_1_int = torch.randint(low=-1000000, high=100000000, size=(1, 256, 56, 56), dtype=torch.int64)
weight_0_int = torch.randint(low=-1000000, high=100000000, size=(512, 256, 1, 1), dtype=torch.int64)
weight_1_int = torch.randint(low=-1000000, high=100000000, size=(512, 256, 1, 1), dtype=torch.int64)

padding = (0, 0)
stride = (2, 2)
dilation = (1, 1)

image_int_0_cuda = image_0_int.to("cuda:0")
image_int_1_cuda = image_1_int.to("cuda:0")

weight_int_0_cuda = weight_0_int.to("cuda:0")
weight_int_1_cuda = weight_1_int.to("cuda:0")

cuda_conv2d_handler = conv2d_handler_factory.create("cuda:0")

t0 = time.time()
out_0 = cuda_conv2d_handler.conv2d(image_int_0_cuda, weight_int_0_cuda, image_int_1_cuda, weight_int_1_cuda, stride=stride, padding=padding, dilation=dilation, groups=1)
t1 = time.time()

print("conv2d - Numba", t1-t0)
