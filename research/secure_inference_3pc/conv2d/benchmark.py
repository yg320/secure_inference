import torch
import time
from research.secure_inference_3pc.conv2d.cuda_conv2d import Conv2DHandler as CudaConv2DHandler
from research.secure_inference_3pc.conv2d.numba_conv2d import Conv2DHandler as NumbaConv2DHandler

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

image_int_numpy_0 = image_0_int.numpy()
image_int_numpy_1 = image_1_int.numpy()
weight_int_numpy_0 = weight_0_int.numpy()
weight_int_numpy_1 = weight_1_int.numpy()

cuda_conv2d_handler = CudaConv2DHandler("cuda:0")
numba_conv2d_handler = NumbaConv2DHandler()

t0 = time.time()
out_1 = numba_conv2d_handler.conv2d(image_int_numpy_0, weight_int_numpy_0, image_int_numpy_1, weight_int_numpy_1, stride=stride, padding=padding, dilation=dilation, groups=1)
t1 = time.time()
out_0 = cuda_conv2d_handler.conv2d(image_int_0_cuda, weight_int_0_cuda, image_int_1_cuda, weight_int_1_cuda, stride=stride, padding=padding, dilation=dilation, groups=1)
t2 = time.time()

print("conv2d - Numba", t1-t0)
print("conv2d - CUDA", t2-t1)

# image_float = torch.rand(size=(1, 128, 224, 224), dtype=torch.float16).to("cuda:1")
# weight_float = torch.rand(size=(256, 128, 7, 7), dtype=torch.float16).to("cuda:1")
# conv_handler = Conv2DHandler("cuda:0")

# t0 = time.time()
# out_1 = torch.nn.functional.conv2d(image_float, weight_float, stride=2, padding=5, dilation=1, groups=1)
# t1 = time.time()
# out_2 = conv_handler.conv2d(image_int_cuda, weight_int_cuda, stride=2, padding=5, dilation=1, groups=1)
# t2 = time.time()
# out_3 = torch.nn.functional.conv2d(image_int, weight_int, stride=2, padding=5, dilation=1, groups=1)
# t3 = time.time()
# out_4 = conv_2d(image_int, weight_int, stride=(2, 2), padding=(5, 5), dilation=(1, 1), groups=1)
# t4 = time.time()
# print("conv2d - float32", t1-t0)
# print("conv2d - out approach", t2-t1)
# print("conv2d - CPU torch", t3-t2)
# print("conv2d - Numba", t4-t3)