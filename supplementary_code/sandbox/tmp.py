import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

x = np.load("/home/yakir/tiny_tesnet18/distortions/conv_stride_2/distortion_collected/layer1_0_1.npy")
plt.imshow(x)

m = 64
i = 128
o = 256
l = 64
f = 3
l_prime = 16
log_p = 8
target = 0.1

conv2d = (2 * m ** 2 * i + 2 * f ** 2 * o * i + m ** 2 * o) * l
dReLU = m ** 2 * l * (6 * log_p + 14) * o
approx_dReLU = m ** 2 * l_prime * (6 * log_p + 14) * o
pDReLU = m ** 2 * l * (6 * log_p + 14) * target * o
approx_pDReLU = m ** 2 * l_prime * (6 * log_p + 14) * target * o
ReLU = 5 * o * m ** 2  * l
efficient_ReLU = (3*o + 2*target*o) *m ** 2 * l

print(conv2d/1000000/8)
print(dReLU/1000000/8)
print(approx_dReLU/1000000/8)
print((approx_pDReLU)/1000000/8)
print((pDReLU)/1000000/8)
print((approx_pDReLU)/1000000/8)
print(ReLU/1000000/8)
print(efficient_ReLU/1000000/8)


def post(self, activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes,
         active_block_sizes_to_channels):
    relu_map = backend.ones_like(activation)
    for i, block_size in enumerate(active_block_sizes):
        orig_shape = (1, active_block_sizes_to_channels[i].shape[0], pad_handlers[i].out_shape[0] // block_size[0],
                      pad_handlers[i].out_shape[1] // block_size[1], 1)
        sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shape)
        tensor = backend.repeat(sign_tensor, block_size[0] * block_size[1])
        cur_channels = active_block_sizes_to_channels[i]
        relu_map[:, cur_channels] = pad_handlers[i].unpad(DepthToSpace(active_block_sizes[i])(tensor))
    return relu_map
