from torch.nn import Module
import torch
import torch.nn.functional as F
import numpy as np


class BlockRelu(Module):

    def __init__(self, block_sizes, num_soft_start_steps=0):
        super(BlockRelu, self).__init__()
        self.block_sizes = np.array(block_sizes)
        self.active_block_sizes = np.unique(self.block_sizes, axis=0)

        # self.num_soft_start_steps = num_soft_start_steps
        # self.training_forward_counter = 0

    def forward(self, activation):

        with torch.no_grad():

            regular_relu_channels = np.all(self.block_sizes == [1, 1], axis=1)
            zero_channels = np.all(self.block_sizes == [1, 0], axis=1)
            identity_channels = np.all(self.block_sizes == [0, 1], axis=1)

            relu_map = torch.zeros_like(activation)
            relu_map[:, regular_relu_channels] = activation[:, regular_relu_channels].sign().add_(1).div_(2)
            relu_map[:, identity_channels] = 1
            relu_map[:, zero_channels] = 0

            for block_size in self.active_block_sizes:
                if np.all(block_size == [1, 1]) or np.all(block_size == [0, 1]) or np.all(block_size == [1, 0]):
                    continue

                channels = np.all(self.block_sizes == block_size, axis=1)
                cur_input = activation[:, channels]

                avg_pool = torch.nn.AvgPool2d(
                    kernel_size=tuple(block_size),
                    stride=tuple(block_size), ceil_mode=True)

                cur_relu_map = avg_pool(cur_input).sign_().add_(1).div_(2)
                o = F.interpolate(input=cur_relu_map, scale_factor=tuple(block_size))
                relu_map[:, channels] = o[:, :, :activation.shape[2], :activation.shape[3]]

                torch.cuda.empty_cache()

            # if self.training and (self.training_forward_counter < self.num_soft_start_steps):
            #     alpha = self.training_forward_counter / self.num_soft_start_steps
            #     relu_map = relu_map.mul_(alpha)
            #     tmp = activation.sign().add_(1).div_(2).mul_(1-alpha)
            #     relu_map = relu_map.add_(tmp)
            #     self.training_forward_counter += 1
            #     torch.cuda.empty_cache()

        return relu_map.mul_(activation)
