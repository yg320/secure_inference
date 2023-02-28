from torch.nn import Module
import torch
import torch.nn.functional as F
import numpy as np
from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.const import IS_TORCH_BACKEND, COMPARISON_NUM_BITS_IGNORED, NUM_OF_LSB_TO_IGNORE


# TODO: don't use you own get_data and data handling. use mmseg one
# TODO: don't use your standalone_inference - use mmseg one

# class BlockRelu(Module):
#
#     def __init__(self, block_sizes):
#         super(BlockRelu, self).__init__()
#         self.block_sizes = np.array(block_sizes)
#         self.active_block_sizes = np.unique(self.block_sizes, axis=0)
#
#     def forward(self, activation):
#
#         with torch.no_grad():
#
#             regular_relu_channels = np.all(self.block_sizes == [1, 1], axis=1)
#
#             relu_map = torch.ones_like(activation)
#             relu_map[:, regular_relu_channels] = activation[:, regular_relu_channels].gt_(0) # TODO: is this smart!?
#
#             for block_size in self.active_block_sizes:
#                 if np.all(block_size == [1, 1]) or np.all(block_size == [0, 1]):
#                     continue
#                 if np.all(block_size == [1, 0]):
#                     assert False
#                 channels = np.all(self.block_sizes == block_size, axis=1)
#                 cur_input = activation[:, channels]
#
#                 avg_pool = torch.nn.AvgPool2d(
#                     kernel_size=tuple(block_size),
#                     stride=tuple(block_size), ceil_mode=True)
#
#                 cur_relu_map = avg_pool(cur_input).gt_(0)
#                 o = F.interpolate(input=cur_relu_map, scale_factor=tuple(block_size))
#                 relu_map[:, channels] = o[:, :, :activation.shape[2], :activation.shape[3]]
#
#                 torch.cuda.empty_cache()
#
#         return relu_map.mul_(activation)



class BlockRelu(Module):

    def __init__(self, block_sizes):
        super(BlockRelu, self).__init__()
        self.block_sizes = np.array(block_sizes)
        self.active_block_sizes = np.unique(self.block_sizes, axis=0)

        self.regular_relu_channels = torch.from_numpy(np.all(self.block_sizes == [1, 1], axis=1))
        self.zero_channels = torch.from_numpy(np.all(self.block_sizes == [1, 0], axis=1))
        self.identity_channels = torch.from_numpy(np.all(self.block_sizes == [0, 1], axis=1))

        self.active_block_sizes = [block_size for block_size in self.active_block_sizes if not (np.all(block_size == [1, 1]) or np.all(block_size == [0, 1]) or np.all(block_size == [1, 0]))]
        self.channels = [torch.from_numpy(np.all(self.block_sizes == block_size, axis=1)) for block_size in self.active_block_sizes]
        self.avg_pools = [torch.nn.AvgPool2d(
                    kernel_size=tuple(block_size),
                    stride=tuple(block_size), ceil_mode=True) for block_size in self.active_block_sizes]

        # self.num_soft_start_steps = 150000
        # self.training_forward_counter = 0

    def forward(self, activation):

        with torch.no_grad():

            relu_map = torch.ones_like(activation)
            relu_map[:, self.regular_relu_channels] = activation[:, self.regular_relu_channels].gt_(0)
            relu_map[:, self.zero_channels] = 0

            for block_size, channels, avg_pool in zip(self.active_block_sizes, self.channels, self.avg_pools):

                cur_input = activation[:, channels]

                cur_relu_map = avg_pool(cur_input).gt_(0)
                o = F.interpolate(input=cur_relu_map, scale_factor=tuple(block_size))
                relu_map[:, channels] = o[:, :, :activation.shape[2], :activation.shape[3]]

            # if self.training and (self.training_forward_counter < self.num_soft_start_steps):
            #     alpha = self.training_forward_counter / self.num_soft_start_steps
            #     relu_map = relu_map.mul_(alpha)
            #     tmp = activation.sign().add_(1).div_(2).mul_(1-alpha)
            #     relu_map = relu_map.add_(tmp)
            #     self.training_forward_counter += 1

        return relu_map.mul_(activation)


