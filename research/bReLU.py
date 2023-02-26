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


class DepthToSpace(Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size


    def forward(self, x):
        N, C, H, W, _ = x.shape
        x = x.reshape(N, C, H, W, self.block_size[0], self.block_size[1])
        x = backend.permute(x, (0, 1, 2, 4, 3, 5))
        x = x.reshape(N, C, H * self.block_size[0], W * self.block_size[1])
        return x

class SpaceToDepth(Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        N, C, H, W = x.shape
        x = x.reshape(N, C, H // self.block_size[0], self.block_size[0], W // self.block_size[1], self.block_size[1])
        x = backend.permute(x, (0, 1, 2, 4, 3, 5))  #Numpy
        x = x.reshape(N, C, H // self.block_size[0], W // self.block_size[1], self.block_size[0] * self.block_size[1])
        return x



class PadHandler:

    def __init__(self, activation_hw, block_size):
        super().__init__()
        self.pad_x_l = 0
        self.pad_x_r = (block_size[0] - activation_hw[0] % block_size[0]) % block_size[0]
        self.pad_y_l = 0
        self.pad_y_r = (block_size[1] - activation_hw[1] % block_size[1]) % block_size[1]
        self.out_shape = (activation_hw[0] + self.pad_x_r, activation_hw[1] + self.pad_y_r)
    def pad(self, x):
        return backend.pad(x, ((0, 0), (0, 0), (self.pad_x_l, self.pad_x_r), (self.pad_y_l, self.pad_y_r)), 'constant')

    def unpad(self, x):
        return x[:, :, self.pad_x_l:x.shape[2] - self.pad_x_r, self.pad_y_l:x.shape[3] - self.pad_y_r]

class SecureOptimizedBlockReLU(Module):

    def __init__(self, block_sizes):
        super(SecureOptimizedBlockReLU, self).__init__()
        self.block_sizes = np.array(block_sizes)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if 0 not in block_size]
        self.active_block_sizes_to_channels = [torch.where(torch.Tensor([bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]))[0] for block_size in self.active_block_sizes]
        if not IS_TORCH_BACKEND:
            self.active_block_sizes_to_channels = [x.numpy() for x in self.active_block_sizes_to_channels]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])
        self.pad_handler_class = PadHandler

    def get_mean_and_padder(self, activation, index, mean_tensors, pad_handlers):
        cur_channels = self.active_block_sizes_to_channels[index]
        block_size = self.active_block_sizes[index]
        cur_input = activation[:, cur_channels]
        padder = self.pad_handler_class(cur_input.shape[2:], block_size)
        cur_input = padder.pad(cur_input)
        reshaped_input = SpaceToDepth(block_size)(cur_input)
        mean_tensor = backend.sum(reshaped_input, axis=-1, keepdims=True) // reshaped_input.shape[-1]
        mean_tensors[index] = mean_tensor.flatten()
        pad_handlers[index] = padder
        return

    def prep(self, activation):

        mean_tensors = [None] * len(self.active_block_sizes)
        pad_handlers = [None] * len(self.active_block_sizes)

        for index, block_size in enumerate(self.active_block_sizes):
            self.get_mean_and_padder(activation, index, mean_tensors, pad_handlers)

        cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
        mean_tensors = backend.concatenate(mean_tensors)
        return mean_tensors, cumsum_shapes, pad_handlers



    def forward(self, activation):

        if np.all(self.block_sizes == [0, 1]):
            return activation
        mean_tensors, cumsum_shapes,  pad_handlers = self.prep(activation)
        mean_tensors = (mean_tensors >> NUM_OF_LSB_TO_IGNORE).astype(np.int16).astype(np.int64) << COMPARISON_NUM_BITS_IGNORED
        sign_tensors = self.DReLU(mean_tensors)

        return self.post_bReLU(activation,
                               sign_tensors,
                               cumsum_shapes,
                               pad_handlers,
                               self.is_identity_channels,
                               self.active_block_sizes,
                               self.active_block_sizes_to_channels)
