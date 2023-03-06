from torch.nn import Module
import torch
import numpy as np
from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.const import IS_TORCH_BACKEND, COMPARISON_NUM_BITS_IGNORED, NUM_OF_LSB_TO_IGNORE


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


def post_brelu(activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes,
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
        mean_tensors = (mean_tensors >> NUM_OF_LSB_TO_IGNORE) << COMPARISON_NUM_BITS_IGNORED
        sign_tensors = self.DReLU(mean_tensors)

        return self.post_bReLU(activation,
                               sign_tensors,
                               cumsum_shapes,
                               pad_handlers,
                               self.is_identity_channels,
                               self.active_block_sizes,
                               self.active_block_sizes_to_channels)
