from torch.nn import Module
import torch
import numpy as np
from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.const import COMPARISON_NUM_BITS_IGNORED, NUM_OF_LSB_TO_IGNORE
from research.secure_inference_3pc.timer import Timer, timer

from numba import njit, int64, int32, prange
@njit((int64[:, :,: , :], int64[:],      int64[:],     int64[:, :], int32[:, :], int64[:], int64[:], int32[:]), parallel=True, nogil=True, cache=True)
def post_brelu_numba_prep(relu_map, sign_tensors, cumsum_shapes, orig_shapes , active_block_sizes, stacked_active_block_sizes_to_channels, offsets, channel_map):

    for i in prange(active_block_sizes.shape[0]):

        block_size = active_block_sizes[i]
        orig_shape = orig_shapes[i]
        offset = offsets[i]
        cur_cumsum = cumsum_shapes[i]

        for channel_index in range(orig_shape[0]):
            cur_channel = stacked_active_block_sizes_to_channels[offset + channel_index]
            cur_index = channel_index * orig_shape[1] * orig_shape[2]
            for patch_i_index in range(orig_shape[1]):
                xxx = patch_i_index * block_size[0]
                cur_cur_index = cur_index + patch_i_index * orig_shape[2]
                for patch_j_index in range(orig_shape[2]):
                    yyy = patch_j_index * block_size[1]

                    sign_value = sign_tensors[cur_cumsum + cur_cur_index + patch_j_index]

                    for patch_i_shift in range(block_size[0]):
                        for patch_j_shift in range(block_size[1]):

                            i_ = xxx + patch_i_shift
                            j_ = yyy + patch_j_shift
                            if i_ < relu_map.shape[2] and j_ < relu_map.shape[3]:
                                relu_map[:, channel_map[cur_channel], i_, j_] = sign_value



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

@timer(name="post_bReLU")
def post_brelu(activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels, stacked_active_block_sizes_to_channels, offsets, is_identity_channels, channel_map):
    shape = np.array([activation.shape[0], sum(~is_identity_channels), activation.shape[2], activation.shape[3]])
    # relu_map = np.ones(shape, dtype=np.int64)
    # for i, block_size in enumerate(active_block_sizes):
    #     orig_shape = (1, active_block_sizes_to_channels[i].shape[0], pad_handlers[i].out_shape[0] // block_size[0],
    #                   pad_handlers[i].out_shape[1] // block_size[1], 1)
    #     sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shape)
    #     tensor = backend.repeat(sign_tensor, block_size[0] * block_size[1])
    #     cur_channels = active_block_sizes_to_channels[i]
    #     relu_map[:, channel_map[cur_channels]] = pad_handlers[i].unpad(DepthToSpace(active_block_sizes[i])(tensor))
    #
    # return relu_map
    orig_shapes = np.array([[active_block_sizes_to_channels[i].shape[0],
                             pad_handlers[i].out_shape[0] // active_block_sizes[i][0],
                             pad_handlers[i].out_shape[1] // active_block_sizes[i][1]] for i in range(len(active_block_sizes))])

    relu_map = np.zeros(shape, dtype=np.int64)
    post_brelu_numba_prep(relu_map, sign_tensors, cumsum_shapes, orig_shapes, active_block_sizes, stacked_active_block_sizes_to_channels, offsets, channel_map)

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

        self.active_block_sizes = np.array([block_size for block_size in np.unique(self.block_sizes, axis=0) if 0 not in block_size])
        self.active_block_sizes_to_channels = [torch.where(torch.Tensor([bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]))[0] for block_size in self.active_block_sizes]
        self.active_block_sizes_to_channels = [x.numpy() for x in self.active_block_sizes_to_channels]

        if len(self.active_block_sizes_to_channels):
            self.stacked_active_block_sizes_to_channels = np.hstack(self.active_block_sizes_to_channels)
        else:
            self.stacked_active_block_sizes_to_channels = np.array([])
        self.offsets = np.array([0] + [len(x) for x in self.active_block_sizes_to_channels]).cumsum()
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])
        self.pad_handler_class = PadHandler

        self.active_channels = np.arange(len(block_sizes))[~self.is_identity_channels]
        self.channel_map = np.zeros(len(block_sizes), dtype=np.int32)
        self.channel_map[self.active_channels] = np.arange(self.active_channels.shape[0])

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

        cumsum_shapes = np.cumsum([0] + [mean_tensor.shape[0] for mean_tensor in mean_tensors])
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
                               self.active_block_sizes_to_channels,
                               self.stacked_active_block_sizes_to_channels,
                               self.offsets,
                               self.channel_map)
