from torch.nn import Module
import torch
import torch.nn.functional as F
import numpy as np
from research.secure_inference_3pc.backend import backend
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

from research.secure_inference_3pc.timer import timer
from research.secure_inference_3pc.const import IS_TORCH_BACKEND
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
        mean_tensor = backend.sum(reshaped_input, axis=-1, keepdims=True)
        mean_tensors[index] = mean_tensor.flatten()
        pad_handlers[index] = padder
        return

    @timer("prep")
    def prep_fake(self, activation):

        pad_handlers = [self.pad_handler_class(activation.shape[2:], block_size) for block_size in self.active_block_sizes]
        a = [np.prod(padder.out_shape) * channels.shape[0] // np.prod(block_size) for padder, block_size, channels in zip(pad_handlers, self.active_block_sizes, self.active_block_sizes_to_channels)]
        cumsum_shapes = [0] + list(np.cumsum([x for x in a]))
        mean_tensors = torch.zeros(size=(sum(a), ), dtype=activation.dtype, device=activation.device)
        return mean_tensors, cumsum_shapes, pad_handlers

    @timer("prep")
    def prep(self, activation):

        mean_tensors = [None] * len(self.active_block_sizes)
        pad_handlers = [None] * len(self.active_block_sizes)

        for index, block_size in enumerate(self.active_block_sizes):
            self.get_mean_and_padder(activation, index, mean_tensors, pad_handlers)

        cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
        mean_tensors = backend.concatenate(mean_tensors)
        return mean_tensors, cumsum_shapes, pad_handlers

    @timer("post")
    def post(self, activation, sign_tensors, cumsum_shapes, pad_handlers):
        relu_map = backend.ones_like(activation)
        for i, block_size in enumerate(self.active_block_sizes):
            orig_shape = (1, self.active_block_sizes_to_channels[i].shape[0], pad_handlers[i].out_shape[0]//block_size[0], pad_handlers[i].out_shape[1]//block_size[1], 1)
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shape)
            tensor = backend.repeat(sign_tensor, block_size[0] * block_size[1])
            cur_channels = self.active_block_sizes_to_channels[i]
            relu_map[:, cur_channels] = pad_handlers[i].unpad(DepthToSpace(self.active_block_sizes[i])(tensor))
        return relu_map

    @timer("bReLU")
    def forward(self, activation):

        if np.all(self.block_sizes == [0, 1]):
            return activation
        mean_tensors, cumsum_shapes,  pad_handlers = self.prep(activation)

        sign_tensors = self.DReLU(mean_tensors)

        relu_map = self.post(activation, sign_tensors, cumsum_shapes,  pad_handlers)
        self.final_mult(activation, relu_map)

        return activation

    @timer("final_mult")
    def final_mult(self, activation, relu_map):
        activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels],
                                                              activation[:, ~self.is_identity_channels])
#

# from research.secure_inference_3pc.timer import timer
# class SecureOptimizedBlockReLU(Module):
#
#     def __init__(self, block_sizes):
#         super(SecureOptimizedBlockReLU, self).__init__()
#         self.block_sizes = np.array(block_sizes)
#
#         self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if 0 not in block_size]
#         self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])
#         self.pad_handler_class = PadHandler
#
#     @timer("prep")
#     def prep(self, activation):
#
#         reshaped_inputs = []
#         mean_tensors = []
#         channels = []
#         orig_shapes = []
#         pad_handlers = []
#
#         for block_size in self.active_block_sizes:
#             cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
#             cur_input = activation[:, cur_channels]
#             padder = self.pad_handler_class(cur_input, block_size)
#             cur_input = padder.pad(cur_input)
#             reshaped_input = SpaceToDepth(block_size)(cur_input)
#             mean_tensor = backend.sum(reshaped_input, axis=-1, keepdims=True)
#
#             channels.append(cur_channels)
#             reshaped_inputs.append(reshaped_input)
#             orig_shapes.append(mean_tensor.shape)
#             mean_tensors.append(mean_tensor.flatten())
#             pad_handlers.append(padder)
#
#         cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
#         mean_tensors = backend.concatenate(mean_tensors)
#         return mean_tensors, cumsum_shapes, orig_shapes, reshaped_inputs, channels, pad_handlers
#
#     @timer("post")
#     def post(self, activation, sign_tensors, cumsum_shapes, orig_shapes, reshaped_inputs, channels, pad_handlers):
#         relu_map = backend.ones_like(activation)
#         for i in range(len(self.active_block_sizes)):
#             sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shapes[i])
#             repeats = reshaped_inputs[i].shape[-1]
#             tensor = backend.repeat(sign_tensor, repeats)
#
#             relu_map[:, channels[i]] = pad_handlers[i].unpad(DepthToSpace(self.active_block_sizes[i])(tensor))
#         return relu_map
#
#     @timer("bReLU")
#     def forward(self, activation):
#
#         if np.all(self.block_sizes == [0, 1]):
#             return activation
#         mean_tensors, cumsum_shapes, orig_shapes, reshaped_inputs, channels, pad_handlers = self.prep(activation)
#
#         sign_tensors = self.DReLU(mean_tensors)
#
#         relu_map = self.post(activation, sign_tensors, cumsum_shapes, orig_shapes, reshaped_inputs, channels, pad_handlers)
#         activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels], activation[:, ~self.is_identity_channels])
#         return activation





if __name__ == "__main__":
    import time

    block_sizes = [[3, 4], [1, 1], [0,1], [0, 1], [0,1], [2, 2], [7, 8], [3, 3], [3, 4], [2, 4], [1, 1], [7,8], [7, 8], [11,3], [2, 4], [3, 3],  [3, 3],  [3, 3], [1, 1], [1, 1], [3, 4],[3, 3],[7, 8], [3, 3], [5, 7], [7, 8], [7, 8], [7, 8]]
    image = np.random.normal(size=(1, len(block_sizes), 224, 224))
    image_torch = torch.from_numpy(image)
    relu_0 = BlockRelu(block_sizes)
    relu_numpy = NumpySecureOptimizedBlockReLU(block_sizes)
    relu_torch = TorchSecureOptimizedBlockReLU(block_sizes)

    t0 = time.time()
    out_0 = relu_0(image_torch)
    t1 = time.time()
    out_numpy = relu_numpy(image)
    t2 = time.time()
    out_torch = relu_torch(image_torch)
    t3 = time.time()

    # out_numpy = torch.from_numpy(out_numpy)
    print(torch.all(torch.isclose(out_0, out_torch)))
    print((t3 - t2) / (t2 - t1))
    print('fds')
