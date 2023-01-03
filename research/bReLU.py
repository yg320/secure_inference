from torch.nn import Module
import torch
import torch.nn.functional as F
import numpy as np

# TODO: don't use you own get_data and data handling. use mmseg one
# TODO: don't use your standalone_inference - use mmseg one

class BlockRelu(Module):

    def __init__(self, block_sizes):
        super(BlockRelu, self).__init__()
        self.block_sizes = np.array(block_sizes)
        self.active_block_sizes = np.unique(self.block_sizes, axis=0)

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

        return relu_map.mul_(activation)

from abc import ABCMeta, abstractmethod


class DepthToSpace(Module, metaclass=ABCMeta):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    @abstractmethod
    def permute(self, x, dims):
        pass

    def forward(self, x):
        N, C, H, W, _ = x.shape
        x = x.reshape(N, C, H, W, self.block_size[0], self.block_size[1])
        x = self.permute(x, (0, 1, 2, 4, 3, 5))
        x = x.reshape(N, C, H * self.block_size[0], W * self.block_size[1])
        return x

class SpaceToDepth(Module, metaclass=ABCMeta):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    @abstractmethod
    def permute(self, x, dims):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        N, C, H, W = x.shape
        x = x.reshape(N, C, H // self.block_size[0], self.block_size[0], W // self.block_size[1], self.block_size[1])
        x = self.permute(x, (0, 1, 2, 4, 3, 5))  #Numpy
        x = x.reshape(N, C, H // self.block_size[0], W // self.block_size[1], self.block_size[0] * self.block_size[1])
        return x


class DepthToSpaceNumpy(DepthToSpace):
    def __init__(self, block_size):
        super().__init__(block_size=block_size)

    def permute(self, x, order):
        return x.transpose(order)


class SpaceToDepthNumpy(SpaceToDepth):
    def __init__(self, block_size):
        super().__init__(block_size=block_size)

    def permute(self, x, order):
        return x.transpose(order)


class DepthToSpaceTorch(DepthToSpace):
    def __init__(self, block_size):
        super().__init__(block_size=block_size)

    def permute(self, x, order):
        return x.permute(order)


class SpaceToDepthTorch(SpaceToDepth):
    def __init__(self, block_size):
        super().__init__(block_size=block_size)

    def permute(self, x, order):
        return x.permute(order)


class PadHandler(metaclass=ABCMeta):

    def __init__(self, x, block_size):
        super().__init__()
        self.pad_x_l = 0
        self.pad_x_r = (block_size[0] - x.shape[2] % block_size[0]) % block_size[0]
        self.pad_y_l = 0
        self.pad_y_r = (block_size[1] - x.shape[3] % block_size[1]) % block_size[1]

    @abstractmethod
    def pad(self, x):
        pass

    def unpad(self, x):
        return x[:, :, self.pad_x_l:x.shape[2] - self.pad_x_r, self.pad_y_l:x.shape[3] - self.pad_y_r]


class TorchPadHandler(PadHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pad(self, x):
        return F.pad(x, (self.pad_y_l, self.pad_y_r, self.pad_x_l, self.pad_x_r), 'constant', 0)


class NumpyPadHandler(PadHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pad(self, x):
        return np.pad(x, ((0, 0), (0, 0), (self.pad_x_l, self.pad_x_r), (self.pad_y_l, self.pad_y_r)), 'constant')


class SecureOptimizedBlockReLU(Module):

    def __init__(self, block_sizes):
        super(SecureOptimizedBlockReLU, self).__init__()
        self.block_sizes = np.array(block_sizes)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if 0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])



    # def mult(self, a, b):
    #     return a * b

    def forward(self, activation):
        # TODO: fuse with next layer
        if np.all(self.block_sizes == [0, 1]):
            return activation
        reshaped_inputs = []
        mean_tensors = []
        channels = []
        orig_shapes = []
        pad_handlers = []

        for block_size in self.active_block_sizes:
            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = activation[:, cur_channels]
            padder = self.pad_handler_class(cur_input, block_size)
            cur_input = padder.pad(cur_input)
            reshaped_input = self.space_to_depth_class(block_size)(cur_input)
            mean_tensor = self.sum(reshaped_input)

            channels.append(cur_channels)
            reshaped_inputs.append(reshaped_input)
            orig_shapes.append(mean_tensor.shape)
            mean_tensors.append(mean_tensor.flatten())
            pad_handlers.append(padder)

        cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
        mean_tensors = self.concat(mean_tensors)

        sign_tensors = self.DReLU(mean_tensors)

        relu_map = self.ones_like(activation)  # TODO: here
        for i in range(len(self.active_block_sizes)):
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shapes[i])
            repeats = reshaped_inputs[i].shape[-1]
            tensor = self.repeat(sign_tensor, repeats)

            relu_map[:, channels[i]] = pad_handlers[i].unpad(self.depth_to_space_class(self.active_block_sizes[i])(tensor))

        activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels], activation[:, ~self.is_identity_channels])
        return activation


class NumpySecureOptimizedBlockReLU(SecureOptimizedBlockReLU):

    def __init__(self, block_sizes):
        super().__init__(block_sizes)
        self.concat = np.concatenate
        self.sum = lambda tensor: np.sum(tensor, axis=-1, keepdims=True)
        self.ones_like = np.ones_like
        self.repeat = lambda tensor, repeats: tensor.repeat(repeats, axis=-1)
        self.depth_to_space_class = DepthToSpaceNumpy
        self.space_to_depth_class = SpaceToDepthNumpy
        self.pad_handler_class = NumpyPadHandler

    def DReLU(self, activation):
        return activation >= 0
        # return activation.sign().add(1).div(2).to(activation.dtype)
class TorchSecureOptimizedBlockReLU(SecureOptimizedBlockReLU):
    def __init__(self, block_sizes):
        super().__init__(block_sizes)
        self.concat = torch.cat
        self.sum = lambda tensor: torch.sum(tensor, dim=-1, keepdim=True)
        self.ones_like = torch.ones_like
        self.repeat = lambda tensor, repeats: tensor.repeat((1, 1, 1, 1, repeats))
        self.depth_to_space_class = DepthToSpaceTorch
        self.space_to_depth_class = SpaceToDepthTorch
        self.pad_handler_class = TorchPadHandler

    def DReLU(self, activation):
        return (activation >= 0).to(activation.dtype)
        # return activation.sign().add(1).div(2).to(activation.dtype)


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
