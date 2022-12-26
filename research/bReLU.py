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


class DepthToSpace(Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W, _ = x.shape
        x = x.reshape(N, C, H, W, self.block_size[0], self.block_size[1])
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(N, C, H * self.block_size[0], W * self.block_size[1])
        return x


class SpaceToDepth(Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x = x.reshape(N, C, H // self.block_size[0], self.block_size[0], W // self.block_size[1], self.block_size[1])

        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(N, C, H // self.block_size[0], W // self.block_size[1], self.block_size[0] * self.block_size[1])
        return x


class Pad:

    def __init__(self, x, block_size):
        super().__init__()
        if x.shape[2] % block_size[0] != 0 or x.shape[3] % block_size[1] != 0:
            self.should_pad = True
            self.pad_x_l = 0
            self.pad_x_r = block_size[0] - x.shape[2] % block_size[0]
            self.pad_y_l = 0
            self.pad_y_r = block_size[1] - x.shape[3] % block_size[1]
            self.pad = torch.nn.ZeroPad2d((self.pad_x_l, self.pad_x_r, self.pad_y_l, self.pad_y_r))
        else:
            self.should_pad = False
            self.pad = torch.nn.Identity()
    #
    # def pad(self, x):
    #     if self.should_pad:
    #         pad_values = [self.pad_y_l, self.pad_y_r, self.pad_x_l, self.pad_x_r]
    #         return F.pad(x, pad=pad_values, mode='constant', value=0)
    #     else:
    #         return x

    def unpad(self, x):
        if self.should_pad:
            return x[:, :, self.pad_x_l:x.shape[2] - self.pad_x_r, self.pad_y_l:x.shape[3] - self.pad_y_r]
        else:
            return x

class BlockReLU_V1(Module):

    def __init__(self, block_sizes):
        super(BlockReLU_V1, self).__init__()
        self.block_sizes = np.array(block_sizes)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if
                                   0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def DReLU(self, activation):
        return (activation >= 0).to(activation.dtype)

    def mult(self, a, b):
        return a * b

    def pad(self, x, pad):
        pad_x = pad[0] - x.shape[2] % pad[0]
        pad_y = pad[1] - x.shape[3] % pad[1]

        pad_x_l = pad_x // 2
        pad_x_r = pad_x - pad_x_l
        pad_y_l = pad_y // 2
        pad_y_r = pad_y - pad_y_l

        if pad_x_l > 0 or pad_x_r > 0 or pad_y_l > 0 or pad_y_r > 0:
            x = F.pad(x, pad=(pad_y_l, pad_y_r, pad_x_l, pad_x_r), mode='constant', value=0)
        return x

    def forward(self, activation):
        if type(activation) == torch.Tensor:
            activation = activation.clone()
        else:
            activation = activation.copy()

        reshaped_inputs = []
        mean_tensors = []
        channels = []
        orig_shapes = []
        padders = []

        for block_size in self.active_block_sizes:
            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = activation[:, cur_channels]
            padder = Pad(cur_input, block_size)
            cur_input = padder.pad(cur_input)
            reshaped_input = SpaceToDepth(block_size)(cur_input)
            mean_tensor = torch.sum(reshaped_input, dim=-1, keepdim=True)

            channels.append(cur_channels)
            reshaped_inputs.append(reshaped_input)
            orig_shapes.append(mean_tensor.shape)
            mean_tensors.append(mean_tensor.flatten())
            padders.append(padder)

        cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
        mean_tensors = torch.cat(mean_tensors)

        sign_tensors = self.DReLU(mean_tensors)

        relu_map = torch.ones_like(activation)
        for i in range(len(self.active_block_sizes)):
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shapes[i])
            tensor = sign_tensor.repeat((1, 1, 1, 1, reshaped_inputs[i].shape[-1]))
            relu_map[:, channels[i]] = padders[i].unpad(DepthToSpace(self.active_block_sizes[i])(tensor))

        activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels],
                                                              activation[:, ~self.is_identity_channels])
        return activation


if __name__ == "__main__":
    image = torch.from_numpy(np.random.rand(1, 4, 224, 224))
    import time

    relu_0 = BlockRelu([[1, 1], [2, 2], [2, 4], [3, 3]])
    relu_1 = BlockReLU_V1([[1, 1], [2, 2], [2, 4], [3, 3]])

    t0 = time.time()
    out_0 = relu_0(image)
    t1 = time.time()
    out_1 = relu_1(image)
    t2 = time.time()
    print(torch.all(torch.isclose(out_0, out_1)))
    print(t2 - t1, t1 - t0)
    print('fds')
