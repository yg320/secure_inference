import torch
import torch.nn as nn
import torch
import numpy as np
import pickle
from research.bReLU import BlockRelu

class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W, _ = x.shape
        # print(N, C, H, W, self.block_size)
        x = x.reshape(N, C, H, W, self.block_size[0], self.block_size[1])
        x = x.transpose(0, 1, 2, 4, 3, 5)#.contiguous()
        x = x.reshape(N, C, H * self.block_size[0], W * self.block_size[1])
        return x


class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        # print(N, C, H, W, self.block_size)
        x = x.reshape(N, C, H // self.block_size[0], self.block_size[0], W // self.block_size[1], self.block_size[1])
        x = x.transpose(0, 1, 2, 4, 3, 5)#.contiguous()
        x = x.reshape(N, C, H // self.block_size[0], W // self.block_size[1], self.block_size[0] * self.block_size[1])
        return x


class BlockReLU_V1(nn.Module):

    def __init__(self, block_sizes):
        super(BlockReLU_V1, self).__init__()
        self.block_sizes = np.array(block_sizes)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if 0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def DReLU(self, activation):
        return (activation >= 0).astype(activation.dtype)

    def mult(self, a, b):
        return a*b

    def forward(self, activation):

        activation = activation.detach().numpy()
        reshaped_inputs = []
        mean_tensors = []
        channels = []
        orig_shapes = []

        for block_size in self.active_block_sizes:

            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = activation[:, cur_channels]

            reshaped_input = SpaceToDepth(block_size)(cur_input)
            mean_tensor = np.sum(reshaped_input, axis=-1, keepdims=True)

            channels.append(cur_channels)
            reshaped_inputs.append(reshaped_input)
            orig_shapes.append(mean_tensor.shape)
            mean_tensors.append(mean_tensor.flatten())

        cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
        mean_tensors = np.concatenate(mean_tensors)
        sign_tensors = self.DReLU(mean_tensors)

        relu_map = np.ones_like(activation)
        for i in range(len(self.active_block_sizes)):
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i+1])].reshape(orig_shapes[i])
            relu_map[:, channels[i]] = DepthToSpace(self.active_block_sizes[i])(sign_tensor.repeat(reshaped_inputs[i].shape[-1], axis=-1)).astype(relu_map.dtype)
            # activation[:, channels[i]] = DepthToSpace(self.active_block_sizes[i])(reshaped_inputs[i])
        desired_relu_map_before = relu_map.copy()
        desired_activation_before = activation.copy()
        activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels], activation[:, ~self.is_identity_channels])
        return torch.from_numpy(activation)#, mean_tensors, sign_tensors, desired_relu_map_before, desired_activation_before



if __name__ == "__main__":
    relu_spec_file ="/home/yakir/Data2/assets_v4/distortions/ade_20k_96x96/ResNet18/block_size_spec.pickle"
    block_sizes = pickle.load(open(relu_spec_file, 'rb'))
    data = pickle.load(open("/home/yakir/image.pickle", 'rb')).cpu()

    x = torch.from_numpy(np.random.random(size=(1,32, 192, 192)))
    br_1 = BlockReLU_V1(block_sizes['stem_2'], 'stem_2')
    br_0 = BlockRelu(block_sizes['stem_2'])

    out_0 = br_0(x)
    out_1 = br_1((x*10000).to(torch.int64)).to(torch.float64) / 10000

    print((out_0 - out_1).abs().max())

    print("fds")