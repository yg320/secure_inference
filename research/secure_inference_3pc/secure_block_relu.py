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
        x = x.view(N, C, H, W, self.block_size[0], self.block_size[1])
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(N, C, H * self.block_size[0], W * self.block_size[1])
        return x


class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        # print(N, C, H, W, self.block_size)
        x = x.view(N, C, H // self.block_size[0], self.block_size[0], W // self.block_size[1], self.block_size[1])
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(N, C, H // self.block_size[0], W // self.block_size[1], self.block_size[0] * self.block_size[1])
        return x


class BlockReLU_V1(nn.Module):

    def __init__(self, block_sizes, layer_name):
        super(BlockReLU_V1, self).__init__()
        self.block_sizes = np.array(block_sizes)
        self.layer_name = layer_name

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if 0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def forward(self, activation):
        print(self.layer_name)
        regular_relu_channels = [bool(x) for x in np.all(self.block_sizes == [1, 1], axis=1)]

        # if sum(regular_relu_channels):
        #     activation[:, regular_relu_channels] = torch.nn.ReLU()(activation[:, regular_relu_channels])

        reshaped_inputs = []
        mean_tensors = []
        channels = []
        orig_shapes = []

        for block_size in self.active_block_sizes:

            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = activation[:, cur_channels]

            reshaped_input = SpaceToDepth(block_size)(cur_input)
            mean_tensor = torch.mean(reshaped_input, dim=-1, keepdim=True)

            channels.append(cur_channels)
            reshaped_inputs.append(reshaped_input)
            orig_shapes.append(mean_tensor.shape)
            mean_tensors.append(mean_tensor.flatten())

        cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
        mean_tensors = torch.cat(mean_tensors)
        sign_tensors = mean_tensors > 0

        relu_map = torch.ones_like(activation)
        for i in range(len(self.active_block_sizes)):
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i+1])].reshape(orig_shapes[i])
            relu_map[:, channels[i]] = DepthToSpace(self.active_block_sizes[i])(sign_tensor.expand(reshaped_inputs[i].shape)).to(relu_map.dtype)
            activation[:, channels[i]] = DepthToSpace(self.active_block_sizes[i])(reshaped_inputs[i])

        activation[:, self.is_identity_channels] = relu_map[:, self.is_identity_channels] * activation[:, self.is_identity_channels]
        return activation



relu_spec_file ="/home/yakir/Data2/assets_v4/distortions/ade_20k_96x96/ResNet18/block_size_spec.pickle"
block_sizes = pickle.load(open(relu_spec_file, 'rb'))
data = pickle.load(open("/home/yakir/image.pickle", 'rb')).cpu()

x = torch.from_numpy(np.random.random(size=(1,32, 192, 192)))
br_1 = BlockReLU_V1(block_sizes['stem_2'], 'stem_2')
br_0 = BlockRelu(block_sizes['stem_2'])

out_0 = br_0(x)
out_1 = br_1(x)

(out_0 == out_1).all()

print("fds")