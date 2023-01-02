import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcls.models.builder import BACKBONES
from mmcls.models.backbones import ResNet
from mmcls.models.backbones.resnet import Bottleneck, BasicBlock


class MyBottleneck(Bottleneck):

    def __init__(self,
                 *args,
                 **kwargs):
        super(MyBottleneck, self).__init__(*args, **kwargs)

        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.relu_3 = nn.ReLU(inplace=True)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu_1(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu_2(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu_3(out)

        return out

class MyBasicBlock(BasicBlock):

    def __init__(self,
                 *args,
                 **kwargs):
        super(MyBasicBlock, self).__init__(*args, **kwargs)

        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu_1(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu_2(out)

        return out


@BACKBONES.register_module()
class MyResNet(ResNet):

    arch_settings = {
        18: (MyBasicBlock, (2, 2, 2, 2)),
        34: (MyBasicBlock, (3, 4, 6, 3)),
        50: (MyBottleneck, (3, 4, 6, 3)),
        101: (MyBottleneck, (3, 4, 23, 3)),
        152: (MyBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, *args, **kwargs):
        super(MyResNet, self).__init__(*args, **kwargs)

# import torch
#
# class MyAvgPool(nn.Module):
#
#     def __init__(self):
#         super(MyAvgPool, self).__init__()
#
#     def forward(self, x):
#         x = torch.from_numpy(x)
#         x = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)(x)
#         x = x.numpy()
#         return x
#
#
# @BACKBONES.register_module()
# class AvgPoolResNet(MyResNet):
#
#     def __init__(self, *args, **kwargs):
#         super(AvgPoolResNet, self).__init__(*args, **kwargs)
#
#     def _make_stem_layer(self, *args, **kwargs):
#         ResNet._make_stem_layer(self, *args, **kwargs)
#         # self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         self.maxpool = MyAvgPool()