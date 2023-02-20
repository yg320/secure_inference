import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcls.models.builder import BACKBONES
from mmcls.models.backbones import ResNet_CIFAR
from mmcls.models.backbones.resnet import Bottleneck, BasicBlock
from mmcv.cnn import build_conv_layer, build_norm_layer

class BottleneckV2(Bottleneck):

    def __init__(self,
                 **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)

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

class BasicBlockV2(BasicBlock):

    def __init__(self,
                 **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)

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
class ResNet_CIFAR_V2(ResNet_CIFAR):

    arch_settings = {
        18: (BasicBlockV2, (2, 2, 2, 2)),
        34: (BasicBlockV2, (3, 4, 6, 3)),
        50: (BottleneckV2, (3, 4, 6, 3)),
        101: (BottleneckV2, (3, 4, 23, 3)),
        152: (BottleneckV2, (3, 8, 36, 3))
    }

    def __init__(self, **kwargs):
        super(ResNet_CIFAR_V2, self).__init__(**kwargs)


@BACKBONES.register_module()
class ResNet_CIFAR_V2_lightweight(ResNet_CIFAR_V2):
    def __init__(self, **kwargs):
        kwargs["stem_channels"] = 32
        kwargs["base_channels"] = 32
        super(ResNet_CIFAR_V2_lightweight, self).__init__(**kwargs)

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
