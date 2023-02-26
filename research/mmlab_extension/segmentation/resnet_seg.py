import torch.nn as nn
# import torch
import torch.utils.checkpoint as cp

from mmseg.models.builder import BACKBONES
from mmseg.models.backbones import ResNetV1c
from mmseg.models.backbones.resnet import Bottleneck, BasicBlock


class MyBottleneck(Bottleneck):

    def __init__(self,
                 *args,
                 **kwargs):
        super(MyBottleneck, self).__init__(*args, **kwargs)

        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.relu_3 = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu_1(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu_1(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu_3(out)

        return out


@BACKBONES.register_module()
class MyResNetSeg(ResNetV1c):

    arch_settings = {
        50: (MyBottleneck, (3, 4, 6, 3)),
        101: (MyBottleneck, (3, 4, 23, 3)),
        152: (MyBottleneck, (3, 8, 36, 3))
    }

    def __init__(self, *args, **kwargs):
        super(MyResNetSeg, self).__init__(*args, **kwargs)


@BACKBONES.register_module()
class AvgPoolResNetSeg(MyResNetSeg):

    def __init__(self, *args, **kwargs):
        super(MyResNetSeg, self).__init__(*args, **kwargs)

    def _make_stem_layer(self, *args, **kwargs):
        ResNetV1c._make_stem_layer(self, *args, **kwargs)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
