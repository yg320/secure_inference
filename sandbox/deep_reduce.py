# Laywe-wise relu pruning

import torch
import torch.nn as nn

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride,
            downsample=None,
            groups=-1,
            groupsize=-1,
            residual=True,
            base_width=64,
            dilation=1,
            norm_layer=None,
            enable_relu=True,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.residual = residual
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.enable_relu = enable_relu
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        if self.enable_relu:
            self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = None
        if self.residual:
            self.downsample = downsample
        self.stride = stride

        if self.enable_relu:
            self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.enable_relu:
            out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual and self.downsample is not None:
            identity = self.downsample(x)

        if self.residual:
            out += identity

        if self.enable_relu:
            out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=-1,
            groupsize=-1,
            residual=True,
            base_width=64,
            dilation=1,
            norm_layer=None,
            enable_relu=True,
    ):
        super(Bottleneck, self).__init__()
        groups = 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.enable_relu = enable_relu
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if self.enable_relu:
            self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        if self.enable_relu:
            self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        if self.enable_relu:
            self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # if not self.enable_relu:
        #    print (".....................................")
        out = self.bn1(out)
        if self.enable_relu:
            out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.enable_relu:
            out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.enable_relu:
            out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            num_classes,
            groups,
            groupsize,
            residual,
            zero_init_residual=False,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
    ):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.groups = groups
        self.groupsize = groupsize
        self.residual = residual

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], enable_relu=True)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], enable_relu=True
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], enable_relu=True
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], enable_relu=True
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, enable_relu=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.groupsize,
                self.residual,
                self.base_width,
                previous_dilation,
                norm_layer,
                enable_relu,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    groups=self.groups,
                    groupsize=self.groupsize,
                    residual=self.residual,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    enable_relu=enable_relu,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(
        arch,
        block,
        layers,
        num_classes,
        groups,
        groupsize,
        residual,
        pretrained,
        progress,
        **kwargs
):
    model = ResNet(block, layers, num_classes, groups, groupsize, residual, **kwargs)
    return model


def resnet18(
        num_classes, groups, groupsize, residual, pretrained=False, progress=True, **kwargs
):
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        num_classes,
        groups,
        groupsize,
        residual,
        pretrained,
        progress,
        **kwargs
    )


def resnet34(
        num_classes, groups, groupsize, residual, pretrained=False, progress=True, **kwargs
):
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        num_classes,
        groups,
        groupsize,
        residual,
        pretrained,
        progress,
        **kwargs
    )


def resnet50(
        num_classes, groups, groupsize, residual, pretrained=False, progress=True, **kwargs
):
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        num_classes,
        groups,
        groupsize,
        residual,
        pretrained,
        progress,
        **kwargs
    )


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )

model = resnet18(groups=0, groupsize=-1, residual=True, num_classes=100)