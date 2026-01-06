from __future__ import division

import torch.nn as nn
import torch
import math


class ConvBnRelu(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 groups=1,
                 use_act=True):
        super(ConvBnRelu, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.use_act = use_act

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.strides,
                              padding=self.padding,
                              groups=self.groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(self.out_channels)
        if self.use_act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.relu(x)

        return x


class VarGNetUnitB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=2,
                 group_base=8,
                 factor=2,
                 **kwargs):
        super(VarGNetUnitB, self).__init__(**kwargs)
        if strides == 2:
            assert out_channels // in_channels == 2

        assert in_channels % group_base == 0
        assert out_channels % group_base == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_increased_channels = self.in_channels * factor
        self.second_increased_channels = self.out_channels * factor
        self.group_base = group_base

        self.group_conv1 = ConvBnRelu(in_channels=self.in_channels,
                                      out_channels=self.first_increased_channels,
                                      kernel_size=3,
                                      strides=strides,
                                      padding=1,
                                      groups=self.in_channels // self.group_base)

        self.pointwise_conv2 = ConvBnRelu(
            in_channels=self.first_increased_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            use_act=False)

        self.group_conv3 = ConvBnRelu(in_channels=self.in_channels,
                                      out_channels=self.first_increased_channels,
                                      kernel_size=3,
                                      strides=strides,
                                      padding=1,
                                      groups=self.in_channels // self.group_base)

        self.pointwise_conv4 = ConvBnRelu(
            in_channels=self.first_increased_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            use_act=False)

        self.group_conv5 = ConvBnRelu(in_channels=self.out_channels,
                                      out_channels=self.second_increased_channels,
                                      kernel_size=3,
                                      strides=1,
                                      padding=1,
                                      groups=self.out_channels // self.group_base)

        self.pointwise_conv6 = ConvBnRelu(
            in_channels=self.second_increased_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            use_act=False)

        self.group_conv7 = ConvBnRelu(in_channels=self.in_channels,
                                      out_channels=self.first_increased_channels,
                                      kernel_size=3,
                                      strides=strides,
                                      padding=1,
                                      groups=self.in_channels // self.group_base)

        self.pointwise_conv8 = ConvBnRelu(
            in_channels=self.first_increased_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            use_act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_cardi1 = self.group_conv1(x)
        x_cardi1 = self.pointwise_conv2(x_cardi1)

        x_cardi2 = self.group_conv3(x)
        x_cardi2 = self.pointwise_conv4(x_cardi2)

        x_cardi = self.relu(x_cardi1 + x_cardi2)

        x_cardi = self.group_conv5(x_cardi)
        x_cardi = self.pointwise_conv6(x_cardi)

        x_out = self.group_conv7(x)
        x_out = self.pointwise_conv8(x_out)

        x_out = self.relu(x_cardi + x_out)

        return x_out


class VarGNetUnitA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=1,
                 group_base=8,
                 factor=2,
                 is_shortcut=False,
                 **kwargs):
        super(VarGNetUnitA, self).__init__()

        assert in_channels == out_channels
        assert in_channels % group_base == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = in_channels * factor
        self.group_base = group_base
        self.is_shortcut = is_shortcut

        self.group_conv1 = ConvBnRelu(in_channels=self.in_channels,
                                      out_channels=self.bottleneck_channels,
                                      kernel_size=3,
                                      strides=strides,
                                      padding=1,
                                      groups=self.in_channels // self.group_base)

        self.pointwise_conv2 = ConvBnRelu(in_channels=self.bottleneck_channels,
                                          out_channels=self.in_channels,
                                          kernel_size=1,
                                          strides=1,
                                          padding=0)

        self.group_conv3 = ConvBnRelu(in_channels=self.in_channels,
                                      out_channels=self.bottleneck_channels,
                                      kernel_size=3,
                                      strides=1,
                                      padding=1,
                                      groups=self.in_channels // self.group_base)

        self.pointwise_conv4 = ConvBnRelu(in_channels=self.bottleneck_channels,
                                          out_channels=self.out_channels,
                                          kernel_size=1,
                                          strides=1,
                                          padding=0,
                                          use_act=False)

        if is_shortcut:
            self.group_conv5 = ConvBnRelu(in_channels=self.in_channels,
                                          out_channels=self.bottleneck_channels,
                                          kernel_size=3,
                                          strides=strides,
                                          padding=1,
                                          groups=self.in_channels // self.group_base)

            self.pointwise_conv6 = ConvBnRelu(in_channels=self.bottleneck_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=1,
                                              strides=1,
                                              padding=0,
                                              use_act=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.group_conv1(x)
        out = self.pointwise_conv2(out)

        out = self.group_conv3(out)
        out = self.pointwise_conv4(out)

        if self.is_shortcut:
            x = self.group_conv5(x)
            x = self.pointwise_conv6(x)

        out = self.relu(x + out)

        return out


class VarGNetV2(nn.Module):
    def __init__(self,
                 channel_lists=[32, 64, 128, 256],
                 units=[3, 7, 4],
                 width_mult=1.0,
                 factor=2,
                 group_base=8,
                 out_stages=[5, 12, 16],
                 **kwargs):
        super(VarGNetV2, self).__init__(**kwargs)

        channel_lists = [int(i * width_mult) for i in channel_lists]
        self.out_stages = out_stages
        self.features = []

        self.features.append(
            nn.Conv2d(in_channels=3,
                      out_channels=channel_lists[0],
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False))
        self.features.append(nn.BatchNorm2d(channel_lists[0]))
        # self.features.append(nn.ReLU(inplace=True))

        self.features.append(
            VarGNetUnitA(in_channels=channel_lists[0],
                         out_channels=channel_lists[0],
                         strides=2,
                         is_shortcut=True,
                         group_base=group_base,
                         factor=1))

        for i in range(len(units)):
            self.features.append(
                VarGNetUnitB(in_channels=channel_lists[i],
                             out_channels=channel_lists[i + 1],
                             strides=2,
                             group_base=group_base,
                             factor=factor))
            for j in range(units[i] - 1):
                self.features.append(
                    VarGNetUnitA(in_channels=channel_lists[i + 1],
                                 out_channels=channel_lists[i + 1],
                                 strides=1,
                                 group_base=group_base,
                                 factor=factor))

        # self.features.append(
        #     nn.Conv2d(in_channels=channel_lists[-1],
        #               out_channels=last_channels,
        #               kernel_size=1,
        #               stride=1,
        #               padding=0,
        #               bias=False))
        # self.features.append(nn.BatchNorm2d(last_channels))
        # self.features.append(nn.ReLU())

        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        outputs = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.out_stages:
                outputs.append(x)
        return tuple(outputs)

    def initialize_weights(self):
        print('init weights...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    model = VarGNetV2(channel_lists=[32, 64, 128, 256],
                      units=[3, 7, 4],
                      width_mult=1.0,
                      factor=2,
                      group_base=8,
                      out_stages=[5, 12, 16],
                      )
    model.initialize_weights()
    # for idx, i in enumerate(model.features):
    #     print('stage', idx)
    #     print(i)
    #     print("-" * 10)
    input = torch.randn(1, 3, 320, 320)
    output = model(input)
    for out in output:
        print(out.shape)
