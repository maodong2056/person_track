"""
Create by Chengqi.Lv
2020/3/20
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from ..operators.conv import act_dict


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,activation='ReLU6'):
        padding = (kernel_size - 1) // 2
        assert activation in act_dict.keys()
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            act_dict[activation]
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,activation='ReLU6'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert activation in act_dict.keys()

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1,activation=activation))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim,activation=activation),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1., out_stages=(1, 2, 4, 6), last_channel=1280,activation='ReLU6'):
        super(MobileNetV2, self).__init__()
        self.width_mult = width_mult
        self.out_stages = out_stages
        input_channel = 32
        self.last_channel = last_channel
        assert activation in act_dict.keys()
        self.activation = activation
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        self.input_channel = int(input_channel * width_mult)
        self.first_layer = ConvBNReLU(3, input_channel, stride=2,activation=self.activation)
        # building inverted residual blocks
        self.stage0 = self.build_mobilenet_stage(stage_num=0)
        self.stage1 = self.build_mobilenet_stage(stage_num=1)
        self.stage2 = self.build_mobilenet_stage(stage_num=2)
        self.stage3 = self.build_mobilenet_stage(stage_num=3)
        self.stage4 = self.build_mobilenet_stage(stage_num=4)
        self.stage5 = self.build_mobilenet_stage(stage_num=5)
        self.stage6 = self.build_mobilenet_stage(stage_num=6)
        # building last several layers

    def build_mobilenet_stage(self, stage_num):
        stage = []
        t, c, n, s = self.interverted_residual_setting[stage_num]
        output_channel = int(c * self.width_mult)
        for i in range(n):
            if i == 0:
                stage.append(InvertedResidual(self.input_channel, output_channel, s, expand_ratio=t,
                                              activation=self.activation))
            else:
                stage.append(InvertedResidual(self.input_channel, output_channel, 1, expand_ratio=t,
                                              activation=self.activation))
            self.input_channel = output_channel
        if stage_num == 6:
            last_layer = ConvBNReLU(self.input_channel, self.last_channel, kernel_size=1,activation=self.activation)  # TODO:到底要不要最后一�?
            stage.append(last_layer)
        stage = nn.Sequential(*stage)

        for m in stage.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return stage

    def forward(self, x):
        x = self.first_layer(x)
        output = []
        for i in range(0, 7):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)

        return tuple(output)

    def init_weights(self):
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
    model = MobileNetV2()
    model.init_weights()
    print(model)
    input = torch.randn(1, 3, 320, 320)
    output = model(input)



