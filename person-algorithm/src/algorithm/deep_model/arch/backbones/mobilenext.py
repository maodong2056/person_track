"""
Create by Chengqi.Lv
2020/3/20
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from src.algorithm.deep_model.arch.operators import act_dict


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, activation='ReLU6'):
        padding = (kernel_size - 1) // 2
        assert activation in act_dict.keys()
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            act_dict[activation]
        )


class SandGlass(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, identity_tensor_multiplier=1.0, activation='ReLU6'):
        super(SandGlass, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_identity = False if identity_tensor_multiplier == 1.0 else True
        self.identity_tensor_channels = int(round(inp * identity_tensor_multiplier))

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)

        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        # dw
        layers.append(ConvBNReLU(inp, inp, kernel_size=3, stride=1, groups=inp, activation=activation))
        if expand_ratio != 1:
            # pw-linear
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            ])
        layers.extend([
            # pw
            ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, groups=1, activation=activation),
            # dw-linear
            nn.Conv2d(oup, oup, kernel_size=3, stride=stride, groups=oup, padding=1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.use_identity:
                identity_tensor = x[:, :self.identity_tensor_channels, :, :] + \
                                  out[:, :self.identity_tensor_channels, :, :]
                out = torch.cat([identity_tensor, out[:, self.identity_tensor_channels:, :, :]], dim=1)
                # out[:,:self.identity_tensor_channels,:,:] += x[:,:self.identity_tensor_channels,:,:]
            else:
                out = x + out
            return out
        else:
            return out

class MobileNeXt(nn.Module):
    def __init__(self,
                 width_mult=1.,
                 identity_tensor_multiplier=1.0,
                 out_stages=(1, 2, 4, 6),
                 last_channel=1280,
                 activation='ReLU6'):

        super(MobileNeXt, self).__init__()
        self.width_mult = width_mult
        self.identity_tensor_multiplier = identity_tensor_multiplier
        self.out_stages = out_stages
        input_channel = 64
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        assert activation in act_dict.keys()
        self.activation = activation
        self.sand_glass_setting = [
            # t, c,  b, s
            [2, 96, 1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 1],
            [6, 576, 4, 2],
            [6, 960, 2, 1],
            [6, self.last_channel, 1, 1],
        ]

        # building first layer
        self.input_channel = int(input_channel * width_mult)
        self.first_layer = ConvBNReLU(3, input_channel, stride=2, activation=self.activation)
        # building inverted residual blocks
        self.stage0 = self.build_mobilenext_stage(stage_num=0)
        self.stage1 = self.build_mobilenext_stage(stage_num=1)
        self.stage2 = self.build_mobilenext_stage(stage_num=2)
        self.stage3 = self.build_mobilenext_stage(stage_num=3)
        self.stage4 = self.build_mobilenext_stage(stage_num=4)
        self.stage5 = self.build_mobilenext_stage(stage_num=5)
        self.stage6 = self.build_mobilenext_stage(stage_num=6)
        self.stage7 = self.build_mobilenext_stage(stage_num=7)

    def build_mobilenext_stage(self, stage_num):
        stage = []
        t, c, n, s = self.sand_glass_setting[stage_num]
        output_channel = int(c * self.width_mult)
        for i in range(n):
            if i == 0:
                stage.append(SandGlass(self.input_channel, output_channel, s, expand_ratio=t,
                                       identity_tensor_multiplier=self.identity_tensor_multiplier,
                                       activation=self.activation))
            else:
                stage.append(SandGlass(self.input_channel, output_channel, 1, expand_ratio=t,
                                       identity_tensor_multiplier=self.identity_tensor_multiplier,
                                       activation=self.activation))
            self.input_channel = output_channel
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
        for i in range(0, 8):
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


class MobileNeXtLite(nn.Module):
    def __init__(self,
                 width_mult=1.,
                 identity_tensor_multiplier=1.0,
                 out_stages=(1, 2, 4, 6),
                 last_channel=1280,
                 activation='ReLU6'):

        super(MobileNeXtLite, self).__init__()
        self.width_mult = width_mult
        self.identity_tensor_multiplier = identity_tensor_multiplier
        self.out_stages = out_stages
        input_channel = 32

        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        assert activation in act_dict.keys()
        self.activation = activation
        self.sand_glass_setting = [
            # t, c,  b, s
            [2, 96, 1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 1],
            [6, 576, 4, 2],
            [6, 480, 2, 1],
            [6, self.last_channel, 1, 1],
        ]

        # building first layer
        self.input_channel = int(input_channel * width_mult)
        self.first_layer = ConvBNReLU(3, input_channel, stride=2, activation=self.activation)
        # building inverted residual blocks
        self.stage0 = self.build_mobilenext_stage(stage_num=0)
        self.stage1 = self.build_mobilenext_stage(stage_num=1)
        self.stage2 = self.build_mobilenext_stage(stage_num=2)
        self.stage3 = self.build_mobilenext_stage(stage_num=3)
        self.stage4 = self.build_mobilenext_stage(stage_num=4)
        self.stage5 = self.build_mobilenext_stage(stage_num=5)
        self.stage6 = self.build_mobilenext_stage(stage_num=6)
        self.stage7 = self.build_mobilenext_stage(stage_num=7)

    def build_mobilenext_stage(self, stage_num):
        stage = []
        t, c, n, s = self.sand_glass_setting[stage_num]
        output_channel = int(c * self.width_mult)
        for i in range(n):
            if i == 0:
                stage.append(SandGlass(self.input_channel, output_channel, s, expand_ratio=t,
                                       identity_tensor_multiplier=self.identity_tensor_multiplier,
                                       activation=self.activation))
            else:
                stage.append(SandGlass(self.input_channel, output_channel, 1, expand_ratio=t,
                                       identity_tensor_multiplier=self.identity_tensor_multiplier,
                                       activation=self.activation))
            self.input_channel = output_channel
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
        for i in range(0, 8):
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
    model = MobileNeXt(out_stages=(7,))
    model.init_weights()
    print(model)
    input = torch.randn(1, 3, 320, 320)
    output = model(input)
    print(output[0].shape)

    from thop import profile

    dummy_input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    flops, params = profile(model, inputs=(dummy_input,))
    print('flops:{}\nparams:{}'.format(flops, params))

    # dummy_input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    # torch.onnx.export(model, dummy_input, r'D:\Projects\mobilenext\mobilenext_23333.onnx', verbose=True, opset_version=10, keep_initializers_as_inputs=True)




