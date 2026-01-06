"""
Create by Chengqi.Lv
2020/3/23
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch.nn as nn
from src.algorithm.deep_model.arch.operators import act_dict


BN_MOMENTUM = 0.1


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DeconvKp(nn.Module):
    def __init__(self,
                 input_channel,
                 deconv_channels,
                 activation='ReLU'
                 ):
        super(DeconvKp, self).__init__()
        assert isinstance(input_channel, tuple)
        assert len(input_channel) == 1, 'deconv upsample module does not support feature fusion, please use FPN'
        self.inplanes = input_channel[0]
        self.deconv_channels = deconv_channels  # TODO: 增加deconv channel的选项
        self.deconv_with_bias = False
        self.activation = activation
        self.deconv_layers = self._make_deconv_layer(
            3,
            self.deconv_channels,
            [4, 4, 4],
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        kernel, padding, output_padding = \
            self._get_deconv_cfg(num_kernels[0], 0)
        planes = num_filters[0]
        up = nn.ConvTranspose2d(
            in_channels=self.inplanes,
            out_channels=planes,
            kernel_size=kernel,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=self.deconv_with_bias)
        fill_up_weights(up)
        layers.append(up)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(act_dict[self.activation])
        for i in range(1, num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # fc = DCN(self.inplanes, planes,
            #         kernel_size=(3,3), stride=1,
            #         padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #                kernel_size=3, stride=1,
            #                padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
            fill_up_weights(up)

            # layers.append(fc)
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            # layers.append(act_dict[self.activation])
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(act_dict[self.activation])
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input):
        assert len(input) == 1
        out = self.deconv_layers(input[0])
        return tuple([out])

class DeconvDWKp(nn.Module):
    def __init__(self,
                 input_channel,
                 deconv_channels,
                 activation='ReLU'
                 ):
        super(DeconvDWKp, self).__init__()
        assert isinstance(input_channel, tuple)
        assert len(input_channel) == 1, 'deconv upsample module does not support feature fusion, please use FPN'
        self.inplanes = input_channel[0]
        self.deconv_channels = deconv_channels  # TODO: 增加deconv channel的选项
        self.deconv_with_bias = False
        self.activation = activation
        self.deconv_layers = self._make_deconv_layer(
            3,
            self.deconv_channels,
            [4, 4, 4],
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        kernel, padding, output_padding = \
            self._get_deconv_cfg(num_kernels[0], 0)
        planes = num_filters[0]
        up_dw = nn.ConvTranspose2d(
            in_channels=self.inplanes,
            out_channels=self.inplanes,
            kernel_size=kernel,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=self.deconv_with_bias,
            groups=self.inplanes)
        up_pw = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=planes,
            kernel_size=1,
            bias=self.deconv_with_bias
            )
        fill_up_weights(up_dw)
        fill_up_weights(up_pw)
        layers.append(up_dw)
        layers.append(up_pw)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(act_dict[self.activation])
        for i in range(1, num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # fc = DCN(self.inplanes, planes,
            #         kernel_size=(3,3), stride=1,
            #         padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #                kernel_size=3, stride=1,
            #                padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up_dw = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias,
                    groups=planes)
            up_pw = nn.Conv2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=1,
                bias=self.deconv_with_bias
            )
            fill_up_weights(up_dw)
            fill_up_weights(up_pw)

            # layers.append(fc)
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            # layers.append(act_dict[self.activation])
            layers.append(up_dw)
            layers.append(up_pw)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(act_dict[self.activation])
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input):
        assert len(input) == 1
        out = self.deconv_layers(input[0])
        return tuple([out])


