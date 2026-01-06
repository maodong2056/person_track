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
    for m in up.modules():
        if isinstance(m, nn.ConvTranspose2d):
            w = m.weight.data
            f = math.ceil(w.size(2) / 2)
            c = (2 * f - 1 - f % 2) / (2. * f)
            for i in range(w.size(2)):
                for j in range(w.size(3)):
                    w[0, 0, i, j] = \
                        (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
            for c in range(1, w.size(0)):
                w[c, 0, :, :] = w[0, 0, :, :]
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class DepthwiseShortcutDeconv(nn.Module):
    def __init__(self,
                 in_channels,
                 deconv_channels=None,
                 activation='ReLU6'
                 ):
        super(DepthwiseShortcutDeconv, self).__init__()
        if deconv_channels is None:
            deconv_channels = [256, 128, 64]
        self.inplanes = in_channels[-1]
        self.deconv_channels = deconv_channels
        self.deconv_with_bias = False
        self.activation = activation
        self.deconv1 = self._make_deconv_layer(self.deconv_channels[0], 4)
        self.connect1 = nn.Conv2d(in_channels[2], self.deconv_channels[0], 1, 1, 0, bias=True)
        self.deconv2 = self._make_deconv_layer(self.deconv_channels[1], 4)
        self.connect2 = nn.Conv2d(in_channels[1], self.deconv_channels[1], 1, 1, 0, bias=True)
        self.deconv3 = self._make_deconv_layer(self.deconv_channels[2], 4)
        self.connect3 = nn.Conv2d(in_channels[0], self.deconv_channels[2], 1, 1, 0, bias=True)

    def _get_deconv_cfg(self, deconv_kernel):
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

    def _make_deconv_layer(self, num_filters, num_kernels):
        layers = []
        kernel, padding, output_padding = \
            self._get_deconv_cfg(num_kernels)

        planes = num_filters

        fc = nn.Sequential(
            # dw
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, groups=self.inplanes, bias=False),
            nn.BatchNorm2d(self.inplanes),
            act_dict[self.activation],

            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(planes),
            act_dict[self.activation],

        )
        fill_fc_weights(fc)

        # up = nn.Sequential(
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.BatchNorm2d(planes),
        #     act_dict[self.activation],
        # )
        up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=planes, out_channels=planes, kernel_size=kernel, stride=2, padding=padding,
                               output_padding=output_padding, groups=planes, bias=self.deconv_with_bias),
            nn.BatchNorm2d(planes),
            act_dict[self.activation],
            # nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
        )
        fill_up_weights(up)

        layers.append(fc)
        layers.append(up)

        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        f1, f2, f3, f4 = x
        d1out = self.deconv1(f4) + self.connect1(f3)
        d2out = self.deconv2(d1out) + self.connect2(f2)
        d3out = self.deconv3(d2out) + self.connect3(f1)
        return tuple([d3out])
