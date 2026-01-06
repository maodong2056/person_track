"""
Create by Chengqi.Lv
2020/3/23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from econn.models.operators.conv import act_dict
from .centernet_head import CenterNetHead, fill_fc_weights


class DepthwiseCenterNetHead(CenterNetHead):
    def __init__(self,
                 input_channel,
                 num_classes,
                 head_conv,
                 strides,
                 loss_cfg,
                 norm_wh,
                 activation='ReLU6'
                 ):
        super(DepthwiseCenterNetHead, self).__init__(input_channel,
                                                     num_classes,
                                                     head_conv,
                                                     strides,
                                                     loss_cfg,
                                                     norm_wh,
                                                     activation)

        self.heatmap = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1, groups=input_channel, bias=False),
            nn.BatchNorm2d(input_channel),
            act_dict[activation],
            nn.Conv2d(input_channel, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, num_classes, 1, 1, 0, bias=True),
        )

        fill_fc_weights(self.heatmap)
        self.heatmap[-1].bias.data.fill_(-4.595)

        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1, groups=input_channel, bias=False),
            nn.BatchNorm2d(input_channel),
            act_dict[activation],
            nn.Conv2d(input_channel, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, 2, 1, 1, 0, bias=True),
        )
        fill_fc_weights(self.wh)

        self.offset = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=1, groups=input_channel, bias=False),
            nn.BatchNorm2d(input_channel),
            act_dict[activation],
            nn.Conv2d(input_channel, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, 2, 1, 1, 0, bias=True),
        )
        fill_fc_weights(self.offset)

    # def forward(self, x):
    #     aaa = self.heatmap(x)
    #     out = (aaa, nn.functional.max_pool2d(aaa, kernel_size=3, stride=1, padding=1), self.wh(x), self.offset(x))
    #     return out
