"""
Create by Chengqi.Lv
2020/3/23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from econn.models.operators.conv import act_dict
from .centernet_keypoints_head import CenterNetKeypointsHead, fill_fc_weights


class DepthwiseCenterNetKeypointsHead(CenterNetKeypointsHead):
    def __init__(self,
                 input_channel,
                 num_classes,
                 head_conv,
                 strides,
                 loss_cfg,
                 norm_wh,
                 activation='ReLU6'
                 ):
        super(DepthwiseCenterNetKeypointsHead, self).__init__(input_channel,
                                                              num_classes,
                                                              head_conv,
                                                              strides,
                                                              loss_cfg,
                                                              norm_wh)

        self.heatmap = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            act_dict[activation],
            nn.Conv2d(64, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, num_classes, 1, 1, 0, bias=True),
        )

        fill_fc_weights(self.heatmap)
        self.heatmap[-1].bias.data.fill_(-2.19)

        self.wh = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            act_dict[activation],
            nn.Conv2d(64, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, 2, 1, 1, 0, bias=True),
        )
        fill_fc_weights(self.wh)

        self.offset = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            act_dict[activation],
            nn.Conv2d(64, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, 2, 1, 1, 0, bias=True),
        )
        fill_fc_weights(self.offset)

        self.hp = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            act_dict[activation],
            nn.Conv2d(64, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, 34, 1, 1, 0, bias=True),
        )
        fill_fc_weights(self.hp)

        self.hp_offset = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            act_dict[activation],
            nn.Conv2d(64, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, 2, 1, 1, 0, bias=True),
        )
        fill_fc_weights(self.hp_offset)

        self.hm_hp = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            act_dict[activation],
            nn.Conv2d(64, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, 17, 1, 1, 0, bias=True),
        )
        fill_fc_weights(self.hm_hp)
        self.hm_hp[-1].bias.data.fill_(-2.19)
