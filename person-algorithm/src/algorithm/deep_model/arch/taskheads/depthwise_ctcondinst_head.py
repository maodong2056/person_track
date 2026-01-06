"""
Create by Chengqi.Lv
2020/8/13
"""
from .ctcondinst_head import CtCondInstHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from econn.models.operators.conv import act_dict
from .centernet_head import fill_fc_weights


class DepthwiseCtCondInstHead(CtCondInstHead):
    def __init__(self,
                 in_channels,
                 num_classes,
                 head_conv,
                 strides,
                 loss_cfg,
                 activation='ReLU'
                 ):
        super(DepthwiseCtCondInstHead, self).__init__((256, 512, 1024, 2048),
                                                      num_classes,
                                                      head_conv,
                                                      strides,
                                                      loss_cfg)
        self.deconv1 = None
        self.connect1 = None
        self.deconv2 = None
        self.connect2 = None
        self.deconv3 = None
        self.connect3 = None

        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            act_dict[activation],
            nn.Conv2d(in_channels, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, num_classes, 1, 1, 0, bias=True),
        )

        fill_fc_weights(self.heatmap)
        self.heatmap[-1].bias.data.fill_(-4.595)

        self.ctl = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            act_dict[activation],
            nn.Conv2d(in_channels, 128, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128,
                      bias=False),
            nn.BatchNorm2d(128),
            act_dict[activation],
            nn.Conv2d(128, 169, 1, 1, 0, bias=True),
        )
        fill_fc_weights(self.ctl)

        self.mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            act_dict[activation],
            nn.Conv2d(in_channels, head_conv, 1, 1, 0, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                      bias=False),
            nn.BatchNorm2d(head_conv),
            act_dict[activation],
            nn.Conv2d(head_conv, 8, 1, 1, 0, bias=True),
        )
        fill_fc_weights(self.mask)

    def forward(self, x):
        x = x[0]
        out = (self.heatmap(x), self.ctl(x), self.mask(x))
        return out