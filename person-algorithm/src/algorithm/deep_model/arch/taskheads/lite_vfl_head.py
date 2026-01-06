"""
Create by Chengqi.Lv
2020/12/07
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from econn.models.operators.conv import ConvModule, DepthwiseConvModule
from econn.models.operators.scale import Scale
from econn.models.operators.init_weights import normal_init
from .vfl_head import VFLHead
from .anchor.anchor_target import multi_apply


class LiteVFLHead(VFLHead):
    """
    Lite version of VFL Head
    """

    def __init__(self,
                 num_classes,
                 loss,
                 input_channel,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        super(LiteVFLHead, self).__init__(num_classes,
                                          loss,
                                          input_channel,
                                          stacked_convs,
                                          octave_base_scale,
                                          scales_per_octave,
                                          conv_cfg,
                                          norm_cfg,
                                          **kwargs)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        for _ in self.anchor_strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
        self.cls_conv = nn.ModuleList([nn.Conv2d(self.feat_channels,
                                                self.cls_out_channels,
                                                3,
                                                padding=1) for _ in self.anchor_strides])
        self.reg_conv = nn.ModuleList([nn.Conv2d(self.feat_channels,
                                                4,
                                                3,
                                                padding=1) for _ in self.anchor_strides])

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                DepthwiseConvModule(chn,
                                    self.feat_channels,
                                    3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    bias=self.norm_cfg is None))
            reg_convs.append(
                DepthwiseConvModule(chn,
                                    self.feat_channels,
                                    3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    bias=self.norm_cfg is None))
        return cls_convs, reg_convs

    def init_weights(self):
        for seq in self.cls_convs:
            for m in seq:
                normal_init(m.depthwise, std=0.01)
                normal_init(m.pointwise, std=0.01)
        for seq in self.reg_convs:
            for m in seq:
                normal_init(m.depthwise, std=0.01)
                normal_init(m.pointwise, std=0.01)
        bias_cls = -4.595  # 用0.01的置信度初始化
        for i in range(len(self.anchor_strides)):
            normal_init(self.cls_conv[i], std=0.01, bias=bias_cls)
            normal_init(self.reg_conv[i], std=0.01)
        print('Finish initialize Lite VFL Head.')

    def forward(self, feats):
        return multi_apply(self.forward_single,
                           feats,
                           self.cls_convs,
                           self.reg_convs,
                           self.cls_conv,
                           self.reg_conv,
                           )

    def forward_single(self, x, cls_convs, reg_convs, cls_conv_head, reg_conv_head):
        cls_feat = x
        reg_feat = x
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = cls_conv_head(cls_feat)
        bbox_pred = reg_conv_head(reg_feat)
        return cls_score, bbox_pred


