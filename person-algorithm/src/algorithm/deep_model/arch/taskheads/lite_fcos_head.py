"""
Create by Chengqi.Lv
2020/4/15
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import math
from econn.models.operators.conv import ConvModule, DepthwiseConvModule
from econn.models.operators.scale import Scale
from econn.models.operators.init_weights import normal_init
from .fcos_head import FCOSHead

INF = 1e8


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class LiteFCOSHead(FCOSHead):
    def __init__(self,
                 num_classes,
                 loss,
                 input_channel,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(LiteFCOSHead, self).__init__(num_classes,
                                           loss,
                                           input_channel,
                                           feat_channels,
                                           stacked_convs,
                                           strides,
                                           regress_ranges,
                                           center_sampling,
                                           center_sample_radius,
                                           conv_cfg,
                                           norm_cfg)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
        self.fcos_cls = nn.ModuleList([nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1) for _ in self.strides])
        self.fcos_reg = nn.ModuleList([nn.Conv2d(self.feat_channels, 4, 3, padding=1) for _ in self.strides])
        self.fcos_centerness = nn.ModuleList([nn.Conv2d(self.feat_channels, 1, 3, padding=1) for _ in self.strides])

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
        for i in range(len(self.strides)):
            normal_init(self.fcos_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.fcos_reg[i], std=0.01)
            normal_init(self.fcos_centerness[i], std=0.01)
        print('Finish initialize FCOS Head.')

    def forward(self, feats):
        return multi_apply(self.forward_single,
                           feats,
                           self.cls_convs,
                           self.reg_convs,
                           self.fcos_cls,
                           self.fcos_reg,
                           self.fcos_centerness)

    def forward_single(self, x, cls_conv, reg_conv, cls_head, reg_head, centerness_head):
        cls_feat = x
        reg_feat = x

        for cls_layer in cls_conv:
            cls_feat = cls_layer(cls_feat)
        cls_score = cls_head(cls_feat)

        for reg_layer in reg_conv:
            reg_feat = reg_layer(reg_feat)

        centerness = centerness_head(reg_feat)
        bbox_pred = F.relu(reg_head(reg_feat))
        return cls_score, bbox_pred, centerness
