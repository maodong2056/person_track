"""
Create by Chengqi.Lv
2020/12/15
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from econn.models.operators.conv import DepthwiseConvModule
from econn.models.operators.init_weights import normal_init
from .one2one_fcos_head import One2OneFCOSHead
from .anchor.anchor_target import multi_apply


class LiteOne2OneHead(One2OneFCOSHead):

    def __init__(self,
                 input_channel,
                 stacked_convs,
                 num_classes,
                 feat_channels,
                 strides,
                 loss,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 activation='ReLU',
                 use_nms=False
                 ):
        super(LiteOne2OneHead, self).__init__(input_channel,
                                              stacked_convs,
                                              num_classes,
                                              feat_channels,
                                              strides,
                                              loss,
                                              conv_cfg,
                                              norm_cfg,
                                              activation,
                                              use_nms)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
        self.cls_conv = nn.ModuleList([nn.Conv2d(self.feat_channels, self.cls_out_channels, 1) for _ in self.strides])
        self.reg_conv = nn.ModuleList([nn.Conv2d(self.feat_channels, 4, 1) for _ in self.strides])

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            ch_in = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                DepthwiseConvModule(ch_in, self.feat_channels, 3, padding=1, norm_cfg=self.norm_cfg,
                                    activation=self.activation, bias=self.norm_cfg is None))
            reg_convs.append(
                DepthwiseConvModule(ch_in, self.feat_channels, 3, padding=1, norm_cfg=self.norm_cfg,
                                    activation=self.activation, bias=self.norm_cfg is None))
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
            normal_init(self.cls_conv[i], std=0.01, bias=bias_cls)
            normal_init(self.reg_conv[i], std=0.01, bias=0.5)
        print('Finish initialize Lite One2One Head.')

    def forward(self, feats):
        return multi_apply(self.forward_single,
                           feats,
                           self.cls_convs,
                           self.reg_convs,
                           self.cls_conv,
                           self.reg_conv,
                           self.strides)

    def forward_single(self, x, cls_convs, reg_convs, cls_conv, reg_conv, stride):
        cls_feature = x
        box_feature = x
        for cl in cls_convs:
            cls_feature = cl(cls_feature)
        for rg in reg_convs:
            box_feature = rg(box_feature)

        pred_hm = cls_conv(cls_feature)
        pred_dis = F.relu(reg_conv(box_feature))
        # if torch.onnx.is_in_onnx_export(): #好像训练时也会触发？？？很奇怪
        #     return pred_hm, pred_dis
        locations = self.locations(x, stride=stride)[None]
        pred_box = self.apply_ltrb(locations, pred_dis * stride)
        return pred_hm, pred_box
