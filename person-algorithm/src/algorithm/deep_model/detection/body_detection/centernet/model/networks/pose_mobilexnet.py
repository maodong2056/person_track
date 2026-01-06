from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn



#from .econnmodels.backbones.mobilenetv2 import MobileNetV2
# from .econnmodels.backbones.mobilenext import MobileNeXt
# from .econnmodels. necks.depthwise_deconv import DepthwiseDeconv
# from .econnmodels. necks.depthwise_shortcut_deconv import DepthwiseShortcutDeconv
from src.algorithm.deep_model.arch.backbones.mobilenext import MobileNeXt, MobileNeXtLite
from src.algorithm.deep_model.arch.necks import DepthwiseShortcutDeconv
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class MobileNetV2banckBone(nn.Module):
    def __init__(self,num_layers ,heads,head_conv=64,out_stages=(1,2,4,7,)):
        super(MobileNetV2banckBone, self).__init__()
        self.backbone_net = MobileNeXt(width_mult=1.,
                                        identity_tensor_multiplier=1.0,
                                        out_stages=out_stages,
                                        last_channel=1280,
                                         activation='ReLU6'

                               )
        self.backbone_net.init_weights()
        # self.neck = DepthwiseDeconv(input_channel=(1280,),
        #                        activation='ReLU')
        self.neck = DepthwiseShortcutDeconv(in_channels=(144,192,384,1280,),
                                       deconv_channels=(256,128,128,),
                                       activation='ReLU6')

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2d(128, 128,
                    kernel_size=3, padding=1, groups= 128,bias=False),
                  nn.BatchNorm2d(128),
                  nn.ReLU6(inplace=True),
                  nn.Conv2d(128, head_conv, 1, 1, 0, bias=True),
                  nn.ReLU6(inplace=True),
                  nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                            bias=False),
                  nn.BatchNorm2d(head_conv),
                  nn.ReLU6(inplace=True),
                  nn.Conv2d(head_conv, classes,
                    kernel_size=1, stride=1,
                    padding=0, bias=True))

              if 'hm' in head or 'fhm' in head or 'cls' in head:
                fc[-1].bias.data.fill_(-4.595)
                # fc[-1].pointwise_conv.conv.bias.data.fill_(-2.19)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(128, classes,
                  kernel_size=1, stride=1,
                  padding=0,bias=True)
              if 'hm' in head or 'fhm' in head or 'cls' in head:
                fc.bias.data.fill_(-4.95)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):

        p5 = self.backbone_net(inputs)
        features = self.neck(p5)

        hm = self.hm(features[0])
        hm = torch.sigmoid(hm)
        wh = self.wh(features[0])
        reg = self.reg(features[0])

        fhm = self.fhm(features[0])
        fhm = torch.sigmoid(fhm)
        fwh = self.fwh(features[0])
        flm = self.flm(features[0])

        freg = self.freg(features[0])

        tag = self.tag(features[0])

        reid_feat = self.id(features[0])

        body_map_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(hm)
        # body_keep = (hm - body_map_max).float() + 1e-8
        # body_keep = nn.ReLU()(body_keep)
        # body_keep = body_keep * 1e8
        # hm = hm * body_keep

        face_map_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(fhm)
        # face_keep = (fhm - face_map_max).float() + 1e-8
        # face_keep = nn.ReLU()(face_keep)
        # face_keep = face_keep * 1e8
        # fhm = fhm * face_keep

        return hm, wh, reg, fhm, fwh, flm, freg, tag, body_map_max, face_map_max, reid_feat

class MobileNetV2banckBoneLite(nn.Module):
    def __init__(self,num_layers ,heads,head_conv=64,out_stages=(1,2,4,7,)):
        super(MobileNetV2banckBoneLite, self).__init__()
        self.backbone_net = MobileNeXtLite(width_mult=1.,
                                        identity_tensor_multiplier=1.0,
                                        out_stages=out_stages,
                                        last_channel=640,
                                         activation='ReLU6'

                               )
        self.backbone_net.init_weights()
        # self.neck = DepthwiseDeconv(input_channel=(1280,),
        #                        activation='ReLU')
        self.neck = DepthwiseShortcutDeconv(in_channels=(144,192,384,640,),
                                       deconv_channels=(256,128,128,),
                                       activation='ReLU6')
        self.downsample = nn.Sequential(
            nn.Conv2d(128, 128,
                      kernel_size=3, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True))

        fill_fc_weights(self.downsample)


        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                # fc = nn.Sequential(
                #     nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),
                #     nn.BatchNorm2d(128),
                #     nn.ReLU6(inplace=True),
                #     nn.Conv2d(128, head_conv, 1, 1, 0, bias=True),
                #     nn.BatchNorm2d(head_conv),
                #     nn.ReLU6(inplace=True),
                #     nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv, bias=False),
                #     nn.BatchNorm2d(head_conv),
                #     nn.ReLU6(inplace=True),
                #     nn.Conv2d(head_conv, classes,
                #               kernel_size=1, stride=1,
                #               padding=0, bias=True)
                # )
              fc = nn.Sequential(
                  nn.Conv2d(128, 128,
                    kernel_size=3, padding=1, groups= 128,bias=False),
                  nn.BatchNorm2d(128),
                  nn.ReLU6(inplace=True),
                  nn.Conv2d(128, head_conv, 1, 1, 0, bias=True),
                  nn.ReLU6(inplace=True),
                  nn.Conv2d(head_conv, head_conv, kernel_size=3, padding=1, groups=head_conv,
                            bias=False),
                  nn.BatchNorm2d(head_conv),
                  nn.ReLU6(inplace=True),
                  nn.Conv2d(head_conv, classes,
                    kernel_size=1, stride=1,
                    padding=0, bias=True))
            #
            # if 'hm' in head or 'fhm' in head or 'cls' in head:
            #     fc[-1].bias.data.fill_(-4.595)
            #     # fc[-1].pointwise_conv.conv.bias.data.fill_(-2.19)
            #   else:
            #     fill_fc_weights(fc)
            # else:
            #     fc = nn.Conv2d(128, classes,
            #       kernel_size=1, stride=1,
            #       padding=0,bias=True)
            #   if 'hm' in head or 'fhm' in head or 'cls' in head:
            #     fc.bias.data.fill_(-4.95)
            #   else:
            #     fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):

        p5 = self.backbone_net(inputs)
        features = self.neck(p5)
        features = self.downsample(features[0])

        # features = tuple([self.downsample(features[0])])

        # hm = self.hm(features)
        # hm = torch.sigmoid(hm)
        # tblr = self.tblr(features)
        #
        # flm = self.flm(features)
        #
        # tag = self.tag(features)
        #
        # reid_feat = self.id(features)
        # map_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(hm)
        #
        # bhm = torch.split(hm, 1, dim=1)[0]
        #
        # # hm = hm[:, 0:1, ...]
        # fhm = torch.split(hm, 1, dim=1)[1]
        #
        # h_tblr = torch.split(tblr, 4, dim=1)[0]
        # f_tblr = torch.split(tblr, 4, dim=1)[1]
        #
        # body_map_max = torch.split(map_max, 1, dim=1)[0]
        # face_map_max = torch.split(map_max, 1, dim=1)[1]
        #
        # bhm = bhm.view(1, 1, 48, 64)
        #
        # h_tblr = h_tblr.view(1, 4, 48, 64)
        # fhm = fhm.view(1, 1, 48, 64)
        #
        # flm = flm.view(1, 10, 48, 64)
        # f_tblr = f_tblr.view(1, 4, 48, 64)
        # tag = tag.view(1, 2, 48, 64)
        # body_map_max = body_map_max.view(1, 1, 48, 64)
        # face_map_max = face_map_max.view(1, 1, 48, 64)
        # reid_feat = reid_feat.view(1, 128, 48, 64)
        #
        #
        # return bhm, h_tblr,fhm, f_tblr,flm, tag, body_map_max,face_map_max, reid_feat
        #
        hm = self.hm(features)
        hm = torch.sigmoid(hm)
        wh = self.wh(features)
        reg = self.reg(features)

        fhm = self.fhm(features)
        fhm = torch.sigmoid(fhm)
        fwh = self.fwh(features)
        flm = self.flm(features)

        freg = self.freg(features)

        tag = self.tag(features)

        reid_feat = self.id(features)

        body_map_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(hm)
        # body_keep = (hm - body_map_max).float() + 1e-8
        # body_keep = nn.ReLU()(body_keep)
        # body_keep = body_keep * 1e8
        # hm = hm * body_keep

        face_map_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(fhm)
        # face_keep = (fhm - face_map_max).float() + 1e-8
        # face_keep = nn.ReLU()(face_keep)
        # face_keep = face_keep * 1e8
        # fhm = fhm * face_keep

        return hm, wh, reg, fhm, fwh, flm, freg, tag, body_map_max, face_map_max, reid_feat




class MobileNetV2banckBoneWhole(MobileNetV2banckBone):
    def __init__(self, num_layers, heads, head_conv=64, out_stages=(1, 2, 4, 7,)):
        super(MobileNetV2banckBoneWhole, self).__init__(num_layers, heads, head_conv=64, out_stages=(1, 2, 4, 7,))

    def forward(self, inputs):
        p5 = self.backbone_net(inputs)
        features = self.neck(p5)

        hm = self.hm(features[0])
        hm = torch.sigmoid(hm)
        wh = self.wh(features[0])
        reg = self.reg(features[0])

        flm = self.flm(features[0])
        lhlm = self.lhlm(features[0])
        rhlm = self.rhlm(features[0])

        tag = self.tag(features[0])

        reid_feat = self.id(features[0])

        map_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(hm)

        return hm, wh, reg, flm, lhlm, rhlm, tag, map_max, reid_feat


def get_pose_net(num_layers, heads, head_conv=64, state=0):
    head_conv = 64
    if state == 0:
        model = MobileNetV2banckBone(num_layers, heads, head_conv=head_conv)
    elif state == 1:
        model = MobileNetV2banckBoneLite(num_layers, heads, head_conv=head_conv)
    elif state == 2:
        model = MobileNetV2banckBoneWhole(num_layers, heads, head_conv=head_conv)
    return model

