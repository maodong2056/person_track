from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn

#from .efficientdet_4.model import BiFPN, EfficientNet, BiFPN_last, SeparableConvBlock, Swish

from src.algorithm.deep_model.arch.backbones import MobileNetV2
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
    def __init__(self,num_layers ,heads,head_conv=64, width_mult=1.0,out_stages=(1,2,4,6,),last_channel=512,activation='ReLU6',final_kernel=1,):
        super(MobileNetV2banckBone, self).__init__()
        self.backbone_net = MobileNetV2(width_mult,
                               out_stages,
                               last_channel,
                               activation
                               )
        self.backbone_net.init_weights()
        # self.neck = DepthwiseDeconv(input_channel=(1280,),
        #                        activation='ReLU')
        self.neck = DepthwiseShortcutDeconv(in_channels=(24,32,96,512,),
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
        #print(inputs.shape)

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

        body_map_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(hm)
        body_keep = (hm - body_map_max).float() + 1e-9
        body_keep = nn.ReLU()(body_keep)
        body_keep = body_keep * 1e9
        hm = hm * body_keep

        face_map_max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(fhm)
        face_keep = (fhm - face_map_max).float() + 1e-9
        face_keep = nn.ReLU()(face_keep)
        face_keep = face_keep * 1e9
        fhm = fhm * face_keep



        return hm,wh,reg,fhm,fwh,flm,freg,tag


def get_pose_net(num_layers, heads, head_conv=64):
    head_conv = 64
    model = MobileNetV2banckBone(num_layers, heads, final_kernel=1,head_conv=head_conv)
    return model

