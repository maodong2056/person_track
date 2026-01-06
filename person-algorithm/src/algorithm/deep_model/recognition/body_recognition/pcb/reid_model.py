import torch
from torch import nn

from src.algorithm.deep_model.arch.backbones import MobileNetV2

class Mobilev2Reidnet(nn.Module):
    def __init__(self):

        super(Mobilev2Reidnet, self).__init__()


        self.part = 2

        self.backbone_net = MobileNetV2(width_mult=1.0,
                                        out_stages=(6,),
                                        last_channel=1280,
                                        activation='ReLU6'
                                        )

        #.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.avgpool = nn.AvgPool2d((6, 6))


    def forward(self, x):
        x = self.backbone_net(x)
        x = x[0]
        y1 = self.avgpool(x)
        y_out = torch.reshape(y1, (y1.size(0), y1.size(1), y1.size(2)))
        pcb_out1 = torch.split(y_out, 1, dim=2)[0]
        pcb_out2 = torch.split(y_out, 1, dim=2)[1]
        feat_out = torch.cat((pcb_out1, pcb_out2), dim=1).reshape(y1.size(0), 2560)

        return feat_out

