import torch
import torch.nn as nn
from torch.nn import functional as F

from src.algorithm.deep_model.arch.backbones import MobileNetV2


class RootNet(nn.Module):

    def __init__(self):
        self.inplanes = 1280
        self.outplanes = 256

        super(RootNet, self).__init__()
        self.deconv_layers = self._make_deconv_layer(3)
        self.xy_layer = nn.Conv2d(
            in_channels=self.outplanes,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.depth_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        inplanes = self.inplanes
        outplanes = self.outplanes
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(outplanes))
            layers.append(nn.ReLU(inplace=True))
            inplanes = outplanes

        return nn.Sequential(*layers)

    def forward(self, x, k_value):
        # x,y
        xy = self.deconv_layers(x)
        xy = self.xy_layer(xy)
        xy = xy.view(-1, 1, 64 * 64)
        xy = F.softmax(xy, 2)
        xy = xy.view(-1, 1, 64, 64)

        hm_x = xy.sum(dim=(2))
        hm_y = xy.sum(dim=(3))

        coord_x = hm_x * torch.arange(64).float().cuda()
        coord_y = hm_y * torch.arange(64).float().cuda()

        coord_x = coord_x.sum(dim=2)
        coord_y = coord_y.sum(dim=2)

        # z
        img_feat = torch.mean(x.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)  # global average pooling
        img_feat = torch.unsqueeze(img_feat, 2)
        img_feat = torch.unsqueeze(img_feat, 3)
        gamma = self.depth_layer(img_feat)
        gamma = gamma.view(-1, 1)
        depth = gamma * k_value.view(-1, 1)

        coord = torch.cat((coord_x, coord_y, depth), dim=1)
        return coord

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.xy_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.depth_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


class ResPoseNet(nn.Module):
    def __init__(self):
        super(ResPoseNet, self).__init__()
        self.backbone = MobileNetV2(width_mult=1.0,
                                    out_stages=(6,),
                                    last_channel=1280,
                                    activation='ReLU6'
                                    )
        self.backbone.init_weights()
        self.root = RootNet()
        self.root.init_weights()

    def forward(self, input_img, k_value):
        fm = self.backbone(input_img)[0]
        coord = self.root(fm, k_value)

        return coord



def get_pose_net():
    model = ResPoseNet()
    return model


