"""
Create by Chengqi.Lv
2020/3/20
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import logging
# from einops import rearrange, repeat
import os
# from ..operators.conv import act_dict
BN_MOMENTUM = 0.1
act_dict = {'ReLU': nn.ReLU(inplace=True),
            'ReLU6': nn.ReLU6(inplace=True),
            'LeakyReLU': nn.LeakyReLU(inplace=True),
            'SELU': nn.SELU(inplace=True),
            None: nn.Sequential()
            }
logger = logging.getLogger(__name__)
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,activation='ReLU6'):
        padding = (kernel_size - 1) // 2
        assert activation in act_dict.keys()
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            act_dict[activation]
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,activation='ReLU6'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert activation in act_dict.keys()

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1,activation=activation))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim,activation=activation),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1., out_stages=(1, 2, 4, 6), last_channel=1280,activation='ReLU6'):
        super(MobileNetV2, self).__init__()
        self.width_mult = width_mult
        self.out_stages = out_stages
        input_channel = 32
        self.last_channel = last_channel
        assert activation in act_dict.keys()
        self.activation = activation
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        self.input_channel = int(input_channel * width_mult)
        self.first_layer = ConvBNReLU(3, input_channel, stride=2,activation=self.activation)
        # building inverted residual blocks
        self.stage0 = self.build_mobilenet_stage(stage_num=0)
        self.stage1 = self.build_mobilenet_stage(stage_num=1)
        self.stage2 = self.build_mobilenet_stage(stage_num=2)
        self.stage3 = self.build_mobilenet_stage(stage_num=3)
        self.stage4 = self.build_mobilenet_stage(stage_num=4)
        self.stage5 = self.build_mobilenet_stage(stage_num=5)
        self.stage6 = self.build_mobilenet_stage(stage_num=6)
        # building last several layers

    def build_mobilenet_stage(self, stage_num):
        stage = []
        t, c, n, s = self.interverted_residual_setting[stage_num]
        output_channel = int(c * self.width_mult)
        for i in range(n):
            if i == 0:
                stage.append(InvertedResidual(self.input_channel, output_channel, s, expand_ratio=t,
                                              activation=self.activation))
            else:
                stage.append(InvertedResidual(self.input_channel, output_channel, 1, expand_ratio=t,
                                              activation=self.activation))
            self.input_channel = output_channel
        if stage_num == 6:
            last_layer = ConvBNReLU(self.input_channel, self.last_channel, kernel_size=1,activation=self.activation)  # TODO:到底要不要最后一�?
            stage.append(last_layer)
        stage = nn.Sequential(*stage)

        for m in stage.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return stage

    def forward(self, x):
        x = self.first_layer(x)
        output = []
        for i in range(0, 7):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)

        return tuple(output)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DeconvKp(nn.Module):
    def __init__(self,
                 input_channel,
                 deconv_channels,
                 activation='ReLU'
                 ):
        super(DeconvKp, self).__init__()
        assert isinstance(input_channel, tuple)
        assert len(input_channel) == 1, 'deconv upsample module does not support feature fusion, please use FPN'
        self.inplanes = input_channel[0]
        self.deconv_channels = deconv_channels  # TODO: 增加deconv channel的选项
        self.deconv_with_bias = False
        self.activation = activation
        self.deconv_layers = self._make_deconv_layer(
            3,
            self.deconv_channels,
            [4, 4, 4],
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        kernel, padding, output_padding = \
            self._get_deconv_cfg(num_kernels[0], 0)
        planes = num_filters[0]
        up = nn.ConvTranspose2d(
            in_channels=self.inplanes,
            out_channels=planes,
            kernel_size=kernel,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=self.deconv_with_bias)
        fill_up_weights(up)
        layers.append(up)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(act_dict[self.activation])
        for i in range(1, num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # fc = DCN(self.inplanes, planes,
            #         kernel_size=(3,3), stride=1,
            #         padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #                kernel_size=3, stride=1,
            #                padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
            fill_up_weights(up)

            # layers.append(fc)
            # layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            # layers.append(act_dict[self.activation])
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(act_dict[self.activation])
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input):
        assert len(input) == 1
        out = self.deconv_layers(input[0])
        return tuple([out])

class PoseMobileV2(nn.Module):
    def __init__(self, num_joints=17, channel_per_joint=14, head_input=672,
                 heatmap_size=(192, 256), simdr_split_ratio=1., onnx_cat_output=False,
                 *args, **kwargs):
        super().__init__()
        # extra = cfg.MODEL.EXTRA
        self.backbone = MobileNetV2(out_stages=(6,))
        # self.keypoint_neck = DeconvKp((1280,), [256, 256, 256])
        # self.coord_representation = cfg.MODEL.COORD_REPRESENTATION
        # self.deconv_with_bias = extra.DECONV_WITH_BIAS
        # self.keypoint_head = TopDownKPHead(256, 51, 1)
        self.num_joints = num_joints
        self.channel_per_joint = channel_per_joint
        self.onnx_cat_output = onnx_cat_output
        self.final_layer = nn.Conv2d(
            in_channels=1280,
            out_channels=self.num_joints * self.channel_per_joint,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # head
        self.mlp_head_x = nn.Linear(head_input, int(heatmap_size[0] * simdr_split_ratio))
        self.mlp_head_y = nn.Linear(head_input, int(heatmap_size[1] * simdr_split_ratio))
        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.keypoint_neck(x)
        x = self.final_layer(x[0])
        # x = rearrange(x, 'b (k t) h w -> b k (t h w)',k=self.num_joints,t=self.channel_per_joint)
        x = x.view(x.size(0), self.num_joints, -1)

        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        soft_x = self.softmax(pred_x)
        soft_y = self.softmax(pred_y)
        if self.onnx_cat_output:
            pred = torch.cat([soft_x, soft_y], dim=-1).transpose(1,2)
            return pred
        else:
            return soft_x, soft_y

def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    # block_class, layers = resnet_spec[num_layers]

    model = PoseMobileV2(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    return model
if __name__ == '__main__':
    model = MobileNetV2()
    model.init_weights()
    print(model)
    input = torch.randn(1, 3, 320, 320)
    output = model(input)



