"""
Create by Chengqi.Lv
2020/3/23
"""
from .resnet import ResNet
from .mobilenetv2 import MobileNetV2
from .shufflenetv2 import ShuffleNetV2
from .ghostnet import GhostNet
from .vovnet import VoVNet
from .mobilenext import MobileNeXt
from .vargnetv2 import VarGNetV2
from ._efficientnet import EfficientNet
from .yolov4_tiny import Yolov4Tiny


def build_backbone(backbone_cfg):
    if backbone_cfg.name == 'ResNet':
        backbone = ResNet(backbone_cfg.depth,
                          backbone_cfg.out_stages,
                          activation=backbone_cfg.activation)
        backbone.init_weights(pretrain=True)
    elif backbone_cfg.name == 'MobileNetV2':
        backbone = MobileNetV2(width_mult=backbone_cfg.width_mult,
                               out_stages=backbone_cfg.out_stages,
                               last_channel=backbone_cfg.last_channel,
                               activation=backbone_cfg.activation
                               )
        backbone.init_weights()
    elif backbone_cfg.name == 'ShuffleNetV2':
        backbone = ShuffleNetV2(model_size=backbone_cfg.model_size,
                                out_stages=backbone_cfg.out_stages,
                                with_last_conv=backbone_cfg.with_last_conv,
                                activation=backbone_cfg.activation
                                )
    elif backbone_cfg.name == 'GhostNet':
        backbone = GhostNet(width_mult=backbone_cfg.width_mult,
                            out_stages=backbone_cfg.out_stages,
                            activation=backbone_cfg.activation
                            )
    elif backbone_cfg.name == 'VoVNet':
        backbone = VoVNet(conv_body=backbone_cfg.conv_body,
                          out_features=backbone_cfg.out_features,
                          )
    elif backbone_cfg.name == 'MobileNeXt':
        backbone = MobileNeXt(width_mult=backbone_cfg.width_mult,
                              out_stages=backbone_cfg.out_stages,
                              identity_tensor_multiplier=backbone_cfg.identity_tensor_multiplier,
                              last_channel=backbone_cfg.last_channel,
                              activation=backbone_cfg.activation
                              )
        backbone.init_weights()
    elif backbone_cfg.name == 'VarGNetV2':
        backbone = VarGNetV2(width_mult=backbone_cfg.width_mult,
                             out_stages=backbone_cfg.out_stages,
                             channel_lists=backbone_cfg.channel_lists,
                             units=backbone_cfg.units,
                             factor=backbone_cfg.factor,
                             group_base=backbone_cfg.group_base
                             )
        backbone.initialize_weights()
    elif backbone_cfg.name == 'EfficientNet':
        backbone = EfficientNet(model_size=backbone_cfg.model_size,
                                load_from=backbone_cfg.load_from,
                                out_stages=backbone_cfg.out_stages)
    elif backbone_cfg.name == 'Yolov4Tiny':
        backbone = Yolov4Tiny(pretrained=None)
    else:
        raise NotImplementedError('{} does not support!'.format(backbone_cfg.name))

    return backbone
