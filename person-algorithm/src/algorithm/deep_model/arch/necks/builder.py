"""
Create by Chengqi.Lv
2020/3/23
"""

from .deconv import Deconv
from .depthwise_deconv import DepthwiseDeconv
from .depthwise_shortcut_deconv import DepthwiseShortcutDeconv
from .fpn import FPN
from .pafpn import PAFPN
from .deconv_fpn import DeconvFPN


def build_neck(neck_cfg):
    if neck_cfg.name == 'Deconv':
        neck = Deconv(input_channel=neck_cfg.input_channel,
                      activation=neck_cfg.activation)
    elif neck_cfg.name == 'DepthwiseDeconv':
        neck = DepthwiseDeconv(input_channel=neck_cfg.input_channel,
                               activation=neck_cfg.activation)
    elif neck_cfg.name == 'DepthwiseShortcutDeconv':
        neck = DepthwiseShortcutDeconv(in_channels=neck_cfg.in_channels,
                                       deconv_channels=neck_cfg.deconv_channels,
                                       activation=neck_cfg.activation)
    elif neck_cfg.name == 'FPN':
        neck_cfg.pop('name')
        neck = FPN(**neck_cfg)
        neck.init_weights()
    elif neck_cfg.name == 'PAFPN':
        neck_cfg.pop('name')
        neck = PAFPN(**neck_cfg)
        neck.init_weights()
    elif neck_cfg.name == 'DeconvFPN':
        neck_cfg.pop('name')
        neck = DeconvFPN(**neck_cfg)
        neck.init_weights()
    elif neck_cfg.name is None:
        return None
    else:
        raise NotImplementedError('{} does not support!')
    return neck
