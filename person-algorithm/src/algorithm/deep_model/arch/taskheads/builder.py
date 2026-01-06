"""
Create by Chengqi.Lv
2020/3/23
"""
import copy
import logging
from .centernet_head import CenterNetHead
from .depthwise_centernet_head import DepthwiseCenterNetHead
from .ctfcos_head import CtFCOSHead
from .one2one_center_head import One2OneCenterHead
from .one2one_fcos_head import One2OneFCOSHead
from .lite_one2one_head import LiteOne2OneHead
from .fcos_head import FCOSHead
from .lite_fcos_head import LiteFCOSHead
from .ctcondinst_head import CtCondInstHead
from .depthwise_ctcondinst_head import DepthwiseCtCondInstHead
from .centernet_keypoints_head import CenterNetKeypointsHead
from .embedding_keypoints_head import EmbeddingKeypointHead
from .depthwise_centernet_keypoints_head import DepthwiseCenterNetKeypointsHead
from .gfl_head import GFLHead
from .gfocalv2_head import GFocalV2Head
from .lite_gfl_head import LiteGFLHead
from .vfl_head import VFLHead
from .lite_vfl_head import LiteVFLHead
# from .yolo_head import BaseYOLOHead


def build_head(taskhead_cfg):
    if taskhead_cfg.name == 'CenterNet':
        task_head = CenterNetHead(taskhead_cfg.input_channel,
                                  taskhead_cfg.num_classes,
                                  taskhead_cfg.head_conv,
                                  taskhead_cfg.strides,
                                  taskhead_cfg.loss,
                                  taskhead_cfg.norm_wh,
                                  taskhead_cfg.activation)
    elif taskhead_cfg.name == 'DepthwiseCenterNet':
        task_head = DepthwiseCenterNetHead(taskhead_cfg.input_channel,
                                           taskhead_cfg.num_classes,
                                           taskhead_cfg.head_conv,
                                           taskhead_cfg.strides,
                                           taskhead_cfg.loss,
                                           taskhead_cfg.norm_wh,
                                           taskhead_cfg.activation)
    elif taskhead_cfg.name == 'CenterNet_Keypoints':
        task_head = CenterNetKeypointsHead(taskhead_cfg.input_channel,
                                           taskhead_cfg.num_classes,
                                           taskhead_cfg.head_conv,
                                           taskhead_cfg.strides,
                                           taskhead_cfg.loss,
                                           taskhead_cfg.norm_wh,
                                           taskhead_cfg.activation)
    elif taskhead_cfg.name == 'Embedding_Keypoints':
        task_head = EmbeddingKeypointHead(taskhead_cfg.input_channel,
                                          taskhead_cfg.input_size,
                                          taskhead_cfg.num_classes,
                                          taskhead_cfg.res_block,
                                          # taskhead_cfg.head_conv,
                                          taskhead_cfg.strides,
                                          # taskhead_cfg.loss,
                                          # taskhead_cfg.norm_wh,
                                          taskhead_cfg.activation)
    elif taskhead_cfg.name == 'Depthwise_CenterNet_Keypoints':
        task_head = DepthwiseCenterNetKeypointsHead(taskhead_cfg.input_channel,
                                                    taskhead_cfg.num_classes,
                                                    taskhead_cfg.head_conv,
                                                    taskhead_cfg.strides,
                                                    taskhead_cfg.loss,
                                                    taskhead_cfg.norm_wh,
                                                    taskhead_cfg.activation)
    elif taskhead_cfg.name == 'CtFCOS':
        task_head = CtFCOSHead(taskhead_cfg.input_channel,
                               taskhead_cfg.num_classes,
                               taskhead_cfg.head_conv,
                               taskhead_cfg.strides,
                               taskhead_cfg.loss)
    elif taskhead_cfg.name == 'CtCondInst':
        task_head = CtCondInstHead(taskhead_cfg.in_channels,
                                   taskhead_cfg.num_classes,
                                   taskhead_cfg.head_conv,
                                   taskhead_cfg.strides,
                                   taskhead_cfg.loss)
    elif taskhead_cfg.name == 'Depthwise_CtCondInst':
        task_head = DepthwiseCtCondInstHead(taskhead_cfg.in_channels,
                                            taskhead_cfg.num_classes,
                                            taskhead_cfg.head_conv,
                                            taskhead_cfg.strides,
                                            taskhead_cfg.loss,
                                            taskhead_cfg.activation)
    elif taskhead_cfg.name == 'FCOS':
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = FCOSHead(**cfg)
        task_head.init_weights()
    elif taskhead_cfg.name == 'LiteFCOS':
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = LiteFCOSHead(**cfg)
        task_head.init_weights()
    elif taskhead_cfg.name == 'GFLHead':
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = GFLHead(**cfg)
        task_head.init_weights()
    elif taskhead_cfg.name == 'GFocalV2Head':
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = GFocalV2Head(**cfg)
        task_head.init_weights()
    elif taskhead_cfg.name == 'LiteGFLHead':
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = LiteGFLHead(**cfg)
        task_head.init_weights()
    elif taskhead_cfg.name == 'LiteGFLHeadV2':
        logging.warning("Warning! LiteGFLHeadV2 has been moved into LiteGFLHead! Change your config file.")
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = LiteGFLHead(**cfg)
        task_head.init_weights()
    elif taskhead_cfg.name == 'VFLHead':
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = VFLHead(**cfg)
        task_head.init_weights()
    elif taskhead_cfg.name == 'LiteVFLHead':
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = LiteVFLHead(**cfg)
        task_head.init_weights()
    elif taskhead_cfg.name == 'One2OneCenterHead':
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = One2OneCenterHead(**cfg)
    elif taskhead_cfg.name == 'One2OneFCOSHead':
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = One2OneFCOSHead(**cfg)
    elif taskhead_cfg.name == 'LiteOne2OneHead':
        cfg = copy.deepcopy(taskhead_cfg)
        cfg.pop('name')
        task_head = LiteOne2OneHead(**cfg)
    # elif taskhead_cfg.name == 'YoloHead':
    #     cfg = copy.deepcopy(taskhead_cfg)
    #     cfg.pop('name')
    #     task_head = BaseYOLOHead(**cfg)
    else:  # TODO:移植CTPolarMask
        raise NotImplementedError()

    return task_head
