"""
Create by Chengqi.Lv
2020/3/24
"""
from .centernet import CenterNet
from .ctfcos import CtFCOS
from .fcos import FCOS
from .gfl import GFL
from .ct_cond_inst import CtCondInst
from .yolo import Yolo
from .multi_task import MultiTask


def build_model(model_cfg):
    if model_cfg.architecture == 'CenterNet':
        model = CenterNet(model_cfg.backbone, model_cfg.neck, model_cfg.task_head)
    elif model_cfg.architecture == 'CtFCOS':
        model = CtFCOS(model_cfg.backbone, model_cfg.neck, model_cfg.task_head)
    elif model_cfg.architecture == 'FCOS':
        model = FCOS(model_cfg.backbone, model_cfg.neck, model_cfg.task_head)
    elif model_cfg.architecture == 'GFL':
        model = GFL(model_cfg.backbone, model_cfg.neck, model_cfg.task_head)
    elif model_cfg.architecture == 'CtCondInst':
        model = CtCondInst(model_cfg.backbone, model_cfg.neck, model_cfg.task_head)
    elif model_cfg.architecture == 'Yolo':
        model = Yolo(backbone_cfg=model_cfg.backbone, task_head_cfg=model_cfg.task_head)

    elif model_cfg.architecture == 'MultiTask':
        model = MultiTask(model_cfg.backbone, model_cfg.neck, model_cfg.task_head)
    else:
        raise NotImplementedError
    return model
