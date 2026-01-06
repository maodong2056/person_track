"""
Create by Chengqi.Lv
2020/4/14
"""
from econn.models.architecture.base_one_stage import BaseOneStage


class CtFCOS(BaseOneStage):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg,
                 task_head_cfg, ):
        super(CtFCOS, self).__init__(backbone_cfg,
                                     neck_cfg,
                                     task_head_cfg)
