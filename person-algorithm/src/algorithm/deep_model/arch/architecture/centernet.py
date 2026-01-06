"""
Create by Chengqi.Lv
2020/3/24
"""
from src.algorithm.deep_model.arch import BaseOneStage


class CenterNet(BaseOneStage):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg,
                 task_head_cfg, ):
        super(CenterNet, self).__init__(backbone_cfg,
                                        neck_cfg,
                                        task_head_cfg)
