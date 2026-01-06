"""
Create by Chengqi.Lv
2020/4/30
"""

from econn.models.architecture.base_one_stage import BaseOneStage


class CtCondInst(BaseOneStage):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg,
                 task_head_cfg
                 ):
        super(CtCondInst, self).__init__(backbone_cfg,
                                         neck_cfg,
                                         task_head_cfg)

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, 'neck') and self.neck is not None:
            x = self.neck(x)
        x = self.task_head(x)
        return x