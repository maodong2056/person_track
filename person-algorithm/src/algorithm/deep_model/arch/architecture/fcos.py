"""
Create by Chengqi.Lv
2020/4/15
"""

from econn.models.architecture.base_one_stage import BaseOneStage


class FCOS(BaseOneStage):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg,
                 task_head_cfg, ):
        super(FCOS, self).__init__(backbone_cfg,
                                   neck_cfg,
                                   task_head_cfg)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.task_head(x)
        return x
