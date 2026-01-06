"""
Create by Chengqi.Lv
2020/11/24
"""
import time
import torch
import torch.nn as nn
from econn.models.backbones.builder import build_backbone
from econn.models.necks.builder import build_neck
from econn.models.taskheads.builder import build_head
from econn.models.architecture.base_one_stage import BaseOneStage


class Yolo(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 task_head_cfg=None,):
        super(Yolo, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if task_head_cfg is not None:
            self.task_head = build_head(task_head_cfg)
            if hasattr(self.task_head, 'encoder'):
                self.encoder = self.task_head.encoder  # encoder必须指向一个静态方法

    def forward(self, x):
        x = self.backbone(x)
        return x

    def inference(self, meta, resize_keep_ratio):
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            preds = self(meta['img'])
            torch.cuda.synchronize()
            time2 = time.time()
            print('forward time: {:.3f}s'.format((time2 - time1)), end=' | ')
            dets = self.task_head.decode(preds, meta, resize_keep_ratio)
            torch.cuda.synchronize()
            print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
        return dets

    def forward_train(self, gt_meta):
        preds = self(gt_meta['img'])
        loss, loss_states = self.task_head.loss(preds, gt_meta)

        return preds, loss, loss_states
