"""
Create by Chengqi.Lv
2020/12/22
"""

import time
import torch
import torch.nn as nn
from econn.models.backbones.builder import build_backbone
from econn.models.necks.builder import build_neck
from econn.models.taskheads.builder import build_head
from econn.utils.scatter_gather import split_out


class MultiTask(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg=None,
                 task_head_cfg=None, ):
        super(MultiTask, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        self.encoder = {}
        self.neck = build_neck(neck_cfg)
        task_heads = {}
        for task in task_head_cfg:
            task_head = build_head(task_head_cfg[task])
            if hasattr(task_head, 'encoder'):
                self.encoder[task] = task_head.encoder  # encoder必须指向一个静态方法
            task_heads[task] = task_head
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        out = {}
        for task, task_head in self.task_heads.items():
            out[task] = task_head(x)
        return out

    def forward_by_task(self, x, task):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.task_heads[task](x)
        return x

    def inference(self, meta, resize_keep_ratio):
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            preds = self(meta['img'])
            torch.cuda.synchronize()
            time2 = time.time()
            print('forward time: {:.3f}s'.format((time2 - time1)), end=' | ')
            dets = {}
            for task, pred in preds.items():
                det = self.task_heads[task].decode(pred, meta, resize_keep_ratio)
                dets[task] = det
            torch.cuda.synchronize()
            print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
        return dets

    def forward_train(self, gt_meta):
        preds = self.forward(gt_meta['imgs'])
        b = gt_meta['imgs'].shape[0]
        step = b // len(preds)
        assert b % len(preds) == 0
        loss = 0
        loss_states = {}
        outputs = {}
        for idx, (task, pred) in enumerate(preds.items()):
            pred_per_task = split_out(pred, idx * step, (idx + 1) * step)
            outputs[task] = pred_per_task
            per_task_loss, per_task_loss_states = self.task_heads[task].loss(pred_per_task, gt_meta[task])
            for loss_name, loss_val in per_task_loss_states.items():
                loss_states[task + '-' + loss_name] = loss_val
            loss += per_task_loss
        return outputs, loss, loss_states
