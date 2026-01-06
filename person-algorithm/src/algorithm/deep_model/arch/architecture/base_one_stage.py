"""
Create by Chengqi.Lv
2020/3/23
"""
import time
import torch
import torch.nn as nn
from src.algorithm.deep_model.arch.backbones.builder import build_backbone
from src.algorithm.deep_model.arch.necks import build_neck
from src.algorithm.deep_model.arch import build_head


class BaseOneStage(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg=None,
                 task_head_cfg=None,):
        super(BaseOneStage, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if neck_cfg is not None:
            self.neck = build_neck(neck_cfg)
        if task_head_cfg is not None:
            self.task_head = build_head(task_head_cfg)
            if hasattr(self.task_head, 'encoder'):
                self.encoder = self.task_head.encoder  # encoder必须指向一个静态方法

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, 'neck') and self.neck is not None:
            x = self.neck(x)
        if hasattr(self, 'task_head'):
            x = self.task_head(x)
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





if __name__ == '__main__':
    from econn.utils.config import cfg, update_config

    update_config(cfg, r'D:\Projects\ECO_NN\configs\normbox_ecoindoor_mobilev2relu6AllDW_example.yml')
    print(cfg)
    detector = BaseOneStage(cfg.model.backbone, cfg.model.neck, cfg.model.task_head)
    print(detector)
    input = torch.randn(1, 3, 320, 320)
    output = detector.inference(input)
    print(output[0][0].shape)
    # torch.onnx.export(detector, input, os.path.join(r"D:\tmp\test.onnx"), verbose=True)
