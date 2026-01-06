# @Time : 2021/3/18 9:41 
# @Author : Altair.Huazj
# @File : model.py 
# @Software: PyCharm
from torch import nn
from src.algorithm.deep_model.arch import MobileNetV2, TopDownKPHead, DeconvDWKp

class TopDownKP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV2(out_stages=(6,))
        # self.keypoint_neck = DeconvKp((1280,), [256,256,256])
        self.keypoint_neck = DeconvDWKp((1280,), [256, 256, 256])
        self.keypoint_head = TopDownKPHead(256, 51, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.keypoint_neck(x)
        x = self.keypoint_head(x)
        return x