"""
Create by Chengqi.Lv
2020/9/27

Focus Module From yolo-v5
https://github.com/ultralytics/yolov5/blob/7220cee1d1dc1f14003dbf8d633bbb76c547000c/models/common.py#L82

"""


import torch
import torch.nn as nn
from .conv import act_dict


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, input_channel, out_channel, kernal_size=1, stride=1, padding=None, group=1, activation='ReLU'):
        super(Focus, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel * 4, out_channel, kernal_size, stride, padding, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
            act_dict[activation]
        )

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
