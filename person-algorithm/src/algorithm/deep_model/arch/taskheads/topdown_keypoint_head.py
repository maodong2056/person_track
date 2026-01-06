# @Time : 2021/3/18 10:51 
# @Author : Altair.Huazj
# @File : topdown_keypoint_head.py 
# @Software: PyCharm
import torch
from torch import nn

class TopDownKPHead(nn.Module):
    def __init__(self, input_channel, out_channel, kernel_size):
        super(TopDownKPHead, self).__init__()
        self.input_channel = input_channel
        pad = 0 if kernel_size==1 else 1
        self.final_layer = nn.Conv2d(input_channel, out_channel, kernel_size, stride=1)

    def forward(self, x):
        return self.final_layer(x[0])

