# -*- coding: utf-8 -*-
# @Time : 3/31/21

'''
created by ZhangXuhao
date: 2021.02.26
refered from https://github.com/LSQsjtu/blazehand/blob/master/palm_detection/blazepalm.py
功能: 手掌检测; 1*3*128*128输入的新版tflite
输出：regression and classification shape:  torch.Size([1, 896, 18]) torch.Size([1, 896, 1])
num_boxes： 896
num_coords: 18
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


"""(256, 256)输入, 采用BlazeBlock(kernel_size=5, activation='pReLU')"""
class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.channel_pad = out_channels - in_channels
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.act = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        # TFLite uses slightly different padding than PyTorch on the depthwise conv layer when the stride is 2.
        if self.stride == 2:
            h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)  #这里是对的,因为pytorch中输出通道在4维上的第2个上,对应倒数第3维
        return self.act(self.convs(h) + x)


class BlazePalm128(nn.Module):
    def __init__(self, img_size=(128, 128)):
        super(BlazePalm128, self).__init__()
        self.img_size = img_size
        self._define_layers()

    def _define_layers(self):
        # 相对于之前的旧版模型,重复的模块组由7->3. 重要不同卷积核大小不是3*3,而是5*5
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0, bias=True),
            nn.PReLU(num_parameters=32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
        )
        self.stage2 = nn.Sequential(
            BlazeBlock(32, 64, stride=2),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            BlazeBlock(64, 128, stride=2),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
        )
        self.stage4 = nn.Sequential(
            BlazeBlock(128, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )
        self.stage5 = nn.Sequential(
            BlazeBlock(256, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )

        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.PReLU(num_parameters=256),
            )
        self.deconv1_blaze = nn.Sequential(BlazeBlock(256, 256), BlazeBlock(256, 256),)

        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.PReLU(num_parameters=128),
            )
        self.deconv2_blaze = nn.Sequential(BlazeBlock(128, 128), BlazeBlock(128, 128),)

        self.classifier_16 = nn.Conv2d(128, 2, 1, bias=True) #跟之前图反一下, 输出16*16大小的图所对应的
        self.classifier_8= nn.Conv2d(256, 6, 1, bias=True)

        self.regressor_16 = nn.Conv2d(128, 36, 1, bias=True)
        self.regressor_8 = nn.Conv2d(256, 108, 1, bias=True)

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer than PyTorch, so do it manually.
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        b = x.shape[0]      # batch size, needed for reshaping later
        x = self.stage1(x)           # (b, 32, 64, 64) num_batchsize,num_output,heatmap_w,heat_h
        x = self.stage2(x)           # (b, 64, 32, 32)
        x_s3 = self.stage3(x)        # (b, 128, 16, 16) 这里输出后面有个支路,一直往下与第2个上采样结果相加
        x_s4 = self.stage4(x_s3)     # (b, 256, 8, 8) 后面有支路,与第一个上采样结果ADD
        x_s5 = self.stage5(x_s4)     # (b, 256, 4, 4)

        x_d1 = self.deconv1(x_s5)    # (b, 256, 8, 8)
        x_d1_blaze = self.deconv1_blaze(x_d1 + x_s4)  # (b, 256, 8, 8)  有两个分支:1) 接class8前面的卷积层; 2)接reg8前面的卷积层

        x_d2 = self.deconv2(x_d1_blaze)
        x_d2_blaze = self.deconv2_blaze(x_d2 + x_s3)  # (b, 128, 16, 16)

        # 分类分支
        c1 = self.classifier_16(x_d2_blaze)  #(b,2,16,16)
        c2 = self.classifier_8(x_d1_blaze)   #(b,6,8,8)
        c1 = c1.permute(0, 2, 3, 1) # 应该是为了转到tensorflow格式,NCWH ->NWHC
        c1 = c1.reshape(b, -1, 1)  # (b,512,1)
        c2 = c2.permute(0, 2, 3, 1)
        c2 = c2.reshape(b, -1, 1)  # (b,384,1)
        c = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        # 回归分支 regressors
        r1 = self.regressor_16(x_d2_blaze)
        r2 = self.regressor_8(x_d1_blaze)
        r1 = r1.permute(0, 2, 3, 1)
        r1 = r1.reshape(b, -1, 18)
        r2 = r2.permute(0, 2, 3, 1)
        r2 = r2.reshape(b, -1, 18)
        r = torch.cat((r1, r2), dim=1)  # (b, 896, 18)

        # reg = r.numpy() ##debug用的
        # cls = c.numpy()
        return [r, c]


if __name__ == '__main__':
    model = BlazePalm128()
    test_input = torch.randn(1, 3, 128, 128)
    r, c = model(test_input)
    print('regression and classification shape: ', r.shape, c.shape)
