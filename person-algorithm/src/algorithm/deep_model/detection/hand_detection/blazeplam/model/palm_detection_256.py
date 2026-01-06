# -*- coding: utf-8 -*-
# @Time : 3/31/21

'''
created by ZhangXuhao
date: 2021.02.26
refered from https://github.com/LSQsjtu/blazehand/blob/master/palm_detection/blazepalm.py
功能:手掌检测; 1*3*256*256输入的旧版tflite
输出：regression and classification shape:  [1, 2944, 18]) torch.Size([1, 2944, 1])
num_boxes： 2944
num_coords: 18
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


"""(256, 256)输入, 采用BlazeBlock(kernel_size=3, activation='ReLU')"""
class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
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
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # TFLite uses slightly different padding than PyTorch on the depthwise conv layer when the stride is 2.
        if self.stride == 2:
            h = F.pad(x, (0, 1, 0, 1), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x
        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        return self.act(self.convs(h) + x)


"""(1, 3, 256, 256)输入, 采用BlazeBlock(kernel_size=3, activation='ReLU')"""
class BlazePalm256(nn.Module):
    """
    The BlazeFace face detection model from MediaPipe.
    The version from MediaPipe is simpler than the one in the paper;
    it does not use the "double" BlazeBlocks.
    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.
    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.
    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.
    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """
    def __init__(self, img_size=(256, 256)):
        super(BlazePalm256, self).__init__()
        self.img_size = img_size
        self._define_layers()

    def _define_layers(self):
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
        )
        self.stage2 = nn.Sequential(
            BlazeBlock(32, 64, stride=2),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
        )
        self.stage3 = nn.Sequential(
            BlazeBlock(64, 128, stride=2),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
        )
        self.stage4 = nn.Sequential(
            BlazeBlock(128, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )
        self.stage5 = nn.Sequential(
            BlazeBlock(256, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256)
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            )
        self.deconv1_blaze = BlazeBlock(256, 256)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            )
        self.deconv2_blaze = BlazeBlock(128, 128)

        self.classifier_8 = nn.Conv2d(128, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(256, 2, 1, bias=True)
        self.classifier_32 = nn.Conv2d(256, 6, 1, bias=True)

        self.regressor_8 = nn.Conv2d(128, 36, 1, bias=True)
        self.regressor_16 = nn.Conv2d(256, 36, 1, bias=True)
        self.regressor_32 = nn.Conv2d(256, 108, 1, bias=True)

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer than PyTorch, so do it manually.
        x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        b = x.shape[0]      # batch size, needed for reshaping later
        x = self.stage1(x)           # (b, 32, 128, 128)
        x = self.stage2(x)           # (b, 64, 64, 64)
        x_s3 = self.stage3(x)           # (b, 128, 32, 32)
        x_s4 = self.stage4(x_s3)     # (b, 256, 16, 16)
        x_s5 = self.stage5(x_s4)     # (b, 256, 8, 8)

        x_d1 = self.deconv1(x_s5)
        x_d1_blaze = self.deconv1_blaze(x_d1 + x_s4)  # (b, 256, 16, 16)

        x_d2 = self.deconv2(x_d1_blaze)
        x_d2_blaze = self.deconv2_blaze(x_d2 + x_s3)  # (b, 128, 32, 32)

        c1 = self.classifier_8(x_d2_blaze)
        c2 = self.classifier_16(x_d1_blaze)
        c3 = self.classifier_32(x_s5)

        c1 = c1.permute(0, 2, 3, 1)
        c1 = c1.reshape(b, -1, 1)
        c2 = c2.permute(0, 2, 3, 1)
        c2 = c2.reshape(b, -1, 1)
        c3 = c3.permute(0, 2, 3, 1)
        c3 = c3.reshape(b, -1, 1)
        c = torch.cat((c1, c2, c3), dim=1)  # (b, 2944, 1)

        r1 = self.regressor_8(x_d2_blaze)
        r2 = self.regressor_16(x_d1_blaze)
        r3 = self.regressor_32(x_s5)

        r1 = r1.permute(0, 2, 3, 1)
        r1 = r1.reshape(b, -1, 18)
        r2 = r2.permute(0, 2, 3, 1)
        r2 = r2.reshape(b, -1, 18)
        r3 = r3.permute(0, 2, 3, 1)
        r3 = r3.reshape(b, -1, 18)
        r = torch.cat((r1, r2, r3), dim=1)  # (b, 2944, 18)
        return [r, c]




