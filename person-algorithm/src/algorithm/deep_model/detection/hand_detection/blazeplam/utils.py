# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

# TODO tmq 方法的优化，函数结构的优化（放到resize中）
#           hand_landmarks中也用到了此方法,但landmarks中还涉及到了rotate_rect的变动。
# def compute_output_letterbox_padding(input_size, output_size): #发现检测框不是正方形，有问题。
#     """size(width, height); padding: left, top, right, bottom"""
#     padding = [0.0, 0.0, 0.0, 0.0]
#     input_aspect_ratio = input_size[1] / input_size[0]
#     output_aspect_ratio = output_size[1] / output_size[0]
#     if input_aspect_ratio > output_aspect_ratio:
#         padding[0] = (1 - input_aspect_ratio / output_aspect_ratio) / 2
#         padding[2] = padding[0]
#     else:
#         padding[1] = (1 - output_aspect_ratio / input_aspect_ratio) / 2
#         padding[3] = padding[1]
#     return padding


def compute_output_letterbox_padding(input_size, output_size):
    """size(width, height); padding: left, top, right, bottom"""
    padding = [0.0, 0.0, 0.0, 0.0]
    input_aspect_ratio = input_size[0] / input_size[1]
    output_aspect_ratio = output_size[0] / output_size[1]
    if input_aspect_ratio < output_aspect_ratio:
        padding[0] = (1 - input_aspect_ratio / output_aspect_ratio) / 2
        padding[2] = padding[0]
    elif output_aspect_ratio < input_aspect_ratio:
        padding[1] = (1 - output_aspect_ratio / input_aspect_ratio) / 2
        padding[3] = padding[1]
    return padding


"""
shape(H*W*C), size(W*H)
"""
def scale_resize(img, scaled_size, resize_keep_ratio=True):
    if resize_keep_ratio:
        shape = np.r_[img.shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 2 # 这个成立的前提是, 已知scaled的图片是正方形的
        img_pad = np.pad(img, ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), mode='constant')
        scaled_img = cv2.resize(img_pad, scaled_size)
    else:
        scaled_img = cv2.resize(img, scaled_size)
    scaled_img = np.ascontiguousarray(scaled_img)
    return scaled_img


















