# -*- coding: utf-8 -*-
# @Time : 4/15/21


import numpy as np
import math

"""
手掌检测结果转化为手部roi区域(xc, yc, w, h, rotation), 
前4项xc, yc, w, h为归一化结果,
rotation未手腕与中指根节点连线与竖直向上方向的夹角, in radians
"""


rotation_vector_start_keypoint_index = 0  # Center of wrist.
rotation_vector_end_keypoint_index = 2  # MCP of middle finger.
rotation_vector_target_angle_degrees = 90

# rect_scale_x = 2.6
# rect_scale_y = 2.6
rect_scale_x = 3.5
rect_scale_y = 3.5
rect_shift_y = -0.5
rect_shift_x = 0.0
rect_square_long = True
rect_square_short = False


# 归一化到[-math.pi, math.pi)之间
def normalize_radians(angle):
    return angle - 2 * math.pi * math.floor((angle - (-math.pi)) / (2 * math.pi))


def compute_rect_rotation(detection, width, height):
    """
    计算手腕与中指根关节连线与竖直向上方向夹角(in radian); relative_keypoints 0 (wrist), 2(mcp)
    :param detection:手掌检测结果
    :param width:手掌检测结果对应图片宽
    :param height:手掌检测结果对应图片高
    :return:rotation (in Radians)
    """
    start_idx = 4 + rotation_vector_start_keypoint_index * 2
    end_idx = 4 + rotation_vector_end_keypoint_index * 2
    target_angle = rotation_vector_target_angle_degrees * math.pi / 180
    x0 = detection[start_idx] * width
    y0 = detection[start_idx + 1] * height
    x1 = detection[end_idx] * width
    y1 = detection[end_idx + 1] * height
    rotation = target_angle - math.atan2(-(y1 - y0), x1 - x0)
    return normalize_radians(rotation)


def detection_to_rect(detection, img_width, img_height, use_keypoints=False):
    """
    :param detection: (xmin, ymin, xmax, ymax , keypoint_x, keypoint_y)
    :param use_keypoints: 采用keypoints最小外接矩,反之采用bbox
    :return:rect (xc, yc, w, h, rotation)
    """
    # print(detection)
    rect = np.zeros(5, dtype=np.float32)
    bbox = np.zeros(4, dtype=np.float32)
    if use_keypoints:
        bbox[0] = min(detection[4:18:2])
        bbox[1] = min(detection[5:18:2])
        bbox[2] = max(detection[4:18:2])
        bbox[3] = max(detection[5:18:2])
    else:
        bbox = detection[:4]
    rect[0] = (bbox[0] + bbox[2]) / 2
    rect[1] = (bbox[1] + bbox[3]) / 2
    rect[2] = bbox[2] - bbox[0]
    rect[3] = bbox[3] - bbox[1]
    rect[4] = compute_rect_rotation(detection, img_width, img_height)
    return rect


def rect_transformation(rect, img_width, img_height):
    """
    对手掌得到的带旋转角度的rect进行尺度变化(偏移,放大),以得到整个手部的roi区域
    :param rect: xc, yc, w, h, rotation
    :return:roi: xc, yc, w, h, rotation
    """
    roi = np.copy(rect)
    x_shift = (img_width * rect[2] * rect_shift_x * math.cos(rect[4]) -
               img_height * rect[3] * rect_shift_y * math.sin(rect[4])) / img_width
    y_shift = (img_width * rect[2] * rect_shift_x * math.sin(rect[4]) +
               img_height * rect[3] * rect_shift_y * math.cos(rect[4])) / img_height
    roi[0] = rect[0] + x_shift
    roi[1] = rect[1] + y_shift
    # 调整为正方型
    if rect_square_long:
        long_side = max(roi[2] * img_width, roi[3] * img_height)
        roi[2] = long_side / img_width
        roi[3] = long_side / img_height
    elif rect_square_short:
        short_side = min(roi[2] * img_width, roi[3] * img_height)
        roi[2] = short_side / img_width
        roi[3] = short_side / img_height
    roi[2] = roi[2] * rect_scale_x
    roi[3] = roi[3] * rect_scale_y
    return roi


def detection_to_roi(detection, img_width, img_height, use_keypoints=False):
    rect = detection_to_rect(detection, img_width, img_height, use_keypoints)
    roi = rect_transformation(rect, img_width, img_height)
    return roi


######------------------------------------------------------------------------------------------------

# """
# 手部关键点检测结果转化为手部roi区域(xc, yc, w, h, rotation),
# 前4项xc, yc, w, h为归一化结果,
# rotation未手腕与中指根节点连线与竖直向上方向的夹角, in radians
# """


# 12个关键点, the landmarks extracted are: wrist, MCP/PIP of five fingers
partial_landmarks_idx = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
# 在12个关键点中的坐标
kWristJoint = 0
kMiddleFingerPIPJoint = 6
kIndexFingerPIPJoint = 4
kRingFingerPIPJoint = 8
kTargetAngle = math.pi * 0.5

# 默认参数
landmarks_scale_x = 2.0
landmarks_scale_y = 2.0
landmarks_shift_x = 0
landmarks_shift_y = -0.1

# landmarks_scale_x = 1.8
# landmarks_scale_y = 1.8
# landmarks_shift_x = 0
# landmarks_shift_y = -0.15

landmarks_square_long = True
landmarks_square_short = False


# 归一化到[-math.pi, math.pi)之间
# note (detection中也会用到,是否需要放置到utils中)


def compute_rotation(landmarks, size):
    xy0 = landmarks[kWristJoint, :2] * size
    xy1 = (landmarks[kIndexFingerPIPJoint, :2] + landmarks[kRingFingerPIPJoint, :2]) / 2
    xy1 = (xy1 + landmarks[kMiddleFingerPIPJoint, :2]) / 2 * size
    angle = kTargetAngle - math.atan2(-(xy1[1] - xy0[1]), xy1[0] - xy0[0])
    angle = normalize_radians(angle)
    return angle

# TODO tmq 复杂的计算过程转为矩阵乘法
def normalized_landmark_list_to_rect(landmarks, img_width, img_height):
    """
    landmarks: 筛选后的归一化关键点, (12,3)
    """
    size = (img_width, img_height)
    rotation = compute_rotation(landmarks, size)
    reverse_angle = normalize_radians(-1 * rotation)

    landmarks = landmarks[:, :2]
    min_xy = np.min(landmarks, axis=0)
    max_xy = np.max(landmarks, axis=0)
    axis_aligned_center = (min_xy + max_xy) / 2

    original = (landmarks - axis_aligned_center) * size
    projected_x = original[:, 0] * math.cos(reverse_angle) - original[:, 1] * math.sin(reverse_angle)
    projected_y = original[:, 0] * math.sin(reverse_angle) + original[:, 1] * math.cos(reverse_angle)
    max_x = max(max_xy[0], np.max(projected_x, axis=0))
    max_y = max(max_xy[1], np.max(projected_y, axis=0))
    min_x = min(min_xy[0], np.min(projected_x, axis=0))
    min_y = min(min_xy[1], np.min(projected_y, axis=0))
    projected_center_x = (max_x + min_x) / 2
    projected_center_y = (max_y + min_y) / 2

    center_x = projected_center_x * math.cos(rotation) - projected_center_y * math.sin(rotation) + size[0] * axis_aligned_center[0]
    center_y = projected_center_x * math.sin(rotation) + projected_center_y * math.cos(rotation) + size[1] * axis_aligned_center[1]
    width = (max_x - min_x) / size[0]
    height = (max_y - min_y) / size[1]

    rect = np.array([center_x / size[0],
                     center_y / size[1],
                     width,
                     height,
                     rotation
                     ])
    return rect

# NOTE tmq 与detection_to_roi函数一致,仅仅配置文件不同
def rect_transformation_landmark(rect, img_width, img_height):
    """
    对手掌得到的带旋转角度的rect进行尺度变化(偏移,放大),以得到整个手部的roi区域
    :param rect: xc, yc, w, h, rotation
    :return:roi: xc, yc, w, h, rotation
    """
    roi = np.copy(rect)
    x_shift = (img_width * rect[2] * landmarks_shift_x * math.cos(rect[4]) -
               img_height * rect[3] * landmarks_shift_y * math.sin(rect[4])) / img_width
    y_shift = (img_width * rect[2] * landmarks_shift_x * math.sin(rect[4]) +
               img_height * rect[3] * landmarks_shift_y * math.cos(rect[4])) / img_height
    roi[0] = rect[0] + x_shift
    roi[1] = rect[1] + y_shift
    # 调整为正方型
    if landmarks_square_long:
        long_side = max(roi[2] * img_width, roi[3] * img_height)
        roi[2] = long_side / img_width
        roi[3] = long_side / img_height
    elif landmarks_square_short:
        short_side = min(roi[2] * img_width, roi[3] * img_height)
        roi[2] = short_side / img_width
        roi[3] = short_side / img_height
    roi[2] = roi[2] * landmarks_scale_x
    roi[3] = roi[3] * landmarks_scale_y
    return roi

def hand_landmarks_to_roi(landmarks, img_width, img_height):
    """
    :param landmarks: numpy.ndarray ([21, 3])
    :return: roi: (xc, yc, w, h, rotation), 由兴趣关键点得到的兴趣区域
    """
    partial_landmarks = landmarks[partial_landmarks_idx]
    rect = normalized_landmark_list_to_rect(partial_landmarks, img_width, img_height)
    roi = rect_transformation_landmark(rect, img_width, img_height)
    return roi



#--added by zhxh,20220405
def xcycwh2xyxy(bbox):
    bx_1 = np.array(bbox.copy())
    if bx_1.shape[0]!=0:
        # x1 = bx_1[:, 0] - bx_1[:, 2] / 2.
        # y1 = bx_1[:, 1] - bx_1[:, 3] / 2.
        # x2 = bx_1[:, 0] + bx_1[:, 2] / 2.
        # y2 = bx_1[:, 1] + bx_1[:, 3] / 2.
        # bx_1[:, :4] = np.c_[x1, y1, x2, y2]
        x1 = bx_1[0] - bx_1[2] / 2.
        y1 = bx_1[1] - bx_1[3] / 2.
        x2 = bx_1[0] + bx_1[2] / 2.
        y2 = bx_1[1] + bx_1[3] / 2.
        bx_1[:4] = np.c_[x1, y1, x2, y2]
    return bx_1








