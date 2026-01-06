# -*- coding: utf-8 -*-
"""
2022.04.01
"""

# """
# 关键点模型仅对一只手的关键点进行预测
# 输入: 一张图片RGB (已经过水平翻转), roi (手部兴趣区域)
# 输出:
# """

import cv2
import numpy as np
import math
import torch


from src.algorithm.deep_model.arch.basic_model import BasicDeepModel
from src.algorithm.deep_model.keypoint.hand_kps2d.blazepalm_kps.model.hand_landmarks_256 import Landmarks
from src.algorithm.baseAlgorithm import detection_to_roi, hand_landmarks_to_roi

# TODO tmq image to tensor中规定数据范围时 0, 1, 而不是(-1,1), 归一化是否有影响;
#          matrix矩阵使用(目前未用）


class ConfigParam(object):
    num_classes = 2
    class_label_text = ["Left", "Right"]
    num_landmarks = 21
    presence_score_threshold = 0.5

    def __init__(self, img_size):
        if img_size == (256, 256):
            self.model_input_width = 256.0
            self.model_input_height = 256.0
            self.normalize_z = 0.4
        else:
            raise NotImplementedError


# def load_model(model_path):
#     model = Landmarks()
#     model.load_state_dict(torch.load(model_path))
#     return model


class HandLandmarks(BasicDeepModel):
    """
    手部关键点模型
    """
    # def __init__(self, model_img_size, model_path, device='cuda:0'):
    def __init__(self, is_useGpu, gpu_num, model_img_size,*args, **kwargs):
        super(HandLandmarks, self).__init__(is_useGpu, gpu_num, *args, **kwargs)
        self.model_img_size = (model_img_size,model_img_size)
        self.param = ConfigParam(self.model_img_size)

        # model = load_model(model_path)
        # self.model = model.to(device).eval()
        self.model =Landmarks()
        # self.device = device
        self.norm_rect = None  # (xc, yx, w, h, rotation)
        self.letterbox_padding = None

    def __get_roi(self, width, height, norm_rect):
        """
        :param width: norm_rect区域所在的原始图片宽
        :param height: norm_rect区域所在的原始图片高
        :param norm_rect: (xc, yc, w, h, rotation) 归一化的结果
        :return: roi (xc, yc, w, h, rotation) 尺寸放大到原始图片上, rotation in rad
        """
        if len(norm_rect) != 0:
            scale = np.array([width, height, width, height, 1.0])
            roi = norm_rect * scale
        else:
            roi = np.array([width * 0.5, height * 0.5, width, height, 0.0])
        return roi

    def __get_letterbox_padding_and_update_roi(self, roi):
        padding = [0.0, 0.0, 0.0, 0.0]
        input_aspect_ratio = roi[3] / roi[2]
        output_aspect_ratio = self.model_img_size[0] / self.model_img_size[1]
        if input_aspect_ratio > output_aspect_ratio:
            padding[0] = (1 - input_aspect_ratio / output_aspect_ratio) / 2
            padding[2] = padding[0]
            roi[2] = roi[3] * input_aspect_ratio
        else:
            padding[1] = (1 - output_aspect_ratio / input_aspect_ratio) / 2
            padding[3] = padding[1]
            roi[3] = roi[2] * output_aspect_ratio
        return padding, roi

    def crop_and_transform_image(self, img, roi):
        src_rect = ((roi[0], roi[1]), (roi[2], roi[3]), roi[4] * 180 / math.pi)
        src_points = cv2.boxPoints(src_rect)
        dst_points = np.float32([[0, self.model_img_size[1]],
                                 [0, 0],
                                 [self.model_img_size[0], 0],
                                 [self.model_img_size[0], self.model_img_size[1]]])
        projection_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(img, projection_matrix, self.model_img_size, cv2.INTER_LINEAR)
        return transformed

    def __landmarks_letterbox_removal(self, norm_landmarks):
        """"
        实现功能: 调整检测结果，使其不处于padding区域, padding[left, top, right, bottom]
        """
        left = self.letterbox_padding[0]
        top = self.letterbox_padding[1]
        left_and_right = self.letterbox_padding[0] + self.letterbox_padding[2]
        top_and_bottom = self.letterbox_padding[1] + self.letterbox_padding[3]
        norm_landmarks[:, 0] = (norm_landmarks[:, 0] - left) / (1.0 - left_and_right)
        norm_landmarks[:, 1] = (norm_landmarks[:, 1] - top) / (1.0 - top_and_bottom)
        norm_landmarks = norm_landmarks.clip(0.0, 1.0)
        return norm_landmarks

    def __landmarks_projection(self, norm_landmarks):
        if len(self.norm_rect) == 0:
            return norm_landmarks
        angle = self.norm_rect[4]
        x = math.cos(angle) * (norm_landmarks[:, 0] - 0.5) - math.sin(angle) * (norm_landmarks[:, 1] - 0.5)
        y = math.sin(angle) * (norm_landmarks[:, 0] - 0.5) + math.cos(angle) * (norm_landmarks[:, 1] - 0.5)
        norm_landmarks[:, 0] = x * self.norm_rect[2] + self.norm_rect[0]
        norm_landmarks[:, 1] = y * self.norm_rect[3] + self.norm_rect[1]
        norm_landmarks[:, 2] = norm_landmarks[:, 2] * self.norm_rect[2]
        return norm_landmarks

    def __preprocess(self, img, norm_rect, *args, **kwargs):
        img = img[:, :, ::-1]
        img_h, img_w, _ = img.shape
        self.norm_rect = norm_rect
        roi_ = self.__get_roi(img_w, img_h, norm_rect)
        self.letterbox_padding, roi = self.__get_letterbox_padding_and_update_roi(roi_)
        transform_img = self.crop_and_transform_image(img, roi)

        # TODO tmq 归一化区间验证
        norm_img = np.ascontiguousarray(2 * ((transform_img / 255) - 0.5).astype('float32'))
        if isinstance(norm_img, np.ndarray):
            norm_img = torch.from_numpy(norm_img).permute((2, 0, 1)).to(self.device)
        norm_img = norm_img.unsqueeze(0)
        norm_img = norm_img.to(self.device)

        # '''可视化debug 2022.04.06-------------------------''' #--------------------todo
        # from src.utils.draw_hand import draw_hand_bbox
        # from src.utils.blazepalm_hand_utils import xcycwh2xyxy
        # img_bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # # print(f'landmark_input roi:{roi}')
        # test = draw_hand_bbox(img_bgr,xcycwh2xyxy(roi_[:-1]), is_norm=False)
        # cv2.imshow('roi_2', test)
        # cv2.waitKey(1)
        #
        # cv2.imshow('landmark_input_2',cv2.cvtColor(transform_img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)
        '''--------------------------------------------'''

        return norm_img

    def __postprocess(self, landmark_tensors, hand_flag_tensor, handedness_tensor):
        """
        后处理, batch_size=1
        :param landmark_tensors (1, 63) 21个关键点坐标 x y z
        :param hand_flag_tensor (1, 1) 手分类
        :param handedness_tensor (1, 1) 手得分
        :return:
        """
        confidence = float(hand_flag_tensor)
        is_right = True if float(handedness_tensor) < 0.5 else False
        landmarks = landmark_tensors.cpu().numpy().squeeze(axis=0).reshape(-1, 3)
        scale = np.array([self.param.model_input_width,
                          self.param.model_input_height,
                          self.param.model_input_width * self.param.normalize_z])
        norm_landmarks = landmarks / scale
        norm_landmarks = self.__landmarks_letterbox_removal(norm_landmarks)
        norm_landmarks = self.__landmarks_projection(norm_landmarks)
        return confidence, is_right, norm_landmarks

    # def get_output(self, img, norm_rect, *args, **kwargs):
    #     """
    #     :param img: RGB image
    #     :param norm_rect: (xc, yc, w, h, rotation) 归一化的结果
    #     :return:
    #     score (float)
    #     handedness (string), Left or Right
    #     landmarks (numpy.ndarray), 21*3
    #     """
    #     norm_img = self.__preprocess(img, norm_rect)
    #     with torch.no_grad():
    #         landmark_tensors, hand_flag_tensor, handedness_tensor = self.model(norm_img)
    #     score, is_right, norm_landmarks = self.__postprocess(landmark_tensors, hand_flag_tensor, handedness_tensor)
    #     landmarks_rotation = landmark_tensors.cpu().numpy().reshape(21, 3)
    #     return score, is_right, norm_landmarks, landmarks_rotation


    def get_output(self, img, inputs,keypoint_hand_thres=0.3, *args, **kwargs):
        """
        :param img: RGB image
        :param inputs:检测的结果，一般是多维数组 ---------
        :return:
        score (float)
        handedness (string), Left or Right
        landmarks (numpy.ndarray), 21*3
        """

        output = []
        for i in range(len(inputs)):
            input = inputs[i]
            if input is None: #roi区域是空值
                output = [{
                    'hand_keypoint' : np.zeros((21,3)),
                    'hand_kps_conf' : None,
                    'hand_isright'  : None,
                    'hand_roi'      : None,
                    'hand_tracking' : 0
                }]
                break
            # norm_roi = detection_to_roi(input, img.shape[1], img.shape[0]) ##todo 后面需要修改看看去掉
            norm_roi = input
            norm_img = self.__preprocess(img, norm_roi)
            with torch.no_grad():
                landmark_tensor, hand_presence_tensor, islefthand_prob_tensor = self.model(norm_img)
            score, is_right, norm_landmark = self.__postprocess(landmark_tensor, hand_presence_tensor, islefthand_prob_tensor)

            if score < keypoint_hand_thres: #关键点可信度分数小于阈值直接过滤
                res = {
                    'hand_keypoint' : np.zeros((21,3)),
                    'hand_kps_conf' : None,
                    'hand_isright'  : None,
                    'hand_roi'      : None,
                    'hand_tracking' : 0
                }

            else:
                landmarks_rotation = landmark_tensor.cpu().numpy().reshape(21, 3)

                kps_roi = hand_landmarks_to_roi(norm_landmark, img.shape[1], img.shape[0]) #当前关键点来得到新的手框,归一化值
                '''可视化ｄｅｂｕｇ'''
                # from src.utils.draw_hand import draw_hand_bbox
                # from src.utils.blazepalm_hand_utils import xcycwh2xyxy
                # img_bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # # print(f'landmark_input roi:{roi}')
                # test = draw_hand_bbox(img_bgr,xcycwh2xyxy(kps_roi[:-1]), is_norm=True)
                # cv2.imshow('roi_3', test)
                # cv2.waitKey(1)
                #-----------------------------------------------------------
                # keypoint = norm_landmark* np.array([img.shape[1], img.shape[0], 1.0]) #相对于原图的坐标 21*3
                keypoint = norm_landmark
                res = {
                    'hand_keypoint' : keypoint, #21*3
                    'hand_kps_conf' : score,
                    'hand_isright'  : is_right,
                    'hand_roi'      : kps_roi,
                    'hand_tracking' : 1

                }
            output.append(res)
            # print(f'kps_out:{keypoint}')
        return output