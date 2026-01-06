# -*- coding: utf-8 -*-
# @Time : 4/1/21

# """
# 输入: 一张图片RGB (经过水平翻转)
# 输出: detections, tensor([num_palm, 19])
# Each detection is a PyTorch tensor consisting of 19 numbers:
#             - xmin, ymin, xmax, ymax
#             - x,y-coordinates for the 7 keypoints
#             - confidence score
# """

import cv2
import math
import numpy as np
from scipy.special import expit
import torch


from src.algorithm.deep_model.arch import BasicDeepModel
from src.algorithm.deep_model.detection.hand_detection.blazeplam.model.palm_detection_128 import BlazePalm128
from src.algorithm.deep_model.detection.hand_detection.blazeplam.model.palm_detection_256 import BlazePalm256
from src.algorithm.deep_model.detection.hand_detection.blazeplam.utils import compute_output_letterbox_padding, \
                                                                                scale_resize

from src.algorithm.baseAlgorithm import detection_to_roi



class ConfigParam(object):
    num_classes = 1
    label_text = ["Palm"]
    num_coords = 18
    box_coord_offset = 0
    keypoint_coord_offset = 4
    num_keypoints = 7
    num_values_per_keypoint = 2
    sigmoid_score = True
    score_clipping_thresh = 100.0
    reverse_output_order = True
    min_score_thresh = 0.5
    nms_min_suppression_threshold = 0.3
    # nms_min_suppression_threshold = 0.1  #todo nms
    nms_overlap_type = 'INTERSECTION_OVER_UNION'
    nms_algorithm = 'WEIGHTED'
    apply_exponential_on_box_size = None

    def __init__(self, img_size):
        if img_size == (128, 128):
            self.num_boxes = 896
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
        elif img_size == (256, 256):
            self.num_boxes = 2944
            self.x_scale = 256.0
            self.y_scale = 256.0
            self.h_scale = 256.0
            self.w_scale = 256.0
        else:
            raise NotImplementedError


# def load_model(model_img_size, model_path):
#     if model_img_size == (128, 128):
#         model = BlazePalm128()
#     elif model_img_size == (256, 256):
#         model = BlazePalm256()
#     else:
#         raise NotImplementedError
#     model.load_state_dict(torch.load(model_path))
#     return model

def create_model(model_img_size):
    if model_img_size == (128, 128):
        model = BlazePalm128()
    elif model_img_size == (256, 256):
        model = BlazePalm256()
    else:
        raise NotImplementedError
    return model


def load_anchors(anchor_path, device):
    anchors = torch.tensor(np.load(anchor_path), dtype=torch.float32, device=device)
    return anchors


class HandDetection(BasicDeepModel):
    """
    检测器, bbox(xmin, ymin, xmin, ymin) + 7个关键点(x, y) + score
    """
    # def __init__(self, model_img_size, model_path, anchor_path, device='cuda:0', *args, **kwargs):
    def __init__(self, is_useGpu, gpu_num, model_img_size, anchor_path, *args, **kwargs):
        super(HandDetection, self).__init__(is_useGpu, gpu_num, *args, **kwargs)
        self.model_img_size = (model_img_size,model_img_size)
        self.param = ConfigParam(self.model_img_size)
        self.model = create_model(self.model_img_size)
        # self.model = model.to(device).eval()
        self.anchors = load_anchors(anchor_path, self.device)
        # self.device = device
        self.letterbox_padding = None


    # def __get_roi(self, width, height, norm_rect):
    #     """
    #     :param width: norm_rect区域所在的原始图片宽
    #     :param height: norm_rect区域所在的原始图片高
    #     :param norm_rect: (xc, yc, w, h, rotation) 归一化的结果
    #     :return: roi (xc, yc, w, h, rotation) 尺寸放大到原始图片上, rotation in rad
    #     """
    #     if norm_rect.size != 0:
    #         scale = np.array([width, height, width, height, 1.0])
    #         roi = norm_rect * scale
    #     else:
    #         roi = np.array([width * 0.5, height * 0.5, width, height, 0.0])
    #     return roi

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

    def __preprocess(self, img, norm_rect):
        img = img[:, :, ::-1]
        img_h, img_w, _ = img.shape
        self.norm_rect = norm_rect

        if self.norm_rect.size == 0:
            roi_img = img
        else:
            scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            rect = self.norm_rect[:4] * scale
            xmin = max(int(rect[0] - rect[2]/2), 0)
            ymin = max(int(rect[1] - rect[3]/2), 0)
            xmax = min(int(rect[0] + rect[2]/2), img.shape[1])
            ymax = min(int(rect[1] + rect[3]/2), img.shape[0])
            roi_img = img[ymin:ymax, xmin:xmax]

        roi_img_size = (roi_img.shape[1], roi_img.shape[0])
        self.letterbox_padding = compute_output_letterbox_padding(roi_img_size, self.model_img_size)
        scaled_img = scale_resize(roi_img, self.model_img_size, resize_keep_ratio=True)
        norm_img = np.ascontiguousarray(2 * ((scaled_img / 255) - 0.5).astype('float32'))
        if isinstance(norm_img, np.ndarray):
            norm_img = torch.from_numpy(norm_img).permute((2, 0, 1)).to(self.device)
        norm_img = norm_img.unsqueeze(0)
        norm_img = norm_img.to(self.device)
        return norm_img

    # TODO 讨论：是否有必要设置param.box_coord_offset
    def __decode_boxes(self, regression, anchors):
        """
        特殊情况说明：
        1.回归输出为x,y,w, h的顺序, reverse_output_order条件的判断处理省去
        2.手掌的anchor中w,h信息全部为１, 考虑到通用性未略去
        3. 简化了对box_coord_offset的处理
        """
        boxes = np.zeros_like(regression)
        scale = np.array([self.param.x_scale, self.param.y_scale]).reshape(1, 2)
        box_start = self.param.box_coord_offset
        box_end = self.param.box_coord_offset + 4
        center = regression[:, box_start:box_start+2] / scale * anchors[:, 2:] + anchors[:, :2]
        wh = regression[:, box_start+2:box_end] / scale * anchors[:, 2:]
        boxes[:, :2] = center - wh / 2
        boxes[:, 2:4] = center + wh / 2
        start = self.param.keypoint_coord_offset
        stop = self.param.keypoint_coord_offset + self.param.num_keypoints * self.param.num_values_per_keypoint
        step = self.param.num_values_per_keypoint
        for idx in range(start, stop, step):
            boxes[:, idx:idx + 2] = regression[:, idx:idx + 2] / scale * anchors[:, 2:] + anchors[:, :2]
        return boxes

    def __tensors_to_detections(self, regression, classification, anchors):
        """This function converts (regression, classification, anchors)
        into into a list of (num_detections, 19) detection results
        :param regression:  numpy.ndarray of shape (num_boxes, num_coords)
        :param classification: a numpy.ndarray of shape (num_boxes, num_classes)
        :param anchors: a numpy.ndarray of shape (num_boxes, 4)
        :return:
        detections: numpy.ndarray (num_detections, 19) for each image
        Each detection contains:
            - xmin, ymin, xmax, ymax
            - x,y-coordinates for the 7 keypoints
            - confidence score
        """
        clipped_score = self.param.score_clipping_thresh
        scores = np.clip(classification, -1 * clipped_score, clipped_score)
        scores = expit(scores)
        mask = scores.squeeze(axis=-1) >= self.param.min_score_thresh
        filtered_scores = scores[mask]
        filtered_reg = regression[mask]
        filtered_anchors = anchors[mask]
        boxes = self.__decode_boxes(filtered_reg, filtered_anchors)
        detections = np.concatenate((boxes, filtered_scores), axis=-1)
        return detections

    # IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
    def __intersect(self, box_a, box_b):
        """ We resize both numpy array to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: bounding boxes, Shape: [A,4].
          box_b: bounding boxes, Shape: [B,4].
        Return:
          intersection area, Shape: [A,B].
        """
        A = box_a.shape[0]
        B = box_b.shape[0]
        max_xy = np.minimum(np.expand_dims(box_a[:, 2:], axis=1).repeat(B, axis=1),
                            np.expand_dims(box_b[:, 2:], axis=0).repeat(A, axis=0))
        min_xy = np.maximum(np.expand_dims(box_a[:, :2], axis=1).repeat(B, axis=1),
                            np.expand_dims(box_b[:, :2], axis=0).repeat(A, axis=0))
        inter = np.maximum(max_xy - min_xy, 0)
        return inter[:, :, 0] * inter[:, :, 1]

    def __jaccard(self, box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: Ground truth bounding boxes, Shape: [A,4]
            box_b: Prior boxes, Shape: [B,4]
        Return:
            jaccard overlap:  Shape: [A, B]
        """
        inter = self.__intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
        area_a = np.expand_dims(area_a, axis=1).repeat(area_b.shape[0], axis=1)  # [A,B]
        area_b = np.expand_dims(area_b, axis=0).repeat(area_a.shape[0], axis=0)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def __overlap_similarity(self, box, other_boxes):
        """Computes the IOU between a bounding box and set of other boxes."""
        return self.__jaccard(np.expand_dims(box, axis=0), other_boxes).squeeze()

    # TODO 不同nms是否需要独立编程,配置文件方法
    def __non_max_suppression(self, detections, type=None):
        if len(detections) == 0:
            return []

        output_detections = []
        remaining = np.argsort(detections[:, 18])
        while len(remaining) > 0:
            detection = detections[remaining[-1]]
            # Compute the overlap between the first box and the other
            # remaining boxes. (the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = self.__overlap_similarity(first_box, other_boxes)
            mask = ious > self.param.nms_min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]
            if type == "weighted":
                weighted_detection = detection.copy()
                if len(overlapping) > 1:
                    coordinates = detections[overlapping, :18]
                    scores = detections[overlapping, 18:19]
                    total_score = scores.sum()
                    weighted = (coordinates * scores).sum(axis=0) / total_score
                    weighted_detection[:18] = weighted
                    weighted_detection[18] = total_score / len(overlapping)
                output_detections.append(weighted_detection)
            else:
                output_detections.append(detection)
        return output_detections

    def __detection_letterbox_removal(self, detections):
        """
        实现功能: 调整检测结果，使其不处于padding区域, adding[left, top, right, bottom]
        :param detections: tensor([num_detecitons, 19]), (归一化结果)
        :return:adjusted_detections
        """
        left = self.letterbox_padding[0]
        top = self.letterbox_padding[1]
        left_and_right = self.letterbox_padding[0] + self.letterbox_padding[2]
        top_and_bottom = self.letterbox_padding[1] + self.letterbox_padding[3]
        detections[:, 0:18:2] = (detections[:, 0:18:2] - left) / (1.0 - left_and_right)
        detections[:, 1:18:2] = (detections[:, 1:18:2] - top) / (1.0 - top_and_bottom)
        detections = detections.clip(0.0, 1.0)
        return detections

    def __detection_projection(self, detections):
        if self.norm_rect.size == 0:
            return detections
        detections[:, 0:18:2] = (detections[:, 0:18:2] - 0.5) * self.norm_rect[2] + self.norm_rect[0]
        detections[:, 1:18:2] = (detections[:, 1:18:2] - 0.5) * self.norm_rect[3] + self.norm_rect[1]
        return detections

    def __postprocess(self, regression, classification):
        regression = regression.cpu().numpy().squeeze(axis=0)
        classification = classification.cpu().numpy().squeeze(axis=0)
        anchors = self.anchors.cpu().numpy()
        detections = self.__tensors_to_detections(regression, classification, anchors)
        # filtered_detections = self.__non_max_suppression(detections, type="weighted")
        filtered_detections = self.__non_max_suppression(detections)
        filtered_detections = np.array(filtered_detections) if len(filtered_detections) > 0 else np.zeros((0, 19))
        adjusted_detections = self.__detection_letterbox_removal(filtered_detections)
        adjusted_detections = self.__detection_projection(adjusted_detections)

        return adjusted_detections

    def get_output(self, img, norm_rect=np.array([]),detection_hand_thres=0.5, *args, **kwargs):
        """
        :param img:RGB image
        :param norm_rect: (xc, yc, w, h, rotation) 归一化的结果.在与人体联合是需要用到此参数
        :param args:
        :param kwargs:
        :return: detections: numpy.ndarray [num_detections, 19] for each image (归一化结果)
        Each detection contains:
            - xmin, ymin, xmax, ymax
            - x,y-coordinates for the 7 keypoints
            - confidence score
        """
        norm_img = self.__preprocess(img, norm_rect)
        with torch.no_grad():
            regression, classification = self.model(norm_img)
        detections = self.__postprocess(regression, classification)
        #added by zhxh,2022.04.01为了和框架对应，将阈值筛选放后处理实现
        dets = [detections[i] for i in range(len(detections)) if detections[i][18]>detection_hand_thres]
        # print(f'det_out:{dets}')
        # '''可视化debug ''' #todo
        # if len(dets)>0:
        #     from src.utils.draw_hand import draw_hand_bbox
        #     bbox = dets[0][:4]
        #     img_bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     det_bbox_img = draw_hand_bbox(img_bgr,(bbox), is_norm=True)
        #     # cv2.putText(det_bbox_img, str(self.frame_num), (det_bbox_img.shape[1] // 5, det_bbox_img.shape[0] // 7),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        #     cv2.imshow('det_bbox_2', det_bbox_img)
        #     cv2.waitKey(10)
        output = []
        if dets != []:
            for predict in dets:
                norm_roi = detection_to_roi(predict, img.shape[1], img.shape[0])
                res = {
                    'hand_box'  : predict[:4],
                    'hand_conf' : predict[18],
                    'hand_lm'   : predict[4:18],
                    'hand_roi'  : norm_roi
                }
                output.append(res)

        return output





