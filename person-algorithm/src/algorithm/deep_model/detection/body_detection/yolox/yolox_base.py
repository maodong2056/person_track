#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch

import numpy as np

import cv2
import torch.nn as nn
from src.algorithm.deep_model.arch import BasicDeepModel
from .models import YOLOPAFPN, YOLOX, YOLOXHead, replace_module, SiLU


class YoloxDet(BasicDeepModel):

    def __init__(self, in_channels, depth, width, num_classes, rgb_means, std, input_shape, *args, **kwargs):
        super(YoloxDet, self).__init__(*args, **kwargs)

        in_channels = in_channels
        self.depth = depth
        self.width = width
        self.num_classes = num_classes
        self.rgb_means = rgb_means
        self.std = std
        self.input_shape = input_shape

        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
        self.model = YOLOX(backbone, head)
        self.model.head.decode_in_inference = False
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        self.model = replace_module(self.model, nn.SiLU, SiLU)
        self.model = self.model.to(self.device).eval()


    def __preprocess(self, image, input_size, mean, std, swap=(2, 0, 1),*args, **kwargs):
        shape = image.shape[:2]
        # ratio_h, ratio_w = float(384) / shape[0], float(512) / shape[1]
        ratio_h, ratio_w = input_size[0] / shape[0], input_size[1] / shape[1]
        img = cv2.resize(image, (input_size[1], input_size[0]), interpolation=cv2.INTER_AREA)
        # cv2.imwrite('img/' + '0000001' + ".jpg", img)

        img = img[:, :, ::-1]
        img = img.astype(np.float32)
        # img = np.ascontiguousarray(img, dtype=np.float32)

        img /= 255.0
        if mean is not None:
            img -= mean
        if std is not None:
            img /= std
        img = img.transpose(swap)
        img = np.expand_dims(img, axis=0)
        # ratio = (ratio_h, ratio_w)
        return img, ratio_h, ratio_w

    def __nms(self,boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def __multiclass_nms(self,boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy"""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.__nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def __postprocess(self,outputs, img_size, p6=False):

        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def get_output(self, img, detection_body_thres=0.5, nms_thres=0.5, *args, **kwargs):
        img0 = img

        img, ratio_h,ratio_w = self.__preprocess(img, self.input_shape, self.rgb_means, self.std)
        img = torch.from_numpy(img).to(self.device)

        with torch.no_grad():
            output = self.model(img)

            output = output.cpu().detach().numpy()

            predictions = self.__postprocess(output,self.input_shape)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.) / ratio_w
            boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.) / ratio_h
            boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.) / ratio_w
            boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.) / ratio_h


            dets = self.__multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thres, score_thr=detection_body_thres)
            output = []
            if dets is not None:
                preidict = dets[:, :-1]

                for bbox in preidict:
                    res = {
                        "body_box": bbox[:4],
                        "body_conf": bbox[4],
                        "body_tag": None,
                        "reid": None,
                        "face_box": None,
                        "face_conf": None,
                        "face_tag": None,
                        "face_lm": None,
                        "lh_box": None,
                        "lh_conf": None,
                        "lh_tag": None,
                        "lh_lm": None,
                        "rh_box": None,
                        "rh_conf": None,
                        "rh_tag": None,
                        "rh_lm": None,
                    }
                    output.append(res)
        return  img0.shape, output

