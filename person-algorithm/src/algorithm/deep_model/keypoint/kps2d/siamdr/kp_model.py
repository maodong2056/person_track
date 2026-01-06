import cv2
import numpy as np
import time
import torch
# import os
import logging
from torchvision.transforms import functional as F
from src.algorithm.deepModel.basic_model import BasicDeepModel
from src.algorithm.deep_model.keypoint.kps2d.siamdr.model import SimDR
# from src.algorithm.deep_model.arch import load_model
from src.algorithm.baseAlgorithm import crop_image_by_bbox, xyxy2xywh, transform_preds
# from ..baseAlgorithm import AutoGammaHSVTrans
logger = logging.getLogger(__name__)

class DeepSiamDRKPModel(BasicDeepModel):
    def __init__(self, *args, **kwargs):
        super(DeepSiamDRKPModel, self).__init__(*args, **kwargs)
        self.model = SimDR(*args, **kwargs)
        # if is_useGpu:
        #     self.device = torch.device('cuda:{}'.format(gpu_num))
        # else:
        #     self.device = torch.device('cpu')
        self.model = self.model.to(self.device).eval()
        self.img_size = (192, 256)
        self.scale_rate = 1.25
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.activate = False

    def __preprocess(self, image, bboxes, *args, **kwargs):
        # image = image[:, :, ::-1]
        bboxes_xywh = xyxy2xywh(bboxes)
        img, c, s = crop_image_by_bbox(image, bboxes_xywh, self.img_size, self.scale_rate)
        input_tensor = [F.normalize(F.to_tensor(im), mean=self.mean, std=self.std) for im in img]
        return input_tensor, img, c, s

    def __postprocess(self, xs, ys, centers, scales, *args, **kwargs):
        N = xs.shape[0]
        W = self.img_size[0]
        H = self.img_size[1]
        pred_x = np.argmax(xs, axis=2)
        coord_x = np.max(xs, axis=2)
        pred_y = np.argmax(ys, axis=2)
        coord_y = np.max(ys, axis=2)
        # strategies to determine the confidence of predicted location
        mask = coord_x < coord_y
        coord_x[mask] = coord_y[mask]
        maxvals = coord_x.copy()[:, :, None]
        preds = np.concatenate((pred_x[:, :, None], pred_y[:, :, None]), axis=2)
        for i in range(N):
            preds[i] = transform_preds(
                preds[i], centers[i], scales[i], [W, H], use_udp=True)
        return preds, maxvals

    def get_output(self, image, bboxes=None, *args, **kwargs):
        input_tensor, imgs, centers, scales = self.__preprocess(image, bboxes)
        start_time = time.time()
        soft_xs = []
        soft_ys = []
        for tensor in input_tensor:
            with torch.no_grad():
                soft_x, soft_y = self.model(tensor.unsqueeze(0).to(self.device))
            soft_xs.append(soft_x.cpu().numpy())
            soft_ys.append(soft_y.cpu().numpy())
        soft_xs = np.concatenate(soft_xs, axis=0)
        soft_ys = np.concatenate(soft_ys, axis=0)
        end_time = time.time()
        # print(f'人体关键点检测时间:{end_time-start_time}')
        self.model_time = end_time - start_time

        preds, maxvals = self.__postprocess(soft_xs, soft_ys, centers, scales)
        return np.concatenate([preds, maxvals], axis=2)
