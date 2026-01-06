import cv2
import numpy as np
import time
import torch
import os
import logging
from torchvision.transforms import functional as F
from src.algorithm.deepModel.basic_model import BasicDeepModel
from src.algorithm.deepModel.deepKeypoint.model import TopDownKP
from src.algorithm.deepModel.deepKeypoint.model import SimDR
from src.algorithm.deep_model.arch import load_model
from src.algorithm.baseAlgorithm import xyxy2xywh, crop_image_by_bbox, transform_preds
# from ..baseAlgorithm import AutoGammaHSVTrans
logger = logging.getLogger(__name__)


class DeepKPModel(BasicDeepModel):
    def __init__(self, *args, **kwargs):
        super(DeepKPModel, self).__init__(*args, **kwargs)
        self.model = TopDownKP()
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
        image = image[:, :, ::-1]
        bboxes_xywh = xyxy2xywh(bboxes)
        img, c, s = crop_image_by_bbox(image, bboxes_xywh, self.img_size, self.scale_rate)
        input_tensor = [F.normalize(F.to_tensor(im), mean=self.mean, std=self.std) for im in img]
        return input_tensor, img, c, s

    def _get_max_preds(self, heatmaps):
        """Get keypoint predictions from score maps.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

        Returns:
            tuple: A tuple containing aggregated results.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        assert isinstance(heatmaps,
                          np.ndarray), ('heatmaps should be numpy.ndarray')
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds, maxvals

    def __postprocess(self, heatmaps, centers, scales,
                      kernel=7,
                      valid_radius_factor=0.0546875,
                      *args, **kwargs):
        N, K, H, W = heatmaps.shape
        for person_heatmaps in heatmaps:
            for i, heatmap in enumerate(person_heatmaps):
                kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
        valid_radius = valid_radius_factor * H
        offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
        offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
        heatmaps = heatmaps[:, ::3, :]
        preds, maxvals = self._get_max_preds(heatmaps)
        index = preds[..., 0] + preds[..., 1] * W
        index += (W * H * np.arange(0, N * K / 3)).reshape(N, -1)
        index = index.astype(np.int).reshape(N, K // 3, 1)
        preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        for i in range(N):
            preds[i] = transform_preds(
                preds[i], centers[i], scales[i], [W, H], use_udp=True)
        return preds, maxvals

    def __mergeOutputs(self, detections, Knum):
        pass

    def load_model(self, model_dir, *args, **kwargs):
        if os.path.exists(model_dir):
            try:
                self.model = load_model(self.model, model_dir)
            except Exception as e:
                logger.error(e)
                self.activate = False
                return False
            self.model = self.model.to(self.device)
            self.activate = True
            return True
        else:
            self.activate = False
            return False


    def get_output(self, image, bboxes=None, *args, **kwargs):
        input_tensor, imgs, centers, scales = self.__preprocess(image, bboxes)
        start_time = time.time()
        heatmaps = []
        for tensor in input_tensor:
            with torch.no_grad():
                r = self.model(tensor.unsqueeze(0).to(self.device))
            heatmaps.append(r.cpu().numpy())
        heatmaps = np.concatenate(heatmaps, axis=0)
        end_time = time.time()
        # print(f'人体关键点检测时间:{end_time-start_time}')
        self.model_time = end_time - start_time

        preds, maxvals = self.__postprocess(heatmaps, centers, scales)
        return np.concatenate([preds, maxvals], axis=2)

class DeepSiamDRKPModel(DeepKPModel):
    def __init__(self, *args, **kwargs):
        super(DeepKPModel, self).__init__(*args, **kwargs)
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