import numpy as np
import torch
import os
import logging
from torchvision.transforms import functional as F
# from src.algorithm.deep_model.arch import load_model
from src.algorithm.deep_model.arch.basic_model import BasicDeepModel
from src.algorithm.deep_model.keypoint.kps3d.posenet.model import MobilePoseNet, MobilePoseNetLite
from src.algorithm.baseAlgorithm import crop_image_by_bbox, xyxy2xywh, transform_preds, pixel2cam, cam2pixel

logger = logging.getLogger(__name__)

class Deep3DKPModel(BasicDeepModel):
    def __init__(self, joint_num: int, focal: list, princpt: list, bbox_3d_shape:list, *args, **kwargs):
        super(Deep3DKPModel, self).__init__(*args, **kwargs)
        self.model = MobilePoseNet(joint_num)
        # if is_useGpu:
        #     self.device = torch.device('cuda:{}'.format(gpu_num))
        # else:
        #     self.device = torch.device('cpu')
        self.model = self.model.to(self.device).eval()
        self.img_size = (256, 256)
        self.output_size = (64, 64)
        self.scale_rate = 1.25
        # self.mean = [0., 0., 0.]
        # self.std = [1., 1., 1.]
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.depth_dim = 64
        self.bbox_3d_shape = bbox_3d_shape
        self.focal = focal
        self.princpt = princpt
        self.activate = False

    # def load_model(self, model_dir, *args, **kwargs):
    #     if os.path.exists(model_dir):
    #         try:
    #             self.model = load_model(self.model, model_dir)
    #         except Exception as e:
    #             logger.error(e)
    #             self.activate = False
    #             return False
    #         self.model = self.model.to(self.device)
    #         self.activate = True
    #         return True
    #     else:
    #         self.activate = False
    #         return False

    def __preprocess(self, image, bboxes, *args, **kwargs):
        image = image[:, :, ::-1]
        bboxes_xywh = xyxy2xywh(bboxes)
        img, c, s = crop_image_by_bbox(image, bboxes_xywh, self.img_size, self.scale_rate)
        input_tensor = [F.normalize(F.to_tensor(im), mean=self.mean, std=self.std) for im in img]
        return input_tensor, img, c, s

    def __postprocess(self, result, centers, scales, shape, root_depth_list,
                      *args, **kwargs):
        N, K, _ = result.shape
        # W, H = shape
        # result[:, :, 0] = result[:, :, 0] / self.output_size[1] * self.img_size[1]
        # result[:, :, 1] = result[:, :, 1] / self.output_size[0] * self.img_size[0]
        result2D = []
        result3D = []
        for i in range(N):
            # result[i, :, :2] = transform_preds(
            #     result[i, :, :2], centers[i], scales[i],
            #     [self.output_size[0], self.output_size[1]], use_udp=False)
            result_2d = result.copy()
            result_2d[i, :, :2] = transform_preds(
                result[i, :, :2], centers[i], scales[i],
                [self.output_size[0], self.output_size[1]], use_udp=False)
            r_2D = np.concatenate([result_2d[i, :, :2], result_2d[i, :, 3:4]], axis=1).copy()
            result2D.append(r_2D)
            if root_depth_list is not None:
                result = result_2d
                result[i, :, 2] = (result[i, :, 2] / self.depth_dim * 2 - 1) * (self.bbox_3d_shape[0] / 2) + root_depth_list[i]
                result[i, :, :3] = pixel2cam(result[i, :, :3], self.focal, self.princpt)
            else:
                result[i, :, 0] = (result[i, :, 0] / self.depth_dim * 2 - 1) * (self.bbox_3d_shape[0] / 2)
                result[i, :, 1] = (result[i, :, 1] / self.depth_dim * 2 - 1) * (self.bbox_3d_shape[1] / 2)
                result[i, :, 2] = (result[i, :, 2] / self.depth_dim * 2 - 1) * (self.bbox_3d_shape[2] / 2)
                result[i, :, :3] = result[i, :, :3] - result[i, 0, :3]
            r_3D = result[i].copy()
            result3D.append(r_3D)

        return np.array(result2D), np.array(result3D)

    def get_output(self, image, bboxes=None, root_depth_list=None, *args, **kwargs):
        input_tensor, imgs, centers, scales = self.__preprocess(image, bboxes)
        # start_time = time.time()
        result = []
        for tensor in input_tensor:
            with torch.no_grad():
                r = self.model(tensor.unsqueeze(0).to(self.device))
            result.append(r.cpu().numpy())
        result = np.concatenate(result, axis=0)
        # end_time = time.time()
        # print(f'人体关键点检测时间:{end_time-start_time}')
        # self.model_time = end_time - start_time

        result2D, result3D = self.__postprocess(result, centers, scales,
                                            shape=(image.shape[1], image.shape[0]),
                                            root_depth_list=root_depth_list)
        return result2D, result3D

class Deep3DKPLiteModel(BasicDeepModel):
    def __init__(self, joint_num: int, focal: list, princpt: list, bbox_3d_shape:list, *args, **kwargs):
        super(Deep3DKPLiteModel, self).__init__(*args, **kwargs)
        self.model = MobilePoseNetLite(joint_num)
        # if is_useGpu:
        #     self.device = torch.device('cuda:{}'.format(gpu_num))
        # else:
        #     self.device = torch.device('cpu')
        self.model = self.model.to(self.device).eval()
        self.img_size = (256, 256)
        self.output_size = (256, 256)
        self.scale_rate = 1.25
        self.mean = [0., 0., 0.]
        self.std = [1., 1., 1.]
        self.depth_dim = 64
        self.bbox_3d_shape = bbox_3d_shape
        self.focal = focal
        self.princpt = princpt
        self.activate = False

    # def load_model(self, model_dir, *args, **kwargs):
    #     if os.path.exists(model_dir):
    #         try:
    #             self.model = load_model(self.model, model_dir)
    #         except Exception as e:
    #             logger.error(e)
    #             self.activate = False
    #             return False
    #         self.model = self.model.to(self.device)
    #         self.activate = True
    #         return True
    #     else:
    #         self.activate = False
    #         return False

    def __preprocess(self, image, bboxes, *args, **kwargs):
        image = image[:, :, ::-1]
        bboxes_xywh = xyxy2xywh(bboxes)
        img, c, s = crop_image_by_bbox(image, bboxes_xywh, self.img_size, self.scale_rate)
        input_tensor = [F.normalize(F.to_tensor(im), mean=self.mean, std=self.std) for im in img]
        return input_tensor, img, c, s

    def __postprocess(self, result, centers, scales, shape, root_depth_list,
                      *args, **kwargs):
        N, K, _ = result.shape
        # W, H = shape
        # result[:, :, 0] = result[:, :, 0] / self.output_size[1] * self.img_size[1]
        # result[:, :, 1] = result[:, :, 1] / self.output_size[0] * self.img_size[0]
        result2D = []
        result3D = []
        for i in range(N):
            # result[i, :, :2] = transform_preds(
            #     result[i, :, :2], centers[i], scales[i],
            #     [self.output_size[0], self.output_size[1]], use_udp=False)
            result[i, :, :2] = transform_preds(
                result[i, :, :2], centers[i], scales[i],
                [self.output_size[0], self.output_size[1]], use_udp=False)
            r_2D = np.concatenate([result[i, :, :2], result[i, :, 3:4]], axis=1).copy()
            result2D.append(r_2D)
            result[i, :, 2] = (result[i, :, 2] / self.depth_dim * 2 - 1) * (self.bbox_3d_shape[0] / 2) + root_depth_list[i]
            result[i, :, :3] = pixel2cam(result[i, :, :3], self.focal, self.princpt)
            r_3D = result[i].copy()
            result3D.append(r_3D)

        return np.array(result2D), np.array(result3D)

    def get_output(self, image, bboxes=None, root_depth_list=None, *args, **kwargs):
        input_tensor, imgs, centers, scales = self.__preprocess(image, bboxes)
        # start_time = time.time()
        result = []
        for tensor in input_tensor:
            with torch.no_grad():
                r = self.model(tensor.unsqueeze(0).to(self.device))
            result.append(r.cpu().numpy())
        result = np.concatenate(result, axis=0)
        # end_time = time.time()
        # print(f'人体关键点检测时间:{end_time-start_time}')
        # self.model_time = end_time - start_time

        result2D, result3D = self.__postprocess(result, centers, scales,
                                            shape=(image.shape[1], image.shape[0]),
                                            root_depth_list=root_depth_list)
        return result2D, result3D

