import numpy as np
import torch
import os
import logging
from torchvision.transforms import functional as F
from src.algorithm.deepModel.basic_model import BasicDeepModel
from src.algorithm.deepModel.deepRootNet.model import ResPoseNet
from src.algorithm.deep_model.arch import load_model

from src.algorithm.baseAlgorithm import crop_image_by_bbox, xyxy2xywh
from src.utils.pose_utils import process_bbox

logger = logging.getLogger(__name__)

class DeepRootModel(BasicDeepModel):
    def __init__(self, focal: list, princpt: list, *args, **kwargs):
        super(DeepRootModel, self).__init__(*args, **kwargs)
        self.model = ResPoseNet()
        # if is_useGpu:
        #     self.device = torch.device('cuda:{}'.format(gpu_num))
        # else:
        #     self.device = torch.device('cpu')
        self.model = self.model.to(self.device).eval()
        self.img_size = (256, 256)

        self.scale_rate = 1.25
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.bbox_real = (2000,2000)
        self.k_value = []

        self.focal = focal
        self.princpt = princpt
        self.activate = False

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

    def __preprocess(self, image, bboxes, *args, **kwargs):
        image = image[:, :, ::-1]
        bboxes_xywh = xyxy2xywh(bboxes)
        original_img_width,original_img_height = image.shape[1],image.shape[0]
        self.k_value = []
        for bbox in bboxes_xywh:

            new_bbox = process_bbox(bbox,original_img_width,original_img_height)
            k_value = np.array(
                [np.sqrt(self.bbox_real[0] * self.bbox_real[1] * self.focal[0] * self.focal[1] / (new_bbox[2] * new_bbox[3]))]).astype(
                np.float32)
            #k_value = torch.FloatTensor([k_value]).(self.device)[None, :]
            self.k_value.append(k_value)

        img, c, s = crop_image_by_bbox(image, bboxes_xywh, self.img_size, self.scale_rate)
        input_tensor = [F.normalize(F.to_tensor(im), mean=self.mean, std=self.std) for im in img]
        return input_tensor, img, c, s,


    def get_output(self, image, bboxes=None, *args, **kwargs):
        input_tensor, imgs, centers, scales = self.__preprocess(image, bboxes)
        # start_time = time.time()
        root_3d = []
        for k,tensor in enumerate(input_tensor):
            input_k_value = self.k_value[k]
            if self.device.type =='cuda':
                input_k_value = torch.FloatTensor([input_k_value]).cuda()[None, :]
            else:
                input_k_value = torch.FloatTensor([input_k_value]).cpu()[None, :]
            with torch.no_grad():
                r = self.model(tensor.unsqueeze(0).to(self.device),input_k_value)
                print(r[0, 2] / input_k_value)
            root_3d.append(r[0].cpu().numpy()[2])
            # print(root_3d)
        return root_3d

