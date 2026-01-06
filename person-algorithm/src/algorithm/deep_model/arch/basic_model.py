import os
import torch
import logging
from src.algorithm.api import load_model

logger = logging.getLogger(__name__)

class BasicDeepModel(object):
    def __init__(self, is_useGpu, gpu_num, *args, **kwargs):
        if is_useGpu:
            self.device = torch.device('cuda:{}'.format(gpu_num))
        else:
            self.device = torch.device('cpu')
        self.activate = False

    def load_model(self, model_path: str, *args, **kwargs):
        if os.path.exists(model_path):
            try:
                self.model = load_model(self.model, model_path)
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

    def __preprocess(self, *args, **kwargs):
        pass

    def __postprocess(self, *args, **kwargs):
        pass

    def get_output(self, image, *args, **kwargs):
        pass


