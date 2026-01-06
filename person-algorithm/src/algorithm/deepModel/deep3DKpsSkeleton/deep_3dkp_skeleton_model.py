import numpy as np
import torch
import os
import logging
from src.algorithm.deep_model.arch import load_model
from src.algorithm.deepModel.basic_model import BasicDeepModel
from src.algorithm.deepModel.deep3DKpsSkeleton.model.GraFormer import GraFormer
from src.utils import pixel2cam

logger = logging.getLogger(__name__)

class Deep3DKPSkeleModel(BasicDeepModel):
    def __init__(self, joint_num: int,        # 关键点个数
                 focal: list, princpt: list,  # 相机参数
                 dim_model: int = 64, n_layer: int = 2, n_head: int = 4, # 模型参数
                 *args, **kwargs):
        super(Deep3DKPSkeleModel, self).__init__(*args, **kwargs)
        self.edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                                   [0, 4], [4, 5], [5, 6],
                                   [0, 7], [7, 8], [8, 9],
                                   [8, 10], [10, 11], [11, 12],
                                   [8, 13], [13, 14], [14, 15]], dtype=torch.long)
        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True]]]).to(self.device)
        self.joint_num = joint_num
        self.model = GraFormer(edges=self.edges, hid_dim=dim_model, coords_dim=(2, 3), n_pts=joint_num,
                          num_layers=n_layer, n_head=n_head, dropout=0., device=self.device)
        self.model = self.model.to(self.device).eval()
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

    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2

        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        return X / w * 2 - [1, h / w]

    def __preprocess(self, skele, w, h, *args, **kwargs):
        normal_skele = skele.copy()
        normal_skele[..., :2] = self.normalize_screen_coordinates(skele[..., :2], w=w, h=h)
        return normal_skele

    def __postprocess(self, skele, pose2D, root_depth_list,
                      *args, **kwargs):
        N, K, _ = skele.shape
        skele[:, :, :] -= skele[:, :1, :]
        skele = skele*1000
        pose3D = []
        for idx, ske in enumerate(skele):
            new_ske = np.concatenate([pose2D[idx], ske[:,2:3]], axis=1)
            new_ske[:,2] = new_ske[:,2] + root_depth_list[idx]
            # new_ske[:, 2] = root_depth_list[idx]
            new_ske = pixel2cam(new_ske, self.focal, self.princpt)
            pose3D.append(new_ske)
        return np.array(pose3D)

    def get_output(self, image, skele, root_depth_list=None, *args, **kwargs):
        H, W, _ = image.shape
        skele_tensor = torch.from_numpy(self.__preprocess(skele, w=W, h=H))
        skele2D = skele[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16], :]
        skele_tensor = skele_tensor[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16], :]
        result = []
        for tensor in skele_tensor:
            with torch.no_grad():
                r = self.model(tensor.unsqueeze(0).to(self.device), self.src_mask)
            result.append(r.cpu().numpy())
        result = np.concatenate(result, axis=0)
        # end_time = time.time()
        # print(f'人体关键点检测时间:{end_time-start_time}')
        # self.model_time = end_time - start_time

        result3D = self.__postprocess(result, skele2D, root_depth_list)
        return result3D