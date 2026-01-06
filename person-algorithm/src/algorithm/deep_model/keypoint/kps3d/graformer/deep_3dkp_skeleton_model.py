import numpy as np
import torch
import os
import logging
from src.algorithm.deep_model.arch import load_model
from src.algorithm.deep_model.arch.basic_model import BasicDeepModel
from src.algorithm.deep_model.keypoint.kps3d.graformer.model.GraFormer import GraFormer
from src.utils import pixel2cam, trans_kps, pixcel_camera_transfer, cam2world
viz = True
if viz:
    from src.utils.draw_kps import draw_hkp_simple
    import cv2

logger = logging.getLogger(__name__)

class Deep3DKPSkeleModel(BasicDeepModel):
    def __init__(self, joint_num: int,        # 关键点个数
                 focal_cam1: list, princpt_cam1: list,  # 相机1内参 本相机参数,
                 focal_cam2: list, princpt_cam2: list,  # 相机2内参 hm36m参数,
                 keypoint_transfer_size: list, shift_center: list,  # 关键点转化参数
                 cam2_R: list, cam2_t: list, cam2_size: list, # 相机2外参 hm36m参数
                 dim_model: int = 64, n_layer: int = 2, n_head: int = 4, # 模型参数
                 *args, **kwargs):
        super(Deep3DKPSkeleModel, self).__init__(*args, **kwargs)
        if joint_num==16:
            self.edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                                       [0, 4], [4, 5], [5, 6],
                                       [0, 7], [7, 8], [8, 9],
                                       [8, 10], [10, 11], [11, 12],
                                       [8, 13], [13, 14], [14, 15]], dtype=torch.long)
            self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                               True, True, True, True, True, True]]]).to(self.device)
        else:
            self.edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                         [0, 4], [4, 5], [5, 6],
                         [0, 7], [7, 8], [8, 9], [9, 10],
                         [8, 11], [11, 12], [12, 13],
                         [8, 14], [14, 15], [15, 16]], dtype=torch.long)
            self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                               True, True, True, True, True, True]]]).to(self.device)
        self.joint_num = joint_num
        self.model = GraFormer(edges=self.edges, hid_dim=dim_model, coords_dim=(2, 3), n_pts=joint_num,
                                num_layers=n_layer, n_head=n_head, dropout=0., device=self.device)
        self.model = self.model.to(self.device).eval()
        self.focal_cam1 = focal_cam1
        self.princpt_cam1 = princpt_cam1
        self.focal_cam2 = focal_cam2
        self.princpt_cam2 = princpt_cam2
        self.keypoint_transfer_size = keypoint_transfer_size
        self.shift_center = shift_center
        self.cam2_R = cam2_R
        self.cam2_t = cam2_t
        self.cam2_size = cam2_size
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
        for idx, sk in enumerate(normal_skele):
            new_sk = pixcel_camera_transfer(sk, self.focal_cam1, self.princpt_cam1,
                                            self.focal_cam2, self.princpt_cam2)
            new_sk = trans_kps(new_sk, self.keypoint_transfer_size)
            center = new_sk[0]
            shift = np.array(self.shift_center) - center
            new_sk = new_sk + shift
            normal_skele[idx, :, :2] = new_sk
            if viz:
                image = np.zeros((1000, 1000, 3))
                conf = np.ones((17, 1))
                sk = np.concatenate([new_sk, conf], axis=1)
                im = draw_hkp_simple(image, sk)
                cv2.imshow("win", im)

        normal_skele[..., :2] = self.normalize_screen_coordinates(normal_skele[..., :2], w=self.cam2_size[0],
                                                                  h=self.cam2_size[1])
        return normal_skele

    def __postprocess(self, skele, pose2D, root_depth_list,
                      *args, **kwargs):
        N, K, _ = skele.shape
        skele[:, :, :] -= skele[:, :1, :]
        skele_world = cam2world(skele.astype(np.double), self.cam2_R, t=0).astype(np.float32)
        skele = skele*1000
        skele_world = skele_world *1000
        # pose3D = np.concatenate([skele, pose2D[:, :, 2:3]], axis=2)
        if root_depth_list is not None:
            pose3D = []
            for idx, ske in enumerate(skele):
                new_ske = np.concatenate([pose2D[idx][:, :2], ske[:, 2:3], pose2D[idx][:, 2:3]], axis=1)
                new_ske[:, 2] = new_ske[:, 2] + root_depth_list[idx]
                # new_ske[:, 2] = 3785
                new_ske[:, :3] = pixel2cam(new_ske, self.focal_cam1, self.princpt_cam1)
                pose3D.append(new_ske)
        else:
            pose3D = np.concatenate([skele[:, :, :2], -skele_world[:, :, 1:2], pose2D[:, :, 2:3]], axis=2)
        return np.array(pose3D)

    def get_output(self, image, skele, root_depth_list=None, *args, **kwargs):
        H, W, _ = image.shape
        skele_tensor = torch.from_numpy(self.__preprocess(skele, w=W, h=H))
        if self.joint_num == 16:
            skele2D = skele[:, [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16], :]
            skele_tensor = skele_tensor[:, [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16], :2]
        else:
            skele2D = skele[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], :]
            skele_tensor = skele_tensor[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], :2]
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