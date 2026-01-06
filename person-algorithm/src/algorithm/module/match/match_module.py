import torch
import numpy as np
import logging
from scipy.optimize import linear_sum_assignment as linear_assignment
from src.algorithm.deepModel.deepFaceRec import FaceRecModel
from src.algorithm.deepModel.deepReidRec import ReidRecModel
from src.algorithm.module.match.people import _People
from src.algorithm.baseAlgorithm import iou_cal

logger = logging.getLogger(__name__)

class MatchModule(FaceRecModel, ReidRecModel):
    def __init__(self, *args, **kwargs):
        # DeepDetModel.__init__(*args, **kwargs)
        # DeepSort.__init__(*args, **kwargs)
        FaceRecModel.__init__(self, *args, **kwargs)
        ReidRecModel.__init__(self, *args, **kwargs)
        is_useGpu = kwargs["is_useGpu"]
        gpu_num = kwargs["gpu_num"]
        if is_useGpu:
            self.device = torch.device('cuda:{}'.format(gpu_num))
        else:
            self.device = torch.device('cpu')
        # super(matchModule, self).__init__(*args, **kwargs)
        self.activate = False
        self.people = _People("../user/lib/persons.json",
                             "../user/lib/facebank/names.npy",
                             "../user/lib/facebank/facebank.pth",
                             device=self.device)
        # super(DeepSort, self).__init__()

    def get_face_embedding(self, image, det_result):
        try:
            embs = FaceRecModel.get_output(self, image, det_result)
            return embs
        except Exception as e:
            logger.error(e)

    def get_face_embedding_image(self, image, det_result):
        try:
            embs, img = FaceRecModel.get_output_image(self, image, det_result)
            return embs, img
        except Exception as e:
            logger.error(e)

    def get_body_embedding(self, image, det_result):
        try:
            embs = ReidRecModel.get_output(self, image, det_result)
            return embs
        except Exception as e:
            logger.error(e)

    def load_model(self, reid_model=None, face_model=None, *args, **kwargs):
        """
        重写 load model， 通过字典传入加载模型路径
        :param model_dict: dict 包含 face_model 或者 reid_model 路径
        :param args:
        :param kwargs:
        :return: 返回加载成功与否
        """
        # for key in model_dict.keys():
        ret_face = True
        ret_reid = True
        if face_model is not None:
            ret_face = FaceRecModel.load_model(self, face_model, *args, **kwargs)
        if reid_model is not None:
            ret_reid = ReidRecModel.load_model(self, reid_model, *args, **kwargs)
        ret = ret_face & ret_reid
        self.activate = ret
        return ret

    def updatePersonBank(self):
        self.people.update_bank()

    def cosin_metric(self, x1, x2):
        try:
            dot_result = (x1.unsqueeze(-1) * x2.transpose(1, 0).unsqueeze(0)).sum(dim=1)
            normal_dot = torch.norm(x1, 2, 1, True) * torch.norm(x2, 2, 1, True).transpose(1, 0)
            return 1. - dot_result / normal_dot
        except Exception as e:
            logger.error(e)

    def euclidean_metric(self, x1, x2):
        try:
            x12, x22 = (x1*x1).sum(axis=1), (x2*x2).sum(axis=1)
            r2 = -2. * torch.matmul(x1, x2.T) + x12[:, None] + x22[None, :]
            # r2 = torch.clip(r2, 0., float(np.inf))
            return r2
        except Exception as e:
            logger.error(e)

    def min_cost_matching(self, distance_matrix, max_distance, uc_person_idx, uc_det_idx, face_dist, body_dist, iou_max):
        try:
            if len(uc_person_idx) == 0 or len(uc_det_idx) == 0:
                return [], uc_person_idx, uc_det_idx  # Nothing to match.

            distance_matrix[distance_matrix > max_distance] = max_distance + 1e-5
            row_indices, col_indices = linear_assignment(distance_matrix)
            matches,unmatched_detections, unmatched_persons = [], [], []
            for col, person_id in enumerate(uc_person_idx):
                if col not in col_indices:
                    unmatched_persons.append(person_id)
            for row, det_id in enumerate(uc_det_idx):
                if row not in row_indices:
                    unmatched_detections.append(det_id)
            for row, col in zip(row_indices, col_indices):
                person_id = uc_person_idx[col]
                det_id = uc_det_idx[row]
                if distance_matrix[row, col] > max_distance:
                    unmatched_detections.append(det_id)
                    unmatched_persons.append(person_id)
                else:
                    matches.append((person_id, det_id, face_dist[row, col], body_dist[row, col], iou_max[det_id]))
            return matches, unmatched_persons, unmatched_detections
        except Exception as e:
            logger.error(e)

    def match(self, image, det_result, use_mutiMatch=True, face_thres=0.8, reid_thres=0.8):
        try:
            body_res = np.array([det["body_box"] for det in det_result])
            face_res = np.array([np.concatenate([det["face_box"], det["face_lm"]]) if det["face_box"] is not None else np.ones(14,)*-1 for det in det_result])
            confirm_idx, unconfirm_person_idx, confirm_det_idx, unconfirm_det_idx, facebank_emb, bodybank_emb = self.people.get_embedding(det_result)
            self.people.step()
            # print(type(bank_emb))
            # print(det_result)
            # 必须要有未匹配的det和未匹配的person才能进行计算
            if len(unconfirm_det_idx) != 0 and len(unconfirm_person_idx) != 0:
                # current_face_emb = self.get_face_embedding(image, det_result[unconfirm_det_idx][:, 4:18])  # input the 4-17 face detection result
                current_face_emb = self.get_face_embedding(image, face_res[unconfirm_det_idx])
                dist_face_metric = self.cosin_metric(current_face_emb, facebank_emb).cpu().numpy()
                # face_emb_dict = dict(zip(unconfirm_det_idx, current_face_emb))  # key with det idx
                # current_body_emb = self.get_body_embedding(image, det_result[unconfirm_det_idx][:, :4])  # input the 0-4 body detection result
                current_body_emb = self.get_body_embedding(image, body_res[unconfirm_det_idx])
                dist_body_metric = self.cosin_metric(current_body_emb, bodybank_emb).cpu().numpy()
                print(dist_body_metric)
                print(unconfirm_person_idx)
                # body_emb_dict = dict(zip(unconfirm_det_idx, current_body_emb))  # key with det idx
                alpha = 1 - dist_face_metric * dist_face_metric * dist_face_metric/\
                        (dist_face_metric * dist_face_metric * dist_face_metric +
                         dist_body_metric * dist_body_metric * dist_body_metric)
                dist_metric = alpha * dist_face_metric + (1. - alpha) * dist_body_metric
                emb_dict = dict(zip(unconfirm_det_idx, zip(current_face_emb, current_body_emb)))
                iou = iou_cal(body_res, body_res)
                iou_max = (iou * np.invert(np.eye(N=iou.shape[0],dtype=bool)).astype(np.float)).max(axis=1)
                matches, unmatched_persons, unmatched_detections = \
                    self.min_cost_matching(dist_metric, face_thres, unconfirm_person_idx, unconfirm_det_idx,
                                           dist_face_metric, dist_body_metric, iou_max)
                print(matches)
            else:
                matches, unmatched_persons, unmatched_detections = [], unconfirm_person_idx, []
                emb_dict = {}
                iou_max = None
            self.people.update(det_result, emb_dict, confirm_idx, confirm_det_idx,
                               matches, unmatched_persons, unmatched_detections, iou_max)
        except Exception as e:
            logger.error(e)
        # print(1)
        # print(current_emb.shape)

    # def updateBank(self, facebankDir):
    def get_confirmed_person(self):
        persons = self.people.get_confirmed_person()
        return persons


    def clear(self):
        self.people.clear()


