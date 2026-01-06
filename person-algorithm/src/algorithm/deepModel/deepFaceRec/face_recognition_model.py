import torch
from PIL import Image
import numpy as np
import onnxruntime as ort
import os
from src.algorithm.baseAlgorithm.face_align_trans import get_reference_facial_points, warp_and_crop_face
# from collections import OrderedDict

# from sklearn.externals import joblib


class FaceRecModel():
    def __init__(self, conf, device, family_limit=None, update=False, use_svm=False):
        self.conf = conf
        if device == "gpu":
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = ort.InferenceSession(self.conf.load_model)
        self.input_name = self.model.get_inputs()[0].name  # 'data'
        self.outputs = self.model.get_outputs()[0].name  # 'fc1'
        self.refrence = get_reference_facial_points(default_square=True)
        self.use_svm = use_svm
        # if self.use_svm:
        #     self.svm = joblib.load("./weights/face_svm.m")
        if not update:
            self.targets, self.names = self.load_facebank()
            if family_limit is not None:
                fam_num = [int(x.split("_")[0].split("family")[1]) if x != "Unknown" else x for x in self.names]
                fam_num = np.array(fam_num[1:])
                fam_mask = (fam_num <= family_limit) \
                    if isinstance(family_limit, int) else (np.isin(fam_num, family_limit))
                self.targets = self.targets[fam_mask]
                self.names = self.names[np.concatenate([np.array([True]), fam_mask])]
            self.targets = self.targets.to(self.device)
            print('facebank loaded')

    def load_facebank(self):
        embeddings = torch.load(os.path.join(self.conf.facebank_path, 'facebank.pth'))
        names = np.load(os.path.join(self.conf.facebank_path, 'names.npy'))
        return embeddings, names

    def preporcess(self, img, landmarks):
        facial5points = [[landmarks[2*j],landmarks[2*j+1]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
        # return Image.fromarray(warped_face)
        return warped_face

    def cosin_metric(self, x1, x2):
        dot_result = (x1.unsqueeze(-1)*x2.transpose(1, 0).unsqueeze(0)).sum(dim=1)
        normal_dot = torch.norm(x1, 2, 1, True) * torch.norm(x2, 2, 1, True).transpose(1, 0)
        return 1. - dot_result / normal_dot

    def infer(self, img, result):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        faces = [self.preporcess(img, landmark[5:]) for landmark in result]
        for face in faces:
            input_face = np.expand_dims(face, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
            output = self.model.run([self.outputs], input_feed={self.input_name: input_face})[0]
            embs.append(torch.tensor(output).to(self.device))
        source_embs = torch.cat(embs)

        # diff = source_embs.unsqueeze(-1) - self.targets.transpose(1, 0).unsqueeze(0)
        # dist = torch.sum(torch.pow(diff, 2), dim=1)
        dist = self.cosin_metric(source_embs, self.targets)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.conf.threshold] = -1  # if no match, set idx to -1
        if self.use_svm:
            input_data = self.cal_X(result)
            y = self.svm.predict(input_data)
            min_idx = torch.where((torch.tensor(y)==1).to(self.device),
                                  min_idx,
                                  (torch.ones_like(min_idx)*-1).to(self.device))
        return min_idx, minimum

    def cal_X(self, points):
        d1 = points[:, 5] - points[:, 9]
        d2 = points[:, 6] - points[:, 10]
        d3 = points[:, 7] - points[:, 9]
        d4 = points[:, 8] - points[:, 10]
        d5 = points[:, 11] - points[:, 9]
        d6 = points[:, 12] - points[:, 10]
        d7 = points[:, 13] - points[:, 9]
        d8 = points[:, 14] - points[:, 10]
        s_ = (points[:, 2]-points[:, 0]) * (points[:, 3]-points[:, 1])
        distance = np.stack([d1, d2, d3, d4, d5, d6, d7, d8, s_], axis=1)
        return distance

    def get_embedding(self, img, result):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        faces = [self.preporcess(img, landmark[5:]) for landmark in result]
        for face in faces:
            input_face = np.expand_dims(face, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
            output = self.model.run([self.outputs], input_feed={self.input_name: input_face})[0]
            embs.append(torch.tensor(output).to(self.device))
        embedding = torch.cat(embs)
        return embedding, [Image.fromarray(x) for x in faces]

    def process_imgs(self, img, result):
        results, score = self.infer(img, result)
        names = []
        scores = []
        for idx, bbox in enumerate(results):
            names.append(self.names[results[idx] + 1])
            scores.append(float(score[idx].item()))
        return names, scores
