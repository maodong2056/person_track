import torch
import cv2
import os
import numpy as np
import onnxruntime as ort
import logging
import time
from src.algorithm.deepModel.basic_model import BasicDeepModel
from src.algorithm.baseAlgorithm.face_align_trans import get_reference_facial_points, warp_and_crop_face

logger = logging.getLogger(__name__)


class FaceRecModel(BasicDeepModel):
    def __init__(self, gpu_num, is_useGpu=True, *args, **kwargs):
        super(FaceRecModel, self).__init__(gpu_num, is_useGpu, *args, **kwargs)
        if is_useGpu:
            self.device = torch.device('cuda:{}'.format(gpu_num))
        else:
            self.device = torch.device('cpu')
        self.img_size = (112, 112)
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        self.face_model = None
        self.input_name = None
        self.outputs = None
        self.gpu_num = gpu_num
        self.refrence = get_reference_facial_points(default_square=True)


    def load_model(self, model_path, *args, **kwargs):
        if os.path.exists(model_path):
            try:
                self.face_model = ort.InferenceSession(model_path)
                self.input_name = self.face_model.get_inputs()[0].name  # 'data'
                self.outputs = self.face_model.get_outputs()[0].name  # 'fc1'
            except Exception as e:
                logger.error(e)
                return False
            return True
        else:
            return False

    def __wrap_faces(self, img, landmarks):
        facial5points = [[landmarks[2*j],landmarks[2*j+1]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
        return warped_face

    def __preprocess(self, img, detResult, *args, **kwargs):
        '''
        :param img: input image
        :param detResult: detection result with [image, result] = [w*h, n*21] = [w*h, n*[dets, fdets, flms, id]]
                                                                              = [w*h, n*[5, 5, 10, 1]]
        :return: aligned face_images list with [n, 112*112]
        '''
        # img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        numPer = detResult.shape[0]
        # print(detResult[0][4:])
        faces = [self.__wrap_faces(img, detResult[i][4:]) for i in range(numPer)]
        # for i in range(numPer):
        #     facial5points = detResult[i][8:18]  # 8到18位为脸部landmark坐标
        #     facial5points = [[facial5points[2 * j], facial5points[2 * j + 1]] for j in range(5)]
        #     refrence = get_reference_facial_points(default_square=True)
        #     face = warp_and_crop_face(np.array(img), facial5points, refrence, crop_size=(112, 112))
        #     Faces.append(face)
        return faces

    def get_output_image(self, img, detResult = None, *args, **kwargs):
        faces = self.__preprocess(img, detResult)
        embs = []
        try:
            for face in faces:
                input_face = np.expand_dims(face, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
                output = self.face_model.run([self.outputs], input_feed={self.input_name: input_face})[0]
                embs.append(torch.tensor(output).to(self.device))
            source_embs = torch.cat(embs)
            return source_embs, faces
        except Exception as e:
            logger.error(e)
            return False

    def get_output(self, img, detResult = None, *args, **kwargs):
        faces = self.__preprocess(img, detResult)
        embs = []
        try:
            for idx, face in enumerate(faces):
                det_none = (detResult[idx]==-1).any()
                if det_none:
                    output = np.zeros((1, 512), dtype=np.float32)
                    output.fill(-1)

                else:
                    input_face = np.expand_dims(face, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
                    a = time.time()
                    output = self.face_model.run([self.outputs], input_feed={self.input_name: input_face})[0]
                    print("face cost:{:.3f}".format(time.time() - a))
                embs.append(torch.tensor(output).to(self.device))
            source_embs = torch.cat(embs)
            return source_embs
        except Exception as e:
            logger.error(e)
            return False


