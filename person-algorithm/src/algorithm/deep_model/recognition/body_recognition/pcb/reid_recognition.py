
import torch
from torch.nn import functional as F
import cv2
import csv
import logging
import numpy as np
import time
from PIL import Image
from torchvision import transforms as T
from src.algorithm.deep_model.arch.basic_model import BasicDeepModel
from src.algorithm.deep_model.recognition.body_recognition.pcb.utils import load_model
from src.algorithm.deep_model.recognition.body_recognition.pcb.dense1stream_capsule import DenseNet
from src.algorithm.deep_model.recognition.body_recognition.pcb.reid_model import Mobilev2Reidnet

logger = logging.getLogger(__name__)

import os


class ReidRecModel(BasicDeepModel):
    def __init__(self, *args, **kwargs):
        super(ReidRecModel,self).__init__(*args, **kwargs)
        # if is_useGpu:
        #     self.device = torch.device('cuda:{}'.format(gpu_num))
        # else:
        #     self.device = torch.device('cpu')

        self.transform = T.Compose([
            T.Resize(size=(384,192), interpolation=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.body_model = None


    def load_model(self, model_path, *args, **kwargs):
        if os.path.exists(model_path):
            try:
                self.body_model = DenseNet(num_feature=1024, num_iteration=4)
                self.body_model = load_model(self.body_model, model_path).eval()
                self.body_model.to(self.device)
                self.body_model.eval()
            except Exception as e:
                logger.error(e)
                return False
            return True
        else:
            return False



    def __preprocess(self,img, detResult, *args, **kwargs):

        bbox = np.array(detResult)

        # convert to top left, bottom right
        #bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(img.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = img[sy:ey, sx:ex, :]
        return image

    def get_output(self,img, detResult = None ,*args,**kwargs):
        features = []
        try:
            for box in detResult:
                patch = self.__preprocess(img, box)

                #patch = Image.fromarray(patch)
                patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

                input_path = self.transform(patch).to(self.device)
                input_path = input_path.unsqueeze(0)
                with torch.no_grad():
                    a = time.time()
                    _,feat = self.body_model(input_path, output_feature='pool5')
                    #print("body cost:{:.3f}".format(time.time() - a))

                features.append(feat)
            if len(features) > 0:
                features = torch.cat(features)
            return features
        except Exception as e:
            logger.error(e)
            return False


class ReidRecMobileModel(BasicDeepModel):
    def __init__(self, *args, **kwargs):
        super(ReidRecMobileModel, self).__init__(*args, **kwargs)
        self.transform = T.Compose([
            T.Resize(size=(384, 192), interpolation=3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = Mobilev2Reidnet()
        self.model.to(self.device)
        self.model.eval()

    def __preprocess(self, img, detResult, *args, **kwargs):

        bbox = np.array(detResult)

        # convert to top left, bottom right
        #bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(img.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = img[sy:ey, sx:ex, :]
        return image

    def get_output(self, img, detResult=None, *args, **kwargs):
        features = []
        try:
            for box in detResult:
                patch = self.__preprocess(img, box)

                #patch = Image.fromarray(patch)
                patch = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

                input_path = self.transform(patch).to(self.device)
                input_path = input_path.unsqueeze(0)
                with torch.no_grad():
                    a = time.time()
                    #feat = self.model(input_path)
                    feat_o = self.model(input_path)

                    after_pcb1 = feat_o[:, :1280]
                    norm_pcb1 = F.normalize(after_pcb1)
                    after_pcb2 = feat_o[:, 1280:]
                    norm_pcb2 = F.normalize(after_pcb2)

                    feat = torch.cat((norm_pcb1, norm_pcb2), dim=1)
                    #print("body cost:{:.3f}".format(time.time() - a))

                features.append(feat)
            if len(features) > 0:
                features = torch.cat(features).cpu().numpy()
            return features
        except Exception as e:
            logger.error(e)
            return False



def main():
    data_root_path = '/raid/yangkang/family_data/'
    download_path = data_root_path + 'Family_001/'

    csvfile = open(download_path+'label.csv','r')
    reader = csv.reader(csvfile)

    for item in reader:
        #print(item)
        if item[0]=='image':
            img_path = data_root_path+item[1]+'/'+item[2]
            img = cv2.imread(img_path)
            # cv2.namedWindow('Image')
            # cv2.imshow('Image',img)
            # cv2.waitKey(0)

            center_x,center_y,box_w,box_h = float(item[9]),float(item[10]),float(item[11]),float(item[12])
            x1 = center_x - box_w/2.0
            y1 = center_y - box_h/2.0
            x2 = center_x+box_w/2.0
            y2 = center_y+box_h/2.0
            box = [x1,y1,x2,y2]
            input_box=[box]
            reid = ReidModel()
            path = os.getcwd() + '/exp/model.pth.tar'
            reid.load_model(path)

            feat = reid.get_output(img=img, detResult=input_box)
            print(feat.shape)


            #region = im.crop([x1,y1,x2,y2])
            #img_crop.show()



if __name__ == '__main__':
    main()