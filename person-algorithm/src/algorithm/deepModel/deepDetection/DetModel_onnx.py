import cv2
import torch
import os
import logging
import numpy as np
from src.algorithm.deepModel.basic_model import BasicDeepModel
from src.algorithm.deepModel.deepDetection.model import create_model, load_model
from src.algorithm.deepModel.deepDetection.utils import *

#from ..baseAlgorithm import AutoGammaHSVTrans
logger = logging.getLogger(__name__)

class DeepDetModel(BasicDeepModel):
    def __init__(self, num_layers, heads, down_ratio, head_conv,
                 is_useGpu, gpu_num, cls_num, *args, **kwargs):
        super(DeepDetModel, self).__init__()
        self.model = create_model(int(num_layers), eval(heads), int(head_conv))
        # self.max_per_image = int(Knum)
        self.down_ratio = 4 #int(down_ratio)
        # self.detection_thres = float(detection_thres)
        self.cls_num = int(cls_num)
        if is_useGpu:
            self.device = torch.device('cuda:{}'.format(gpu_num))
        else:
            self.device = torch.device('cpu')
        self.model = self.model.to(self.device).eval()
        self.img_size = (512, 384)
        self.activate = False

    def __preprocess(self, img, *args, **kwargs):
        #img = AutoGammaHSVTrans(img)
        img0 = img.copy()
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        # Padded resize
        img, _, _, _ = letterbox(img, height=self.img_size[1], width=self.img_size[0])

        # BGR2RGB && Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return img0, img

    def __postprocess(self, dets, meta, face=False):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        if not face:
            dets = ctdet_post_process(
                dets.copy(), [meta['c']], [meta['s']],
                meta['out_height'], meta['out_width'], self.cls_num)
        else:
            dets = ctdet_post_process(
                dets.copy(), [meta['c']], [meta['s']],
                meta['out_height'], meta['out_width'], self.cls_num, landmark=True)
        for j in range(1, self.cls_num + 1):
            if not face:
                dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            else:
                dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 15)
        return dets[0]

    def __mergeOutputs(self, detections, Knum):
        results = {}
        for j in range(1, self.cls_num + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.cls_num + 1)])
        if len(scores) > Knum:
            kth = len(scores) - Knum
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.cls_num + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def load_model(self, detection_model, *args, **kwargs):
        if os.path.exists(detection_model):
            try:
                self.model = load_model(self.model, detection_model)
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


    def get_output(self, img, detection_face_thres=0.5,
                   detection_body_thres=0.5, Knum=10, *args, **kwargs):
        # cv2.imwrite("1.jpg", img)
        img0, img = self.__preprocess(img)
        img = torch.from_numpy(img).to(self.device).unsqueeze(0)
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = img.shape[2]
        inp_width = img.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.down_ratio,
                'out_width': inp_width // self.down_ratio}
        with torch.no_grad():
            # body decode
            output = self.model(img)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            #id_feature = output['id']
            # id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg']
            tag = output['tag']
            dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=False,
                                    K=Knum)
            #id_feature = tranpose_and_gather_feat(id_feature, inds)
            btag = tranpose_and_gather_feat(tag[:, 0:1, ...], inds).cpu().numpy()[0]

            #id_feature = id_feature.squeeze(0)
            #id_feature = id_feature.cpu().numpy()
            # face decode
            fhm = output['fhm'].sigmoid_()
            fwh = output['fwh']
            flm = output['flm']
            freg = output['freg']
            fdets, finds = face_decode(fhm, fwh, freg=freg, flm=flm, cat_spec_fwh=False,
                                       K=Knum)
            ftag = tranpose_and_gather_feat(tag[:, 1:2, ...], finds).cpu().numpy()[0]

        # body det postprocess
        dets = self.__postprocess(dets, meta)
        dets = self.__mergeOutputs([dets], Knum)[1]
        dets[:, 0:4:2] = dets[:, 0:4:2].clip(0, width)
        dets[:, 1:4:2] = dets[:, 1:4:2].clip(0, height)

        remain_inds = dets[:, 4] > detection_body_thres
        dets = dets[remain_inds]
        #id_feature = id_feature[remain_inds]
        btag = btag[remain_inds]

        # face det postprocess
        fdets = self.__postprocess(fdets, meta, face=True)
        fdets = self.__mergeOutputs([fdets], Knum)[1]
        remain_inds = fdets[:, 4] > detection_face_thres
        fdets = fdets[remain_inds]
        ftag = ftag[remain_inds]
        fdets[:, 0:4:2] = fdets[:, 0:4:2].clip(0, width)
        fdets[:, 1:4:2] = fdets[:, 1:4:2].clip(0, height)
        output = [dets, fdets, btag, ftag]
        return output


# class DetModel():
#     def __init__(self, conf, device):
#         self.conf = conf
#         if device == "gpu":
#             self.device = torch.device('cuda')
#         else:
#             self.device = torch.device('cpu')
#         self.model = create_model(self.conf.num_layers,
#                                   self.conf.heads,
#                                   self.conf.head_conv)
#         self.model = load_model(self.model, self.conf.load_model)
#         self.model = self.model.to(self.device)
#         self.model.eval()
#         self.max_per_image = self.conf.K
#
#
#     def preprocess(self, img):
#         img0 = img.copy()
#         img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
#         # Padded resize
#         img, _, _, _ = letterbox(img0, height=self.conf.img_size[1], width=self.conf.img_size[0])
#
#         # Normalize RGB
#         img = img[:, :, ::-1].transpose(2, 0, 1)
#         img = np.ascontiguousarray(img, dtype=np.float32)
#         img /= 255.0
#         return img0, img
#
#     def post_process(self, dets, meta):
#         dets = dets.detach().cpu().numpy()
#         dets = dets.reshape(1, -1, dets.shape[2])
#         dets = ctdet_post_process(
#             dets.copy(), [meta['c']], [meta['s']],
#             meta['out_height'], meta['out_width'], self.conf.num_classes)
#         for j in range(1, self.conf.num_classes + 1):
#             dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
#         return dets[0]
#
#     def merge_outputs(self, detections):
#         results = {}
#         for j in range(1, self.conf.num_classes + 1):
#             results[j] = np.concatenate(
#                 [detection[j] for detection in detections], axis=0).astype(np.float32)
#
#         scores = np.hstack(
#             [results[j][:, 4] for j in range(1, self.conf.num_classes + 1)])
#         if len(scores) > self.max_per_image:
#             kth = len(scores) - self.max_per_image
#             thresh = np.partition(scores, kth)[kth]
#             for j in range(1, self.conf.num_classes + 1):
#                 keep_inds = (results[j][:, 4] >= thresh)
#                 results[j] = results[j][keep_inds]
#         return results
#
#     def get_output(self, img, heat=False):
#         img0, img = self.preprocess(img)
#         img = torch.from_numpy(img).to(self.device).unsqueeze(0)
#         width = img0.shape[1]
#         height = img0.shape[0]
#         inp_height = img.shape[2]
#         inp_width = img.shape[3]
#         c = np.array([width / 2., height / 2.], dtype=np.float32)
#         s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
#         meta = {'c': c, 's': s,
#                 'out_height': inp_height // self.conf.down_ratio,
#                 'out_width': inp_width // self.conf.down_ratio}
#         with torch.no_grad():
#             output = self.model(img)[-1]
#             hm = output['hm'].sigmoid_()
#             wh = output['wh']
#             id_feature = output['id']
#             # id_feature = F.normalize(id_feature, dim=1)
#
#             reg = output['reg'] if self.conf.reg_offset else None
#             dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.conf.cat_spec_wh,
#                                     K=self.conf.K)
#             id_feature = tranpose_and_gather_feat(id_feature, inds)
#
#             id_feature = id_feature.squeeze(0)
#             id_feature = id_feature.cpu().numpy()
#         dets = self.post_process(dets, meta)
#         dets = self.merge_outputs([dets])[1]
#         dets[:, 0:4:2] = dets[:, 0:4:2].clip(0, width)
#         dets[:, 1:4:2] = dets[:, 1:4:2].clip(0, height)
#
#         remain_inds = dets[:, 4] > self.conf.conf_thres
#         dets = dets[remain_inds]
#         id_feature = id_feature[remain_inds]
#         if not heat:
#             return img0, dets, id_feature
#         else:
#             return img0, dets, id_feature, hm
#
#
#
# class FaceDetModel(DetModel):
#     def __init__(self, conf, device):
#         super(FaceDetModel, self).__init__(conf, device)
#
#     def post_process(self, dets, meta):
#         dets = dets.detach().cpu().numpy()
#         dets = dets.reshape(1, -1, dets.shape[2])
#         dets = ctdet_post_process(
#             dets.copy(), [meta['c']], [meta['s']],
#             meta['out_height'], meta['out_width'], self.conf.num_classes, landmark = True)
#         for j in range(1, self.conf.num_classes + 1):
#             dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 15)
#         return dets[0]
#
#     def get_output(self, img, heat = False):
#         img0, img = self.preprocess(img)
#         img = torch.from_numpy(img).to(self.device).unsqueeze(0)
#         width = img0.shape[1]
#         height = img0.shape[0]
#         inp_height = img.shape[2]
#         inp_width = img.shape[3]
#         c = np.array([width / 2., height / 2.], dtype=np.float32)
#         s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
#         meta = {'c': c, 's': s,
#                 'out_height': inp_height // self.conf.down_ratio,
#                 'out_width': inp_width // self.conf.down_ratio}
#         with torch.no_grad():
#             output = self.model(img)[-1]
#             fhm = output['fhm'].sigmoid_()
#             fwh = output['fwh']
#             flm = output['flm']
#             freg = output['freg'] if self.conf.reg_offset else None
#
#             dets, inds = face_decode(fhm, fwh, freg=freg, flm=flm, cat_spec_fwh=self.conf.cat_spec_wh, K=self.conf.K)
#
#         dets = self.post_process(dets, meta)
#         dets = self.merge_outputs([dets])[1]
#         remain_inds = dets[:, 4] > self.conf.conf_thres
#         dets = dets[remain_inds]
#         dets[:, 0:4:2] = dets[:, 0:4:2].clip(0, width)
#         dets[:, 1:4:2] = dets[:, 1:4:2].clip(0, height)
#         if not heat:
#             return img0, dets
#         else:
#             return img0, dets, fhm
#
# class EcovasDetModel():
#     def __init__(self, conf, device):
#         self.conf = conf
#         if device == "gpu":
#             self.device = torch.device('cuda')
#         else:
#             self.device = torch.device('cpu')
#         self.model = create_model(self.conf.num_layers,
#                                   self.conf.heads,
#                                   self.conf.head_conv)
#         self.model = load_model(self.model, self.conf.load_model)
#         self.model = self.model.to(self.device)
#         self.model.eval()
#         self.max_per_image = self.conf.K
#
#
#     def preprocess(self, img):
#         img0 = img.copy()
#         img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
#         # Padded resize
#         img, _, _, _ = letterbox(img0, height=self.conf.img_size[1], width=self.conf.img_size[0])
#
#         # Normalize RGB
#         img = img[:, :, ::-1].transpose(2, 0, 1)
#         img = np.ascontiguousarray(img, dtype=np.float32)
#         img /= 255.0
#         return img0, img
#
#     def post_process(self, dets, meta, face=False):
#         dets = dets.detach().cpu().numpy()
#         dets = dets.reshape(1, -1, dets.shape[2])
#         if not face:
#             dets = ctdet_post_process(
#                 dets.copy(), [meta['c']], [meta['s']],
#                 meta['out_height'], meta['out_width'], self.conf.num_classes)
#         else:
#             dets = ctdet_post_process(
#                 dets.copy(), [meta['c']], [meta['s']],
#                 meta['out_height'], meta['out_width'], self.conf.num_classes, landmark=True)
#         for j in range(1, self.conf.num_classes + 1):
#             if not face:
#                 dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
#             else:
#                 dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 15)
#         return dets[0]
#
#     def merge_outputs(self, detections):
#         results = {}
#         for j in range(1, self.conf.num_classes + 1):
#             results[j] = np.concatenate(
#                 [detection[j] for detection in detections], axis=0).astype(np.float32)
#
#         scores = np.hstack(
#             [results[j][:, 4] for j in range(1, self.conf.num_classes + 1)])
#         if len(scores) > self.max_per_image:
#             kth = len(scores) - self.max_per_image
#             thresh = np.partition(scores, kth)[kth]
#             for j in range(1, self.conf.num_classes + 1):
#                 keep_inds = (results[j][:, 4] >= thresh)
#                 results[j] = results[j][keep_inds]
#         return results
#
#     # @timecal
#     def get_output(self, img, heat=False, tagmap=False):
#         img0, img = self.preprocess(img)
#         img = torch.from_numpy(img).to(self.device).unsqueeze(0)
#         width = img0.shape[1]
#         height = img0.shape[0]
#         inp_height = img.shape[2]
#         inp_width = img.shape[3]
#         c = np.array([width / 2., height / 2.], dtype=np.float32)
#         s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
#         meta = {'c': c, 's': s,
#                 'out_height': inp_height // self.conf.down_ratio,
#                 'out_width': inp_width // self.conf.down_ratio}
#         with torch.no_grad():
#             # body decode
#             output = self.model(img)[-1]
#             hm = output['hm'].sigmoid_()
#             wh = output['wh']
#             id_feature = output['id']
#             # id_feature = F.normalize(id_feature, dim=1)
#
#             reg = output['reg'] if self.conf.reg_offset else None
#             tag = output['tag'] if self.conf.tag_reg else None
#             dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=self.conf.cat_spec_wh,
#                                     K=self.conf.K)
#             id_feature = tranpose_and_gather_feat(id_feature, inds)
#             btag = tranpose_and_gather_feat(tag[:, 0:1, ...], inds).cpu().numpy()[0] if self.conf.tag_reg else None
#
#             id_feature = id_feature.squeeze(0)
#             id_feature = id_feature.cpu().numpy()
#             # face decode
#             fhm = output['fhm'].sigmoid_()
#             fwh = output['fwh']
#             flm = output['flm']
#             freg = output['freg'] if self.conf.reg_offset else None
#             fdets, finds = face_decode(fhm, fwh, freg=freg, flm=flm, cat_spec_fwh=self.conf.cat_spec_wh,
#                                        K=self.conf.K)
#             ftag = tranpose_and_gather_feat(tag[:, 1:2, ...], finds).cpu().numpy()[0] if self.conf.tag_reg else None
#
#         # body det postprocess
#         dets = self.post_process(dets, meta)
#         dets = self.merge_outputs([dets])[1]
#         dets[:, 0:4:2] = dets[:, 0:4:2].clip(0, width)
#         dets[:, 1:4:2] = dets[:, 1:4:2].clip(0, height)
#
#         remain_inds = dets[:, 4] > self.conf.body_conf_thres
#         dets = dets[remain_inds]
#         id_feature = id_feature[remain_inds]
#         btag = btag[remain_inds] if self.conf.tag_reg else None
#
#         # face det postprocess
#         fdets = self.post_process(fdets, meta, face=True)
#         fdets = self.merge_outputs([fdets])[1]
#         remain_inds = fdets[:, 4] > self.conf.face_conf_thres
#         fdets = fdets[remain_inds]
#         ftag = ftag[remain_inds] if self.conf.tag_reg else None
#         fdets[:, 0:4:2] = fdets[:, 0:4:2].clip(0, width)
#         fdets[:, 1:4:2] = fdets[:, 1:4:2].clip(0, height)
#
#         # import matplotlib.pyplot as plt
#         # fig = plt.figure(figsize=(6, 3), dpi=150)
#         # ax1 = fig.add_subplot(121)
#         # ax2 = fig.add_subplot(122)
#         # ax1.set_xticks([])
#         # ax1.set_yticks([])
#         # ax2.set_xticks([])
#         # ax2.set_yticks([])
#         # ax1.imshow(hm.cpu().numpy()[0, 0])
#         # ax2.imshow(fhm.cpu().numpy()[0, 0])
#         output = [img0, dets, fdets, id_feature, btag, ftag]
#         if heat:
#             output += [hm[0, 0].cpu().numpy(), fhm[0, 0].cpu().numpy()]
#         if tagmap:
#             tager = tag[0].cpu().numpy()
#             tager = (tager - tager.mean())/tager.std()
#             output += [tager[0], tager[1]]
#         return output
#         # if not heat:
#         #     return img0, dets, fdets, id_feature, btag, ftag
#         # else:
#         #     return img0, dets, fdets, id_feature, hm, fhm
