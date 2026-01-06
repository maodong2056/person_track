import cv2
import torch
import logging
import numpy as np
from src.utils.bbox import xyxy2xcycwh, xcycwh2xyxy
from src.utils.iou_cal import iou_cal
from src.algorithm.deep_model.arch import BasicDeepModel
from src.algorithm.deep_model.detection.body_detection.centernet.model import create_model
from src.algorithm.deep_model.detection.body_detection.centernet.utils import letterbox, ctdet_post_process
from src.algorithm.baseAlgorithm import min_cost_matching_tag, min_cost_matching_openpose, \
                                        tag_cost

#from ..baseAlgorithm import AutoGammaHSVTrans
logger = logging.getLogger(__name__)


def NMS(dets, thresh):


    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]



    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]



    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])


        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)


        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]
    return temp

class DeepDetModel(BasicDeepModel):
    def __init__(self, num_layers, heads, down_ratio, head_conv,
                 cls_num, state=0, square_input=False, *args, **kwargs):
        super(DeepDetModel, self).__init__(*args, **kwargs)
        self.model = create_model(int(num_layers), eval(heads), int(head_conv), state=state)
        self.down_ratio = down_ratio # int(down_ratio)
        self.cls_num = int(cls_num)
        self.model = self.model.to(self.device).eval()
        if square_input ==True:
            self.img_size = (512, 512)
        else:
            self.img_size = (512, 384)

    def __preprocess(self, img, *args, **kwargs):
        # Padded resize
        img0 = img.copy()
        shape = img.shape[:2]
        ratio_h, ratio_w = float(self.img_size[1]) / shape[0], float(self.img_size[0]) / shape[1]
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        # BGR2RGB && Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = img.astype(np.float32)
        #img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img0, img, ratio_h, ratio_w

    def __preprocess_pad(self, img, *args, **kwargs):
        # Padded resize
        img0 = img.copy()
        img, ratio, dw, dh = letterbox(img, height=self.img_size[1], width=self.img_size[0])

        # BGR2RGB && Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        # img = img.transpose(2, 0, 1)
        img = img.astype(np.float32)
        #img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img0, img, ratio, dw, dh

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

    def body_postprocess(self, hm, reg, reid_feat, wh, ratios, dw, dh, tag, thresh=0.4):
        clses, regs, whs, tags, reid_feat = hm, reg.cpu().numpy(), wh.cpu().numpy(), \
                                            tag.cpu().numpy(), reid_feat.cpu().numpy()
        # clses: (b,c,h,w)
        # regs:  (b,2,h,w)
        bboxes = []

        box_with_feat = []


        for cls, reg, wh, tag in zip(clses, regs, whs,tags):
            index = np.where(cls >= thresh)
            ratio = ratios
            score = np.array(cls[index])

            ctx, cty = index[-1], index[-2]

            body_tag = tag[0, cty, ctx]

            reid_feature = reid_feat[0, :, cty, ctx]


            # =====

            w, h = wh[0, cty, ctx], wh[1, cty, ctx]
            off_x, off_y = reg[0, cty, ctx], reg[1, cty, ctx]

            ctx = np.array(ctx) + np.array(off_x)
            cty = np.array(cty) + np.array(off_y)



            x1, x2 = ctx - np.array(w) / 2, ctx + np.array(w) / 2
            y1, y2 = cty - np.array(h) / 2, cty + np.array(h) / 2
            x1, y1, x2, y2 = (x1 * self.down_ratio - dw) / ratio[1], (y1*self.down_ratio - dh) / ratio[0], (x2 * self.down_ratio - dw) / ratio[1], (y2 * self.down_ratio - dh) / ratio[0]
            bbox = np.stack((x1, y1, x2, y2, score, body_tag), axis=1).tolist()
            # ===============================
            # print('score',score)

            for k, i_box in enumerate(bbox):

                box_feat = i_box+reid_feature[k].tolist()

                box_with_feat.append(box_feat)

            bbox = box_with_feat

            bbox = sorted(bbox, key=lambda x: x[4], reverse=True)
            bboxes.append(bbox)

        return bboxes

    def face_postprocess(self, fhm, freg, fwh, flm, ratios ,dw,dh ,tag, thresh=0.3):
        fclses, fregs, fwhs, flms, tags = fhm, freg.cpu().numpy(), fwh.cpu().numpy(), flm.cpu().numpy(), tag.cpu().numpy()
        # clses: (b,c,h,w)
        # regs:  (b,2,h,w)
        face_bboxes = []

        for cls, reg, wh, flm, tag in zip(fclses, fregs, fwhs, flms, tags):
            index = np.where(cls >= thresh)
            ratio = ratios
            score = np.array(cls[index])

            ctx, cty = index[-1], index[-2]

            face_tag = tag[1, cty, ctx]

            flm_c = flm[:, cty, ctx]

            flm0 = ((flm_c[0] + ctx) * self.down_ratio - dw) / ratio[1]
            flm1 = ((flm_c[1] + cty) * self.down_ratio - dh) / ratio[0]
            flm2 = ((flm_c[2] + ctx) * self.down_ratio - dw) / ratio[1]
            flm3 = ((flm_c[3] + cty) * self.down_ratio - dh) / ratio[0]
            flm4 = ((flm_c[4] + ctx) * self.down_ratio - dw) / ratio[1]
            flm5 = ((flm_c[5] + cty) * self.down_ratio - dh) / ratio[0]
            flm6 = ((flm_c[6] + ctx) * self.down_ratio - dw) / ratio[1]
            flm7 = ((flm_c[7] + cty) * self.down_ratio - dh) / ratio[0]
            flm8 = ((flm_c[8] + ctx) * self.down_ratio - dw) / ratio[1]
            flm9 = ((flm_c[9] + cty) * self.down_ratio - dh) / ratio[0]

            w, h = wh[0, cty, ctx], wh[1, cty, ctx]
            off_x, off_y = reg[0, cty, ctx], reg[1, cty, ctx]
            ctx = np.array(ctx) + np.array(off_x)
            cty = np.array(cty) + np.array(off_y)
            x1, x2 = ctx - np.array(w) / 2, ctx + np.array(w) / 2
            y1, y2 = cty - np.array(h) / 2, cty + np.array(h) / 2
            x1, y1, x2, y2 = (x1 * self.down_ratio -dw) / ratio[1], (y1 * self.down_ratio -dh) / ratio[0], (x2 * self.down_ratio-dw) / ratio[1], (y2 * self.down_ratio-dh)/ratio[0]
            bbox = np.stack((x1, y1, x2, y2, score,
                             flm0, flm1, flm2, flm3, flm4, flm5, flm6, flm7, flm8, flm9,
                             face_tag), axis=1).tolist()
            bbox = sorted(bbox, key=lambda x: x[4], reverse=True)
            face_bboxes.append(bbox)

        return face_bboxes

    def new_body_postprocess(self, bhm, btblr, reid_feat, ratios, dw, dh, tag, thresh=0.4,
                            ):
        bclses, btblrs, tags, reid_feat = bhm, btblr.cpu().numpy(), tag.cpu().numpy(), reid_feat.cpu().numpy()

        bboxes = []
        box_with_feat = []
        for cls, tblr, tag in zip(bclses, btblrs, tags):
            index = np.where(cls >= thresh)
            ratio = ratios
            score = np.array(cls[index])

            ctx, cty = index[-1], index[-2]
            # ====
            body_tag = tag[0, cty, ctx]

            reid_feature = reid_feat[0, :, cty, ctx]

            x1, y1 = np.array(ctx) + np.array(tblr[0, cty, ctx]), np.array(cty) + np.array(tblr[1, cty, ctx])
            x2, y2 = np.array(ctx) + np.array(tblr[2, cty, ctx]), np.array(cty) + np.array(tblr[3, cty, ctx])
            x1, y1, x2, y2 = \
                (x1 * self.down_ratio - dw) / ratio[1], \
                (y1 *  self.down_ratio - dh) / ratio[0], \
                (x2 *  self.down_ratio - dw) / ratio[1], \
                (y2 *  self.down_ratio - dh) / ratio[0]
            ctx = (ctx *  self.down_ratio - dw) / ratio[1]
            cty = (cty *  self.down_ratio - dh) / ratio[0]

            bbox = np.stack((x1, y1, x2, y2, score, body_tag), axis=1).tolist()

            for k, i_box in enumerate(bbox):
                box_feat = i_box + reid_feature[k].tolist()
                box_with_feat.append(box_feat)

            bbox = box_with_feat
            bbox = sorted(bbox, key=lambda x: x[4], reverse=True)
            bboxes.append(bbox)
        return bboxes

    def new_face_postprocess(self, fhm, ftblr, flm, ratios ,dw,dh ,tag, thresh=0.3):
        fclses, ftblrs, flms, tags = fhm, ftblr.cpu().numpy(), flm.cpu().numpy(), tag.cpu().numpy()
        # clses: (b,c,h,w)
        # regs:  (b,2,h,w)
        face_bboxes = []

        for cls, tblr, flm, tag in zip(fclses, ftblrs, flms, tags):
            index = np.where(cls >= thresh)
            ratio = ratios
            score = np.array(cls[index])

            ctx, cty = index[-1], index[-2]

            face_tag = tag[1,cty, ctx]

            flm_c = flm[:, cty, ctx]

            flm0 = ((flm_c[0] + ctx) * self.down_ratio - dw) / ratio[1]
            flm1 = ((flm_c[1] + cty) * self.down_ratio - dh) / ratio[0]
            flm2 = ((flm_c[2] + ctx) * self.down_ratio - dw) / ratio[1]
            flm3 = ((flm_c[3] + cty) * self.down_ratio - dh) / ratio[0]
            flm4 = ((flm_c[4] + ctx) * self.down_ratio - dw) / ratio[1]
            flm5 = ((flm_c[5] + cty) * self.down_ratio - dh) / ratio[0]
            flm6 = ((flm_c[6] + ctx) * self.down_ratio - dw) / ratio[1]
            flm7 = ((flm_c[7] + cty) * self.down_ratio - dh) / ratio[0]
            flm8 = ((flm_c[8] + ctx) * self.down_ratio - dw) / ratio[1]
            flm9 = ((flm_c[9] + cty) * self.down_ratio - dh) / ratio[0]

            x1, y1 = np.array(ctx) + np.array(tblr[0, cty, ctx]), np.array(cty) + np.array(tblr[1, cty, ctx])
            x2, y2 = np.array(ctx) + np.array(tblr[2, cty, ctx]), np.array(cty) + np.array(tblr[3, cty, ctx])
            x1, y1, x2, y2 = \
                (x1 * self.down_ratio - dw) / ratio[1], \
                (y1 * self.down_ratio - dh) / ratio[0], \
                (x2 * self.down_ratio - dw) / ratio[1], \
                (y2 * self.down_ratio - dh) / ratio[0]
            bbox = np.stack((x1, y1, x2, y2, score,
                             flm0, flm1, flm2, flm3, flm4, flm5, flm6, flm7, flm8, flm9,
                             face_tag), axis=1).tolist()
            bbox = sorted(bbox, key=lambda x: x[4], reverse=True)
            face_bboxes.append(bbox)
        #
        # for cls, reg, wh, flm, tag in zip(fclses, ftblrs, flms, tags):
        #     index = np.where(cls >= thresh)
        #     ratio = ratios
        #     score = np.array(cls[index])
        #
        #     ctx, cty = index[-1], index[-2]
        #
        #     face_tag = tag[1, cty, ctx]
        #
        #     flm_c = flm[:, cty, ctx]
        #
        #     flm0 = ((flm_c[0] + ctx) * self.down_ratio - dw) / ratio[1]
        #     flm1 = ((flm_c[1] + cty) * self.down_ratio - dh) / ratio[0]
        #     flm2 = ((flm_c[2] + ctx) * self.down_ratio - dw) / ratio[1]
        #     flm3 = ((flm_c[3] + cty) * self.down_ratio - dh) / ratio[0]
        #     flm4 = ((flm_c[4] + ctx) * self.down_ratio - dw) / ratio[1]
        #     flm5 = ((flm_c[5] + cty) * self.down_ratio - dh) / ratio[0]
        #     flm6 = ((flm_c[6] + ctx) * self.down_ratio - dw) / ratio[1]
        #     flm7 = ((flm_c[7] + cty) * self.down_ratio - dh) / ratio[0]
        #     flm8 = ((flm_c[8] + ctx) * self.down_ratio - dw) / ratio[1]
        #     flm9 = ((flm_c[9] + cty) * self.down_ratio - dh) / ratio[0]
        #
        #     w, h = wh[0, cty, ctx], wh[1, cty, ctx]
        #     off_x, off_y = reg[0, cty, ctx], reg[1, cty, ctx]
        #     ctx = np.array(ctx) + np.array(off_x)
        #     cty = np.array(cty) + np.array(off_y)
        #     x1, x2 = ctx - np.array(w) / 2, ctx + np.array(w) / 2
        #     y1, y2 = cty - np.array(h) / 2, cty + np.array(h) / 2
        #     x1, y1, x2, y2 = (x1 * self.down_ratio -dw) / ratio[1], (y1 * self.down_ratio -dh) / ratio[0], (x2 * self.down_ratio-dw) / ratio[1], (y2 * self.down_ratio-dh)/ratio[0]
        #     bbox = np.stack((x1, y1, x2, y2, score,
        #                      flm0, flm1, flm2, flm3, flm4, flm5, flm6, flm7, flm8, flm9,
        #                      face_tag), axis=1).tolist()
        #     bbox = sorted(bbox, key=lambda x: x[4], reverse=True)
        #     face_bboxes.append(bbox)

        return face_bboxes



    def get_output(self, img, detection_face_thres=0.4,
                   detection_body_thres=0.4, preprocess_pad=False, Knum=10, *args, **kwargs):
        height, width = img.shape[0:2]

        preprocess_pad = True
        if preprocess_pad == True:
            img0, img, ratios_min, pad_w, pad_h = self.__preprocess_pad(img)

            ratios = (ratios_min, ratios_min)
        else:
            img0, img, ratio_h, ratio_w = self.__preprocess(img)
            ratios = (ratio_h, ratio_w)
            pad_w = 0
            pad_h = 0
        img = torch.from_numpy(img).to(self.device)

        with torch.no_grad():

            hm, wh, reg, fhm, fwh, flm, freg, tag, body_map_max, face_map_max, reid_feat = self.model(img)
            # hm, h_tblr,fhm, f_tblr, flm, tag, body_map_max, face_map_max, reid_feat = self.model(img)

            body_keep = (hm.cpu().numpy() - body_map_max.cpu().numpy()) + 1e-9
            body_keep = np.maximum(0, body_keep)
            body_keep = body_keep * 1e9
            hm = hm.cpu().numpy() * body_keep

            face_keep = (fhm.cpu().numpy() - face_map_max.cpu().numpy()) + 1e-9
            face_keep = np.maximum(0, face_keep)
            face_keep = face_keep * 1e9
            fhm = fhm.cpu().numpy() * face_keep

            body_bboxes = self.body_postprocess(hm, reg, reid_feat, wh, ratios,
                                                dw=pad_w, dh=pad_h, tag=tag, thresh=detection_body_thres)

            # body_bboxes = self.new_body_postprocess(hm, h_tblr, reid_feat, ratios,
            #                                               dw=pad_w, dh=pad_h, tag=tag, thresh=detection_body_thres)
            dets = np.array(body_bboxes[0])

            face_bboxes = self.face_postprocess(fhm, freg, fwh, flm, ratios,
                                                dw=pad_w, dh=pad_h, tag=tag, thresh=detection_face_thres)
            #
            # face_bboxes = self.new_face_postprocess(fhm, f_tblr, flm, ratios,
            #                                               dw=pad_w, dh=pad_h, tag=tag, thresh=detection_face_thres)
            fdets = np.array(face_bboxes[0])

            # hm, wh, reg, fhm, fwh, flm, freg, tag, maxp_hm, maxp_fhm, reid_feat = self.model(img)
            #
            # body_keep = (hm.cpu().numpy() - maxp_hm.cpu().numpy()) + 1e-9
            # body_keep = np.maximum(0, body_keep)
            # body_keep = body_keep * 1e9
            # hm = hm.cpu().numpy() * body_keep
            #
            # face_keep = (fhm.cpu().numpy() - maxp_fhm.cpu().numpy()) + 1e-9
            # face_keep = np.maximum(0, face_keep)
            # face_keep = face_keep * 1e9
            # fhm = fhm.cpu().numpy() * face_keep
            #
            # body_bboxes = self.body_postprocess(hm, reg, reid_feat, wh, ratios,
            #                                     dw=pad_w, dh=pad_h, tag=tag, thresh=detection_body_thres)
            # dets = np.array(body_bboxes[0])
            #
            # face_bboxes = self.face_postprocess(fhm, freg, fwh, flm, ratios,
            #                                     dw=pad_w, dh=pad_h, tag=tag, thresh=detection_face_thres)
            # fdets = np.array(face_bboxes[0])

        if len(dets) != 0:

            keep_dets = NMS(dets, 0.5)
            dets = dets[keep_dets]
            body_box = dets[:, :5]
            body_box[:, [0, 2]] = np.clip(body_box[:, [0, 2]], 0, width)
            body_box[:, [1, 3]] = np.clip(body_box[:, [1, 3]], 0, height)
            body_tag = dets[:, 5:6]
            body_reid = dets[:, 6:]
        else:
            body_box = np.zeros((0, 5))
            body_tag = np.zeros((0, 1))
            body_reid = np.zeros((0, 128))
        if len(fdets) != 0:
            face_box = fdets[:, :15]
            face_tag = fdets[:, 15:16]
        else:
            face_box = np.zeros((0, 15))
            face_tag = np.zeros((0, 1))
        match, um_ftags, um_btags = min_cost_matching_tag(tag_cost, 0.5, face_tag, body_tag, face_box, body_box)
        output = []
        for m in match:
            res = {
                "body_box": body_box[m[1], :4],
                "body_conf": body_box[m[1], 4],
                "body_tag": body_tag[m[1], 0],
                "reid": body_reid[m[1], :],
                "face_box": face_box[m[0], :4],
                "face_conf": face_box[m[0], 4],
                "face_tag": face_tag[m[0], 0],
                "face_lm": face_box[m[0], 5:15],
                "lh_box": None,
                "lh_conf": None,
                "lh_tag": None,
                "lh_lm": None,
                "rh_box": None,
                "rh_conf": None,
                "rh_tag": None,
                "rh_lm": None,
            }
            output.append(res)
        for ub in um_btags:
            res = {
                "body_box": body_box[ub, :4],
                "body_conf": body_box[ub, 4],
                "body_tag": body_tag[ub, 0],
                "reid": body_reid[ub, :],
                "face_box": None,
                "face_conf": None,
                "face_tag": None,
                "face_lm": None,
                "lh_box": None,
                "lh_conf": None,
                "lh_tag": None,
                "lh_lm": None,
                "rh_box": None,
                "rh_conf": None,
                "rh_tag": None,
                "rh_lm": None,
            }
            output.append(res)
            # print(1)
        # print(ae)
        # output = [img0, body_box, face_box, body_reid, body_tag, face_tag]
        return img0.shape, output


class DeepDetModelWhole(DeepDetModel):
    def __init__(self, num_layers, heads, down_ratio, head_conv,
                 cls_num, state=2, *args, **kwargs):
        super(DeepDetModel, self).__init__(*args, **kwargs)
        self.model = create_model(int(num_layers), eval(heads), int(head_conv), state=state)
        self.down_ratio = down_ratio # int(down_ratio)
        self.cls_num = int(cls_num)
        self.model = self.model.to(self.device).eval()
        self.img_size = (512, 384)

    def __preprocess(self, img, *args, **kwargs):
        # Padded resize
        img0 = img.copy()
        shape = img.shape[:2]
        ratio_h, ratio_w = float(self.img_size[1]) / shape[0], float(self.img_size[0]) / shape[1]
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        # BGR2RGB && Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = img.astype(np.float32)
        #img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img0, img, ratio_h, ratio_w

    def __preprocess_pad(self, img, *args, **kwargs):
        # Padded resize
        img0 = img.copy()
        img, ratio, dw, dh = letterbox(img, height=self.img_size[1], width=self.img_size[0])

        # BGR2RGB && Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = img.astype(np.float32)
        #img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img0, img, ratio, dw, dh

    def hand_postprocess(self, hhm, hreg, hwh, hlm, ratios, dw, dh, tag, thresh=0.3):
        hclses, hregs, hwhs, hlms, tags = hhm, hreg.cpu().numpy(), hwh.cpu().numpy(), hlm.cpu().numpy(), tag.cpu().numpy()
        # clses: (b,c,h,w)
        # regs:  (b,2,h,w)
        hand_bboxes = []

        for cls, reg, wh, hlm, tag in zip(hclses, hregs, hwhs, hlms, tags):
            index = np.where(cls >= thresh)
            ratio = ratios
            score = np.array(cls[index])

            ctx, cty = index[-1], index[-2]

            hand_tag = tag[cty, ctx]

            hlm_c = hlm[:, cty, ctx]

            hlm0 = ((hlm_c[0] + ctx) * 4 - dw) / ratio[1]
            hlm1 = ((hlm_c[1] + cty) * 4 - dh) / ratio[0]
            hlm2 = ((hlm_c[2] + ctx) * 4 - dw) / ratio[1]
            hlm3 = ((hlm_c[3] + cty) * 4 - dh) / ratio[0]
            hlm4 = ((hlm_c[4] + ctx) * 4 - dw) / ratio[1]
            hlm5 = ((hlm_c[5] + cty) * 4 - dh) / ratio[0]
            hlm6 = ((hlm_c[6] + ctx) * 4 - dw) / ratio[1]
            hlm7 = ((hlm_c[7] + cty) * 4 - dh) / ratio[0]

            w, h = wh[0, cty, ctx], wh[1, cty, ctx]
            off_x, off_y = reg[0, cty, ctx], reg[1, cty, ctx]
            ctx = np.array(ctx) + np.array(off_x)
            cty = np.array(cty) + np.array(off_y)
            x1, x2 = ctx - np.array(w) / 2, ctx + np.array(w) / 2
            y1, y2 = cty - np.array(h) / 2, cty + np.array(h) / 2
            x1, y1, x2, y2 = (x1 * 4 - dw) / ratio[1], (y1 * 4 - dh) / ratio[0], (x2 * 4 - dw) / ratio[1], (y2 * 4 - dh) / ratio[0]
            bbox = np.stack((x1, y1, x2, y2, score, hlm0, hlm1, hlm2, hlm3, hlm4, hlm5, hlm6, hlm7, hand_tag),
                            axis=1).tolist()
            bbox = sorted(bbox, key=lambda x: x[4], reverse=True)
            hand_bboxes.append(bbox)

        return hand_bboxes

    def hand_nms(self, left_hand, right_hand, iou_thres=0.3):
        # left_hand_bbox = np.array(left_hand)
        # right_hand_bbox = np.array(right_hand)
        left_hand_bbox = xyxy2xcycwh(left_hand)
        right_hand_bbox = xyxy2xcycwh(right_hand)
        right_keep = []
        del_right_keep = []
        del_left_keep = []
        left_keep = []
        if len(left_hand) != 0 and len(right_hand) != 0:
            iou_res = iou_cal(left_hand_bbox, right_hand_bbox)
            for left_idx in range(len(left_hand)):
                if (iou_res[left_idx, :] > iou_thres).any():
                    right_idx = iou_res[left_idx, :].argmax()
                    left_conf = left_hand_bbox[left_idx, 4]
                    right_conf = right_hand_bbox[right_idx, 4]
                    if left_conf > right_conf:
                        left_keep.append(left_idx)
                        del_right_keep.append(right_idx)
                    else:
                        right_keep.append(right_idx)
                        del_left_keep.append(left_idx)

        for idx in range(len(right_hand)):
            if idx not in del_right_keep and idx not in right_keep:
                right_keep.append(idx)
        for idx in range(len(left_hand)):
            if idx not in del_left_keep and idx not in left_keep:
                left_keep.append(idx)
        right_keep.sort()
        left_keep.sort()
        left_hand_bbox_remain = xcycwh2xyxy(left_hand_bbox[left_keep])
        right_hand_bbox_remain = xcycwh2xyxy(right_hand_bbox[right_keep])
        return left_hand_bbox_remain[np.newaxis, ...].tolist(), \
               right_hand_bbox_remain[np.newaxis, ...].tolist()


    def get_output(self, img, detection_face_thres=0.4,
                   detection_body_thres=0.4, Knum=10, preprocess_pad=True, *args, **kwargs):
        if preprocess_pad == True:
            img0, img, ratios_min, pad_w, pad_h = self.__preprocess_pad(img)

            ratios = (ratios_min, ratios_min)
        else:
            img0, img, ratio_h, ratio_w = self.__preprocess(img)
            ratios = (ratio_h, ratio_w)
            pad_w = 0
            pad_h = 0
        img = torch.from_numpy(img).to(self.device)

        with torch.no_grad():
            # body decode
            hm, wh, reg, flm, lhlm, rhlm, tag, map_max, reid_feat = self.model(img)

            keep = (hm.cpu().numpy() - map_max.cpu().numpy()) + 1e-9
            keep = np.maximum(0, keep)
            keep = keep * 1e9
            hm = hm.cpu().numpy() * keep

            body_bboxes = self.body_postprocess(hm[:, 0, ...], reg[:, 0:2, ...], reid_feat, wh[:, 0:2, ...], ratios,
                                                dw=pad_w, dh=pad_h, tag=tag, thresh=detection_body_thres)
            dets = np.array(body_bboxes[0])

            face_bboxes = self.face_postprocess(hm[:, 1, ...], reg[:, 2:4, ...], wh[:, 2:4, ...], flm, ratios,
                                                dw=pad_w, dh=pad_h, tag=tag, thresh=detection_face_thres)
            fdets = np.array(face_bboxes[0])
            lhand_bboxes = self.hand_postprocess(hm[:, 2, ...], reg[:, 4:6, ...], wh[:, 4:6, ...], lhlm, ratios,
                                                 dw=pad_w, dh=pad_h, tag=tag[:, 2, ...], thresh=0.25)
            # lhdets = np.array(lhand_bboxes[0])
            rhand_bboxes = self.hand_postprocess(hm[:, 3, ...], reg[:, 6:8, ...], wh[:, 6:8, ...], rhlm, ratios,
                                                 dw=pad_w, dh=pad_h, tag=tag[:, 3, ...], thresh=0.25)
            # rhdets = np.array(rhand_bboxes[0])
        lhand_bboxes, rhand_bboxes = self.hand_nms(lhand_bboxes[0], rhand_bboxes[0])
        lhdets = np.array(lhand_bboxes[0])
        rhdets = np.array(rhand_bboxes[0])
        if len(dets) != 0:
            body_box = dets[:, :5]
            body_tag = dets[:, 5:6]
            body_reid = dets[:, 6:]
        else:
            body_box = np.zeros((0, 5))
            body_tag = np.zeros((0, 1))
            body_reid = np.zeros((0, 128))
        if len(fdets) != 0:
            face_box = fdets[:, :15]
            face_tag = fdets[:, 15:16]
        else:
            face_box = np.zeros((0, 15))
            face_tag = np.zeros((0, 1))
        if len(lhdets) != 0:
            lhand_box = lhdets[:, :13]
            lhand_tag = lhdets[:, 13:14]
        else:
            lhand_box = np.zeros((0, 13))
            lhand_tag = np.zeros((0, 1))
        if len(rhdets) != 0:
            rhand_box = rhdets[:, :13]
            rhand_tag = rhdets[:, 13:14]
        else:
            rhand_box = np.zeros((0, 15))
            rhand_tag = np.zeros((0, 1))
        match, um_ftags, um_btags = min_cost_matching_tag(tag_cost, 0.5, face_tag, body_tag, face_box, body_box)
        lhmatch, um_lhtags, um_lhbtags = min_cost_matching_tag(tag_cost, 0.5, lhand_tag, body_tag, lhand_box, body_box, iou_dis=0.5, use_conf=True)
        rhmatch, um_rhtags, um_rhbtags = min_cost_matching_tag(tag_cost, 0.5, rhand_tag, body_tag, rhand_box, body_box, iou_dis=0.5, use_conf=True)

        output = []
        match = np.array(match)
        lhmatch = np.array(lhmatch)
        rhmatch = np.array(rhmatch)
        for idx in range(len(dets)):
            b_box = body_box[idx, :4]
            b_conf = body_box[idx, 4]
            b_tag = body_box[idx, 0]
            reid = body_reid[idx, :]
            f_idx = None if len(match)==0 else np.where(match[:, 1] == idx)[0]
            f_box = None if f_idx is None or len(f_idx)==0 \
                else face_box[match[f_idx[0], 0], :4]
            f_conf = None if f_idx is None or len(f_idx)==0 \
                else face_box[match[f_idx[0], 0], 4]
            f_tag = None if f_idx is None or len(f_idx)==0 \
                else face_tag[match[f_idx[0], 0], 0]
            f_lm = None if f_idx is None or len(f_idx)==0 \
                else face_box[match[f_idx[0], 0], 5:15]
            lh_idx = None if len(lhmatch)==0 else np.where(lhmatch[:, 1] == idx)[0]
            lh_box = None if lh_idx is None or len(lh_idx)==0 \
                else lhand_box[lhmatch[lh_idx[0], 0], :4]
            lh_conf = None if lh_idx is None or len(lh_idx)==0 \
                else lhand_box[lhmatch[lh_idx[0], 0], 4]
            lh_tag = None if lh_idx is None or len(lh_idx)==0 \
                else lhand_tag[lhmatch[lh_idx[0], 0], 0]
            lh_lm = None if lh_idx is None or len(lh_idx)==0 \
                else lhand_box[lhmatch[lh_idx[0], 0], 5:13]
            rh_idx = None if len(rhmatch)==0 else np.where(rhmatch[:, 1] == idx)[0]
            rh_box = None if rh_idx is None or len(rh_idx)==0 \
                else rhand_box[rhmatch[rh_idx[0], 0], :4]
            rh_conf = None if rh_idx is None or len(rh_idx)==0 \
                else rhand_box[rhmatch[rh_idx[0], 0], 4]
            rh_tag = None if rh_idx is None or len(rh_idx)==0 \
                else rhand_tag[rhmatch[rh_idx[0], 0], 0]
            rh_lm = None if rh_idx is None or len(rh_idx)==0 \
                else rhand_box[rhmatch[rh_idx[0], 0], 5:13]
            res = {
                "body_box": b_box,
                "body_conf": b_conf,
                "body_tag": b_tag,
                "reid": reid,
                "face_box": f_box,
                "face_conf": f_conf,
                "face_tag": f_tag,
                "face_lm": f_lm,
                "lh_box": lh_box,
                "lh_conf": lh_conf,
                "lh_tag": lh_tag,
                "lh_lm": lh_lm,
                "rh_box": rh_box,
                "rh_conf": rh_conf,
                "rh_tag": rh_tag,
                "rh_lm": rh_lm,
            }
            output.append(res)
        # for m in match:
        #     res = {
        #         "body_box": body_box[m[1], :4],
        #         "body_conf": body_box[m[1], 4],
        #         "body_tag": body_tag[m[1], 0],
        #         "reid": body_reid[m[1], :],
        #         "face_box": face_box[m[0], :4],
        #         "face_conf": face_box[m[0], 4],
        #         "face_tag": face_tag[m[0], 0],
        #         "face_lm": face_box[m[0], 5:15]
        #     }
        #     output.append(res)
        # for ub in um_btags:
        #     res = {
        #         "body_box": body_box[ub, :4],
        #         "body_conf": body_box[ub, 4],
        #         "body_tag": body_tag[ub, 0],
        #         "reid": body_reid[ub, :],
        #         "face_box": None,
        #         "face_conf": None,
        #         "face_tag": None,
        #         "face_lm": None
        #     }
        #     output.append(res)
            # print(1)
        # print(ae)
        # output = [img0, body_box, face_box, body_reid, body_tag, face_tag]
        return img0.shape, output

class DeepDetModelOpenPose(DeepDetModel):
    def __init__(self, num_layers, heads, down_ratio, head_conv,
                 cls_num, *args, **kwargs):
        super(DeepDetModel, self).__init__(*args, **kwargs)
        self.model = create_model(int(num_layers), eval(heads), int(head_conv), whole=True)
        self.down_ratio = down_ratio # int(down_ratio)
        self.cls_num = int(cls_num)
        self.model = self.model.to(self.device).eval()
        self.img_size = (512, 384)

    def __preprocess(self, img, *args, **kwargs):
        # Padded resize
        img0 = img.copy()
        shape = img.shape[:2]
        ratio_h, ratio_w = float(self.img_size[1]) / shape[0], float(self.img_size[0]) / shape[1]
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        # BGR2RGB && Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = img.astype(np.float32)
        #img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img0, img, ratio_h, ratio_w

    def __preprocess_pad(self, img, *args, **kwargs):
        # Padded resize
        img0 = img.copy()
        img, ratio, dw, dh = letterbox(img, height=self.img_size[1], width=self.img_size[0])

        # BGR2RGB && Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = img.astype(np.float32)
        #img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img0, img, ratio, dw, dh

    def hand_postprocess(self, hhm, hreg, hwh, hlm, ratios, dw, dh, tag, thresh=0.3):
        hclses, hregs, hwhs, hlms, tags = hhm, hreg.cpu().numpy(), hwh.cpu().numpy(), hlm.cpu().numpy(), tag.cpu().numpy()
        # clses: (b,c,h,w)
        # regs:  (b,2,h,w)
        hand_bboxes = []
        center = []
        for cls, reg, wh, hlm, tag in zip(hclses, hregs, hwhs, hlms, tags):
            index = np.where(cls >= thresh)
            ratio = ratios
            score = np.array(cls[index])

            ctx, cty = index[-1], index[-2]
            center = np.stack((ctx, cty), axis=1)
            hand_tag = tag[cty, ctx]

            hlm_c = hlm[:, cty, ctx]

            hlm0 = ((hlm_c[0] + ctx) * 4 - dw) / ratio[1]
            hlm1 = ((hlm_c[1] + cty) * 4 - dh) / ratio[0]
            hlm2 = ((hlm_c[2] + ctx) * 4 - dw) / ratio[1]
            hlm3 = ((hlm_c[3] + cty) * 4 - dh) / ratio[0]
            hlm4 = ((hlm_c[4] + ctx) * 4 - dw) / ratio[1]
            hlm5 = ((hlm_c[5] + cty) * 4 - dh) / ratio[0]
            hlm6 = ((hlm_c[6] + ctx) * 4 - dw) / ratio[1]
            hlm7 = ((hlm_c[7] + cty) * 4 - dh) / ratio[0]

            w, h = wh[0, cty, ctx], wh[1, cty, ctx]
            off_x, off_y = reg[0, cty, ctx], reg[1, cty, ctx]
            ctx = np.array(ctx) + np.array(off_x)
            cty = np.array(cty) + np.array(off_y)
            x1, x2 = ctx - np.array(w) / 2, ctx + np.array(w) / 2
            y1, y2 = cty - np.array(h) / 2, cty + np.array(h) / 2
            x1, y1, x2, y2 = (x1 * 4 - dw) / ratio[1], (y1 * 4 - dh) / ratio[0], (x2 * 4 - dw) / ratio[1], (y2 * 4 - dh) / ratio[0]
            bbox = np.stack((x1, y1, x2, y2, score, hlm0, hlm1, hlm2, hlm3, hlm4, hlm5, hlm6, hlm7),
                            axis=1).tolist()
            # bbox = sorted(bbox, key=lambda x: x[4], reverse=True)
            hand_bboxes.append(bbox)

        return hand_bboxes, center

    def body_postprocess(self, hm, reg, reid_feat, wh, ratios, dw, dh, tag, thresh=0.4):
        clses, regs, whs, reid_feat = hm, reg.cpu().numpy(), wh.cpu().numpy(), \
                                             reid_feat.cpu().numpy()
        # clses: (b,c,h,w)
        # regs:  (b,2,h,w)
        bboxes = []

        box_with_feat = []
        center = []

        for cls, reg, wh in zip(clses, regs, whs):
            index = np.where(cls >= thresh)
            ratio = ratios
            score = np.array(cls[index])

            ctx, cty = index[-1], index[-2]

            # body_tag = tag[0, cty, ctx]

            reid_feature = reid_feat[0, :, cty, ctx]
            center = np.stack((ctx, cty), axis=1)

            w, h = wh[0, cty, ctx], wh[1, cty, ctx]
            off_x, off_y = reg[0, cty, ctx], reg[1, cty, ctx]

            ctx = np.array(ctx) + np.array(off_x)
            cty = np.array(cty) + np.array(off_y)



            x1, x2 = ctx - np.array(w) / 2, ctx + np.array(w) / 2
            y1, y2 = cty - np.array(h) / 2, cty + np.array(h) / 2
            x1, y1, x2, y2 = (x1 * 4 - dw) / ratio[1], (y1*4 - dh) / ratio[0], (x2 * 4 - dw) / ratio[1], (y2 * 4 - dh) / ratio[0]
            bbox = np.stack((x1, y1, x2, y2, score), axis=1).tolist()


            for k, i_box in enumerate(bbox):

                box_feat = i_box+reid_feature[k].tolist()

                box_with_feat.append(box_feat)

            bbox = box_with_feat

            # bbox = sorted(bbox, key=lambda x: x[4], reverse=True)
            bboxes.append(bbox)

        return bboxes, center

    def face_postprocess(self, fhm, freg, fwh, flm, ratios ,dw,dh ,tag, thresh=0.3):
        fclses, fregs, fwhs, flms, tags = fhm, freg.cpu().numpy(), fwh.cpu().numpy(), flm.cpu().numpy(), tag.cpu().numpy()
        # clses: (b,c,h,w)
        # regs:  (b,2,h,w)
        face_bboxes = []
        center = []

        for cls, reg, wh, flm, tag in zip(fclses, fregs, fwhs, flms, tags):
            index = np.where(cls >= thresh)
            ratio = ratios
            score = np.array(cls[index])

            ctx, cty = index[-1], index[-2]
            center = np.stack((ctx, cty), axis=1)

            face_tag = tag[1, cty, ctx]

            flm_c = flm[:, cty, ctx]

            flm0 = ((flm_c[0] + ctx) * self.down_ratio - dw) / ratio[1]
            flm1 = ((flm_c[1] + cty) * self.down_ratio - dh) / ratio[0]
            flm2 = ((flm_c[2] + ctx) * self.down_ratio - dw) / ratio[1]
            flm3 = ((flm_c[3] + cty) * self.down_ratio - dh) / ratio[0]
            flm4 = ((flm_c[4] + ctx) * self.down_ratio - dw) / ratio[1]
            flm5 = ((flm_c[5] + cty) * self.down_ratio - dh) / ratio[0]
            flm6 = ((flm_c[6] + ctx) * self.down_ratio - dw) / ratio[1]
            flm7 = ((flm_c[7] + cty) * self.down_ratio - dh) / ratio[0]
            flm8 = ((flm_c[8] + ctx) * self.down_ratio - dw) / ratio[1]
            flm9 = ((flm_c[9] + cty) * self.down_ratio - dh) / ratio[0]

            w, h = wh[0, cty, ctx], wh[1, cty, ctx]
            off_x, off_y = reg[0, cty, ctx], reg[1, cty, ctx]
            ctx = np.array(ctx) + np.array(off_x)
            cty = np.array(cty) + np.array(off_y)
            x1, x2 = ctx - np.array(w) / 2, ctx + np.array(w) / 2
            y1, y2 = cty - np.array(h) / 2, cty + np.array(h) / 2
            x1, y1, x2, y2 = (x1 * 4 -dw) / ratio[1], (y1 * 4 -dh) / ratio[0], (x2 * 4-dw) / ratio[1], (y2 * 4-dh)/ratio[0]
            bbox = np.stack((x1, y1, x2, y2, score,
                             flm0, flm1, flm2, flm3, flm4, flm5, flm6, flm7, flm8, flm9), axis=1).tolist()
            # bbox = sorted(bbox, key=lambda x: x[4], reverse=True)
            face_bboxes.append(bbox)

        return face_bboxes, center

    def hand_nms(self, left_hand, right_hand, lh_ct, rh_ct, iou_thres=0.3):
        # left_hand_bbox = np.array(left_hand)
        # right_hand_bbox = np.array(right_hand)
        left_hand_bbox = xyxy2xcycwh(left_hand)
        right_hand_bbox = xyxy2xcycwh(right_hand)
        right_keep = []
        del_right_keep = []
        del_left_keep = []
        left_keep = []
        if len(left_hand) != 0 and len(right_hand) != 0:
            iou_res = iou_cal(left_hand_bbox, right_hand_bbox)
            for left_idx in range(len(left_hand)):
                if (iou_res[left_idx, :] > iou_thres).any():
                    right_idx = iou_res[left_idx, :].argmax()
                    left_conf = left_hand_bbox[left_idx, 4]
                    right_conf = right_hand_bbox[right_idx, 4]
                    if left_conf > right_conf:
                        left_keep.append(left_idx)
                        del_right_keep.append(right_idx)
                    else:
                        right_keep.append(right_idx)
                        del_left_keep.append(left_idx)

        for idx in range(len(right_hand)):
            if idx not in del_right_keep and idx not in right_keep:
                right_keep.append(idx)
        for idx in range(len(left_hand)):
            if idx not in del_left_keep and idx not in left_keep:
                left_keep.append(idx)
        right_keep.sort()
        left_keep.sort()
        left_hand_bbox_remain = xcycwh2xyxy(left_hand_bbox[left_keep])
        right_hand_bbox_remain = xcycwh2xyxy(right_hand_bbox[right_keep])
        return left_hand_bbox_remain[np.newaxis, ...].tolist(), \
               right_hand_bbox_remain[np.newaxis, ...].tolist(), \
               lh_ct[left_keep], rh_ct[right_keep]


    def get_output(self, img, detection_face_thres=0.4,
                   detection_body_thres=0.4, Knum=10, preprocess_pad=True, *args, **kwargs):
        if preprocess_pad == True:
            img0, img, ratios_min, pad_w, pad_h = self.__preprocess_pad(img)

            ratios = (ratios_min, ratios_min)
        else:
            img0, img, ratio_h, ratio_w = self.__preprocess(img)
            ratios = (ratio_h, ratio_w)
            pad_w = 0
            pad_h = 0
        img = torch.from_numpy(img).to(self.device)

        with torch.no_grad():
            # body decode
            hm, wh, reg, fhm, fwh, flm, freg, tag, body_map_max, face_map_max, reid_feat = self.model(img)

            # hm, tblr, flm, lhlm, rhlm, tag, map_max, reid_feat = self.model(img)



            keep = (hm.cpu().numpy() - body_map_max.cpu().numpy()) + 1e-9
            keep = np.maximum(0, keep)
            keep = keep * 1e9
            hm = hm.cpu().numpy() * keep

            # print(hm.max())
            #
            body_bboxes, b_ct = self.body_postprocess(hm[:, 0, ...], reg[:, 0:2, ...], reid_feat, wh[:, 0:2, ...], ratios,
                                                dw=pad_w, dh=pad_h, tag=tag, thresh=detection_body_thres)

            # body_bboxes, b_ct = self.new_body_postprocess(hm[:, 0, ...], tblr[:, 0:4, ...], reid_feat, ratios,
            #                                     dw=pad_w, dh=pad_h, tag=tag, thresh=detection_body_thres)
            dets = np.array(body_bboxes[0])

            face_bboxes, f_ct = self.face_postprocess(hm[:, 1, ...], reg[:, 2:4, ...], wh[:, 2:4, ...], flm, ratios,
                                                dw=pad_w, dh=pad_h, tag=tag, thresh=detection_face_thres)

            # face_bboxes, f_ct = self.new_face_postprocess(hm[:, 1, ...], tblr[:, 4:8, ...], flm, ratios,
            #                                     dw=pad_w, dh=pad_h, tag=tag, thresh=detection_face_thres)
            fdets = np.array(face_bboxes[0])
            # lhand_bboxes, lh_ct = self.hand_postprocess(hm[:, 2, ...], reg[:, 4:6, ...], wh[:, 4:6, ...], lhlm, ratios,
            #                                      dw=pad_w, dh=pad_h, tag=tag[:, 2, ...], thresh=0.25)
            # # lhdets = np.array(lhand_bboxes[0])
            # rhand_bboxes, rh_ct = self.hand_postprocess(hm[:, 3, ...], reg[:, 6:8, ...], wh[:, 6:8, ...], rhlm, ratios,
            #                                      dw=pad_w, dh=pad_h, tag=tag[:, 3, ...], thresh=0.25)
            # # rhdets = np.array(rhand_bboxes[0])
        # lhand_bboxes, rhand_bboxes, lh_ct, rh_ct = self.hand_nms(lhand_bboxes[0], rhand_bboxes[0], lh_ct, rh_ct)
        lhdets = np.array([])
        rhdets = np.array([])
        if len(dets) != 0:
            body_box = dets[:, :5]
            # body_tag = dets[:, 5:6]
            body_reid = dets[:, 5:]
        else:
            body_box = np.zeros((0, 5))
            # body_tag = np.zeros((0, 1))
            body_reid = np.zeros((0, 128))
        if len(fdets) != 0:
            face_box = fdets[:, :15]
            # face_tag = fdets[:, 15:16]
        else:
            face_box = np.zeros((0, 15))
            # face_tag = np.zeros((0, 1))
        if len(lhdets) != 0:
            lhand_box = lhdets[:, :13]
            # lhand_tag = lhdets[:, 13:14]
        else:
            lhand_box = np.zeros((0, 13))
            # lhand_tag = np.zeros((0, 1))
        if len(rhdets) != 0:
            rhand_box = rhdets[:, :13]
            # rhand_tag = rhdets[:, 13:14]
        else:
            rhand_box = np.zeros((0, 15))
            # rhand_tag = np.zeros((0, 1))
        tag = tag.cpu().numpy()
        match, um_ftags, um_btags = min_cost_matching_openpose(b_ct, f_ct, tag[:, [0, 1]], face_box, body_box)
        lhmatch, um_lhtags, um_lhbtags = min_cost_matching_openpose(b_ct, lh_ct, tag[:, [2, 3]], lhand_box, body_box)
        rhmatch, um_rhtags, um_rhbtags = min_cost_matching_openpose(b_ct, rh_ct, tag[:, [4, 5]], rhand_box, body_box)
        lhmatch, um_lhtags, um_lhbtags = min_cost_matching_tag(tag_cost, 0.5, lhand_tag, body_tag, lhand_box, body_box, iou_dis=0.5, use_conf=True)
        rhmatch, um_rhtags, um_rhbtags = min_cost_matching_tag(tag_cost, 0.5, rhand_tag, body_tag, rhand_box, body_box, iou_dis=0.5, use_conf=True)

        output = []
        match = np.array(match)
        # lhmatch = np.array(lhmatch)
        # rhmatch = np.array(rhmatch)
        for idx in range(len(dets)):
            b_box = body_box[idx, :4]
            b_conf = body_box[idx, 4]
            # b_tag = body_box[idx, 0]
            reid = body_reid[idx, :]
            f_idx = None if len(match)==0 else np.where(match[:, 1] == idx)[0]
            f_box = None if f_idx is None or len(f_idx)==0 \
                else face_box[match[f_idx[0], 0], :4]
            f_conf = None if f_idx is None or len(f_idx)==0 \
                else face_box[match[f_idx[0], 0], 4]
            # f_tag = None if f_idx is None or len(f_idx)==0 \
            #     else face_tag[match[f_idx[0], 0], 0]
            f_lm = None if f_idx is None or len(f_idx)==0 \
                else face_box[match[f_idx[0], 0], 5:15]
            lh_idx = None if len(lhmatch)==0 else np.where(lhmatch[:, 1] == idx)[0]
            lh_box = None if lh_idx is None or len(lh_idx)==0 \
                else lhand_box[lhmatch[lh_idx[0], 0], :4]
            lh_conf = None if lh_idx is None or len(lh_idx)==0 \
                else lhand_box[lhmatch[lh_idx[0], 0], 4]
            lh_tag = None if lh_idx is None or len(lh_idx)==0 \
                else lhand_tag[lhmatch[lh_idx[0], 0], 0]
            lh_lm = None if lh_idx is None or len(lh_idx)==0 \
                else lhand_box[lhmatch[lh_idx[0], 0], 5:13]
            rh_idx = None if len(rhmatch)==0 else np.where(rhmatch[:, 1] == idx)[0]
            rh_box = None if rh_idx is None or len(rh_idx)==0 \
                else rhand_box[rhmatch[rh_idx[0], 0], :4]
            rh_conf = None if rh_idx is None or len(rh_idx)==0 \
                else rhand_box[rhmatch[rh_idx[0], 0], 4]
            # rh_tag = None if rh_idx is None or len(rh_idx)==0 \
            #     else rhand_tag[rhmatch[rh_idx[0], 0], 0]
            rh_lm = None if rh_idx is None or len(rh_idx)==0 \
                else rhand_box[rhmatch[rh_idx[0], 0], 5:13]
            res = {
                "body_box": b_box,
                "body_conf": b_conf,
                # "body_tag": b_tag,
                "reid": reid,
                "face_box": f_box,
                "face_conf": f_conf,
                "face_tag": f_tag,
                "face_lm": f_lm,
                "lh_box": lh_box,
                "lh_conf": lh_conf,
                # "lh_tag": lh_tag,
                "lh_lm": lh_lm,
                "rh_box": rh_box,
                "rh_conf": rh_conf,
                "rh_tag": rh_tag,
                "rh_lm": rh_lm,
            }
            output.append(res)
        # for m in match:
        #     res = {
        #         "body_box": body_box[m[1], :4],
        #         "body_conf": body_box[m[1], 4],
        #         "body_tag": body_tag[m[1], 0],
        #         "reid": body_reid[m[1], :],
        #         "face_box": face_box[m[0], :4],
        #         "face_conf": face_box[m[0], 4],
        #         "face_tag": face_tag[m[0], 0],
        #         "face_lm": face_box[m[0], 5:15]
        #     }
        #     output.append(res)
        # for ub in um_btags:
        #     res = {
        #         "body_box": body_box[ub, :4],
        #         "body_conf": body_box[ub, 4],
        #         "body_tag": body_tag[ub, 0],
        #         "reid": body_reid[ub, :],
        #         "face_box": None,
        #         "face_conf": None,
        #         "face_tag": None,
        #         "face_lm": None
        #     }
        #     output.append(res)
            # print(1)
        # print(ae)
        # output = [img0, body_box, face_box, body_reid, body_tag, face_tag]
        return img0.shape, output