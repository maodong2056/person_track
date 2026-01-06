"""
Create by Chengqi.Lv
2020/3/23
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

from econn.models.operators.conv import act_dict
from econn.utils.visualize import add_coco_bbox
from econn.utils.debugger import Debugger
from econn.utils.transforms import get_affine_transform, affine_transform
from econn.models.loss.centernet_loss import CenterNetFocalLoss as FocalLoss
from econn.models.loss.centernet_loss import CenterNetQualityFocalLoss as QFocalLoss
from econn.models.loss.centernet_loss import RegL1Loss
from econn.models.loss.centernet_loss import _tranpose_and_gather_feat, _gather_feat

# debugger = Debugger(['PuTongMen', 'GuiZi', 'ShaFa', 'Ren',
#                      'ChuangHu', 'ZhuoZi', 'ChaJi', 'DianShiGui',
#                      'TuiLaMen', 'DianShi', 'Chuang', 'LiShiKongTiao',
#                      'BingXiang', 'ChuangTouGui', 'XieJia', 'GuaShiKongTiao',
#                      'MaTong', 'Tou', 'uchair'])
#
mean = np.array([0.40789654, 0.44719302, 0.47026115],
                dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.28863828, 0.27408164, 0.27809835],
               dtype=np.float32).reshape(1, 1, 3)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class CenterNetHead(nn.Module):
    def __init__(self,
                 input_channel,
                 num_classes,
                 head_conv,
                 strides,
                 loss_cfg,
                 norm_wh=False,
                 activation='ReLU'
                 ):
        super(CenterNetHead, self).__init__()
        self.max_objs = 128
        self.num_classes = num_classes
        self.strides = strides
        self.encoder = partial(encoder, num_classes=num_classes, stride=strides[0], norm_wh=norm_wh)  # encoder必须指向一个静态方法

        self.loss_cfg = loss_cfg
        self.norm_wh = norm_wh
        self.loss_class = FocalLoss() # TODO: add QFL to config
        # self.loss_class = QFL()
        self.loss_reg = RegL1Loss()

        self.heatmap = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, num_classes,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))

        fill_fc_weights(self.heatmap)
        self.heatmap[-1].bias.data.fill_(-2.19)

        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, 2,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)

        self.offset = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, 2,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.offset)

    def forward(self, x):
        x = x[0]
        out = (self.heatmap(x), self.wh(x), self.offset(x))
        return out

    def loss(self,
             preds,
             gt_meta,
             ):
        pred_hm, pred_wh, pred_offset = preds
        pred_hm = _sigmoid(pred_hm)
        b, c, h, w = pred_hm.shape
        # gt_hm, gt_wh, gt_reg, gt_reg_mask, gt_ind = self.encode(pred_hm, pred_wh, pred_offset, gt_meta)
        device = pred_hm.device
        gt_hm = gt_meta['encoded_gt']['gt_hm'].to(device)
        gt_wh = gt_meta['encoded_gt']['gt_wh'].to(device)
        gt_reg = gt_meta['encoded_gt']['gt_reg'].to(device)
        gt_reg_mask = gt_meta['encoded_gt']['gt_reg_mask'].to(device)
        gt_ind = gt_meta['encoded_gt']['gt_ind'].to(device)

        loss_cls = self.loss_class(pred_hm, gt_hm) * self.loss_cfg.loss_hm.weight
        loss_wh = self.loss_reg(pred_wh, gt_reg_mask, gt_ind, gt_wh) * self.loss_cfg.loss_wh.weight
        if self.norm_wh:
            loss_wh *= math.sqrt(w*h)
        loss_offset = self.loss_reg(pred_offset, gt_reg_mask, gt_ind, gt_reg) * self.loss_cfg.loss_offset.weight

        # TODO: add flood loss to config

        # b = 0.535
        # loss_cls_flood = torch.abs(loss_cls - b) + b
        #
        # loss = loss_cls_flood + loss_wh + loss_offset

        loss = loss_cls + loss_wh + loss_offset
        loss_states = dict(loss=loss,
                           hm_loss=loss_cls,
                           wh_loss=loss_wh,
                           offset_loss=loss_offset)

        return loss, loss_states

    def decode(self, preds, meta, resize_keep_ratio):
        pred_hm, pred_wh, pred_offset = preds
        pred_hm = _sigmoid(pred_hm)
        dets = ctdet_decode(pred_hm, pred_wh, pred_offset, self.norm_wh)
        dets = dets.detach().cpu().numpy()
        img_height = meta['img_info']['height'].cpu().numpy() \
            if isinstance(meta['img_info']['height'], torch.Tensor) else meta['img_info']['height']
        img_width = meta['img_info']['width'].cpu().numpy() \
            if isinstance(meta['img_info']['width'], torch.Tensor) else meta['img_info']['width']
        c = np.array([img_width / 2., img_height / 2.], dtype=np.float32).reshape(-1, 2)
        s = np.array([img_width, img_height], dtype=np.float32).reshape(-1, 2)
        dets_out = ctdet_post_process(dets.copy(),
                                      c,
                                      s,
                                      preds[0].shape[2],
                                      preds[0].shape[3],
                                      preds[0].shape[1],
                                      resize_keep_ratio=resize_keep_ratio)

        # # ------------debug----------------------
        # img = ((meta['img'][0].cpu().numpy().transpose(1, 2, 0) * std + mean)*255).astype(np.uint8)
        # # pred = debugger.gen_colormap(pred_hm[0].detach().cpu().numpy())
        # # debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(1))
        # debugger.add_img(cv2.resize(img, (1280,720)))
        # for label, bboxes in dets_out[0].items():
        #     for bbox in bboxes:
        #         if bbox[-1] > 0.3:
        #             debugger.add_coco_bbox(bbox[:-1], label-1, bbox[-1])
        # debugger.show_all_imgs(pause=True, expname='aaa')

        # # ------------Matplotlib debug----------------------
        # img = meta['img'][0].cpu().numpy().transpose(1, 2, 0) * std + mean
        # img = cv2.resize(img, (80, 80))
        # plt.cla()
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
        # plt.imshow(torch.max(pred_hm[0].detach(), dim=0)[0].cpu().numpy(), cmap='jet', vmin=0, vmax=1, alpha=0.7)
        # plt.pause(0.01)

        # TODO:直接解码出json格式
        return dets_out

    def show_result(self, img, dets, class_names, score_thres=0.3, show=True, save_path=None):
        dets = dets[0]
        for label in dets:
            for bbox in dets[label]:
                if bbox[4] > score_thres:
                    add_coco_bbox(img, bbox[:4], label-1, bbox[4], class_names)
        return img


def centernet_single_img(num_classes,
                         num_target,
                         gt_labels,
                         gt_bboxes,
                         hm_height,
                         hm_width,
                         norm_wh):
    hm = np.zeros((num_classes, hm_height, hm_width), dtype=np.float32)
    wh = np.zeros((128, 2), dtype=np.float32)
    reg = np.zeros((128, 2), dtype=np.float32)
    reg_mask = np.zeros((128), dtype=np.uint8)
    ind = np.zeros((128), dtype=np.int64)
    for k in range(min(num_target, 128)):
        bbox = gt_bboxes[k]
        cls_id = gt_labels[k]

        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_width - 0.25)  # 裁剪，把值限制在0～output_w - 1/stride之间，防止越界
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_height - 0.25)  # TODO:更好的实现方式
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]  # 输出大小下的高宽
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))  # 向上取整
            radius = max(0, int(radius))  # 取正整数
            ct = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)  # bbox中心点坐标
            ct_int = ct.astype(np.int32)  # 取整
            draw_gaussian(hm[cls_id], ct_int, radius)  # 画heatmap的高斯点
            if norm_wh:
                wh[k] = 1. * w / hm_width, 1. * h / hm_height  # GT的 w,h, 做归一化
            else:
                wh[k] = 1. * w, 1. * h

            ind[k] = ct_int[1] * hm_width + ct_int[0]  # 索引
            reg[k] = ct - ct_int  # GT offset
            reg_mask[k] = 1
    return hm, wh, reg, reg_mask, ind


def encoder(gt_meta, num_classes, stride, norm_wh):
    """
    encoder in dataloader, run per img
    :param norm_wh:
    :param stride:
    :param num_classes:
    :param gt_meta:
    :return:
    """
    _, img_height, img_width = gt_meta['img'].shape
    hm_height = img_height // stride
    hm_width = img_width // stride
    gt_labels = gt_meta['gt_labels']
    gt_bboxes = gt_meta['gt_bboxes']

    single_gt_labels = gt_labels
    single_gt_bboxes = gt_bboxes / stride  # 缩小到feature map大小
    hm, wh, reg, reg_mask, ind = centernet_single_img(num_classes,
                                                      len(single_gt_labels),
                                                      single_gt_labels,
                                                      single_gt_bboxes,
                                                      hm_height,
                                                      hm_width,
                                                      norm_wh)
    encoded_gt = dict(gt_hm=torch.from_numpy(hm),
                      gt_wh=torch.from_numpy(wh),
                      gt_reg=torch.from_numpy(reg),
                      gt_reg_mask=torch.from_numpy(reg_mask),
                      gt_ind=torch.from_numpy(ind))
    gt_meta['encoded_gt'] = encoded_gt

    # # -------------debug-----------
    # img = ((gt_meta['img'].cpu().numpy().transpose(1, 2, 0) * std + mean)*255).astype(np.uint8)
    # bboxes = single_gt_bboxes
    # labels = single_gt_labels
    # pred = debugger.gen_colormap(hm)
    # debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(1))
    # debugger.add_img(img)
    # for idx, bbox in enumerate(bboxes):
    #     debugger.add_coco_bbox(bbox*4, labels[idx])
    # debugger.show_all_imgs(pause=True, expname='aaa')
    return gt_meta


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    # print("heat pam size:\n", heat.size())
    return heat * keep


def transform_preds(coords, center, scale, output_size, resize_keep_ratio=False):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1, resize_keep_ratio=resize_keep_ratio)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    # 把score的二维图像转为一维，求前K个score,
    # 得到每个batch下各个cat的前k个值和地址
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()  # 得到每个batch下各个cat的前k个score的行
    topk_xs = (topk_inds % width).int().float()  # 得到每个batch下各个cat的前k个score的列

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)  # 对batch下求得的cat*k个值再求前k个
    topk_clses = (topk_ind // K)  # 得到这topk对应的类别
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)  # 得到最终topk在heat map上的位置
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def ctdet_decode(heat, wh, reg, norm_wh):
    K = 100
    batch, cat, height, width = heat.size()

    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    reg = _tranpose_and_gather_feat(reg, inds)  # regress local offset
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

    wh = _tranpose_and_gather_feat(wh, inds)

    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    # print("scores size:", scores.size())
    if norm_wh:
        bboxes = torch.cat([xs - wh[..., 0:1] * width / 2,
                            ys - wh[..., 1:2] * height / 2,
                            xs + wh[..., 0:1] * width / 2,
                            ys + wh[..., 1:2] * height / 2], dim=2)
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections


def ctdet_post_process(dets, c, s, h, w, num_classes, resize_keep_ratio):
    # dets: batch x max_dets x dim
    # return 1-based class det dict

    # 后处理，主要是把heatmap上预测的点，做仿射变换，变成输出大小下的点坐标
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h), resize_keep_ratio)
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h), resize_keep_ratio)
        classes = dets[i, :, -1]
        for j in range(num_classes):  # 按类别拆分结果
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret
