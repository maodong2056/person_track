"""
Create by Chengqi.Lv
2020/4/13
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import math

from econn.utils.visualize import add_coco_bbox
from econn.utils.debugger import Debugger
from econn.utils.transforms import distance2bbox
from econn.models.taskheads.centernet_head import ctdet_post_process
from econn.models.loss.centernet_loss import CenterNetFocalLoss as FocalLoss
from econn.models.loss.centernet_loss import _tranpose_and_gather_feat, _gather_feat
from econn.models.loss.diou_loss import DIOULoss

# debugger = Debugger(['PuTongMen', 'GuiZi', 'ShaFa', 'Ren',
#                      'ChuangHu', 'ZhuoZi', 'ChaJi', 'DianShiGui',
#                      'TuiLaMen', 'DianShi', 'Chuang', 'LiShiKongTiao',
#                      'BingXiang', 'ChuangTouGui', 'XieJia', 'GuaShiKongTiao',
#                      'MaTong', 'Tou', 'uchair'])
#
# mean = np.array([0.40789654, 0.44719302, 0.47026115],
#                 dtype=np.float32).reshape(1, 1, 3)
# std = np.array([0.28863828, 0.27408164, 0.27809835],
#                dtype=np.float32).reshape(1, 1, 3)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class CtFCOSHead(nn.Module):
    def __init__(self,
                 input_channel,
                 num_classes,
                 head_conv,
                 strides,
                 loss_cfg
                 ):
        super(CtFCOSHead, self).__init__()
        self.max_objs = 128
        self.num_classes = num_classes
        self.strides = strides
        self.encoder = partial(encoder, num_classes=num_classes, stride=strides[0])  # encoder必须指向一个静态方法

        self.loss_cfg = loss_cfg
        self.loss_class = FocalLoss()
        self.loss_bbox = DIOULoss()  # TODO: 支持更多种IOU Loss

        self.heatmap = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))

        fill_fc_weights(self.heatmap)
        self.heatmap[-1].bias.data.fill_(-2.19)

        self.dis = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 4,
                      kernel_size=1, stride=1,
                      padding=0, bias=True),
            nn.ReLU(inplace=True),  # TODO: 验证一下到底是按FCOS的指数输出还是按PolarMask的ReLu输出比较好？？？
        )
        fill_fc_weights(self.dis)
        self.dis[-2].bias.data.fill_(0.1)

    def forward(self, x):
        x = x[0]
        out = (self.heatmap(x), self.dis(x))
        return out

    def loss(self,
             preds,
             gt_meta,
             ):
        pred_hm, pred_dis = preds[0]
        pred_hm = _sigmoid(pred_hm)

        device = pred_hm.device
        gt_hm = gt_meta['encoded_gt']['gt_hm'].to(device)
        gt_dense_dis = gt_meta['encoded_gt']['gt_dense_dis'].to(device)
        gt_dense_weight = gt_meta['encoded_gt']['gt_dense_weight'].to(device)

        loss_cls = self.loss_class(pred_hm, gt_hm) * self.loss_cfg.loss_hm.weight

        num_pos, pos_dis_preds, pos_dis_targets, pos_weights, pos_points = \
            get_pos_feature(pred_dis,
                            gt_dense_weight,
                            gt_dense_dis)
        if num_pos > 0:
            pred_bbox = distance2bbox(pos_points, pos_dis_preds)
            target_bbox = distance2bbox(pos_points, pos_dis_targets)
            loss_bbox = self.loss_bbox(pred_bbox, target_bbox, pos_weights, pos_weights.sum())
        else:
            loss_bbox = pos_dis_preds.sum()
        loss_bbox = loss_bbox * self.loss_cfg.loss_bbox.weight
        loss = loss_cls + loss_bbox
        loss_states = dict(loss=loss,
                           hm_loss=loss_cls,
                           bbox_loss=loss_bbox,
                           )
        return loss, loss_states

    def decode(self, preds, meta, resize_keep_ratio):
        pred_hm, pred_dis = preds
        pred_hm = _sigmoid(pred_hm)
        dets = ctfcos_decode(pred_hm, pred_dis)
        dets = dets.detach().cpu().numpy()
        img_height = meta['img_info']['height'].cpu().numpy() \
            if isinstance(meta['img_info']['height'], torch.Tensor) else meta['img_info']['height']
        img_width = meta['img_info']['width'].cpu().numpy() \
            if isinstance(meta['img_info']['width'], torch.Tensor) else meta['img_info']['width']
        c = np.array([img_width / 2., img_height / 2.], dtype=np.float32).reshape(-1, 2)
        s = np.array([img_width, img_height], dtype=np.float32).reshape(-1, 2)
        # 由于decode成了bbox， 所以与centernet共用同一个后处理
        dets_out = ctdet_post_process(
            dets.copy(), c, s,
            preds[0].shape[2], preds[0].shape[3], preds[0].shape[1], resize_keep_ratio=resize_keep_ratio)

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

        # TODO:直接解码出json格式
        return dets_out

    def show_result(self, img, dets, class_names, score_thres=0.3, show=True, save_path=None):
        dets = dets[0]
        for label in dets:
            for bbox in dets[label]:
                if bbox[4] > score_thres:
                    add_coco_bbox(img, bbox[:4], label - 1, bbox[4], class_names)
        return img


def encoder(gt_meta, num_classes, stride):
    """
    encoder in dataloader, run per img
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
    # TODO: 支持归一化宽高
    single_gt_labels = gt_labels
    single_gt_bboxes = gt_bboxes / stride  # 缩小到feature map大小
    hm, dense_dis, dense_weight = ctfcos_single_img(num_classes,
                                                    len(single_gt_labels),
                                                    single_gt_labels,
                                                    single_gt_bboxes,
                                                    hm_height,
                                                    hm_width)
    encoded_gt = dict(gt_hm=torch.from_numpy(hm),
                      gt_dense_dis=torch.from_numpy(dense_dis),
                      gt_dense_weight=torch.from_numpy(dense_weight))
    gt_meta['encoded_gt'] = encoded_gt
    return gt_meta


def ctfcos_single_img(num_classes,
                      num_target,
                      gt_labels,
                      gt_bboxes,
                      hm_height,
                      hm_width):
    hm = np.zeros((num_classes, hm_height, hm_width), dtype=np.float32)
    dense_weight = np.zeros((hm_height, hm_width), dtype=np.float32)
    dense_dis = np.zeros((4, hm_height, hm_width), dtype=np.float32)
    for k in range(min(num_target, 128)):
        bbox = gt_bboxes[k]
        cls_id = gt_labels[k]
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_width - 1)  # 裁剪，把值限制在0～output_w - 1之间，防止越界
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_height - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]  # 输出大小下的高宽
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))  # 向上取整
            radius = max(0, int(radius))  # 取正整数
            ct = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)  # bbox中心点坐标
            ct_int = ct.astype(np.int32)  # 取整
            draw_dense_dis_single_target(h, w, dense_dis, dense_weight, ct_int, bbox)
            draw_gaussian(hm[cls_id], ct_int, radius)  # 画heatmap的高斯点
    return hm, dense_dis, dense_weight


def get_fcos_dists(c_x, c_y, bbox):
    left = c_x - bbox[0]
    right = bbox[2] - c_x
    top = c_y - bbox[1]
    bottom = bbox[3] - c_y
    distances = np.array([left, top, right, bottom], dtype=np.float32)
    return distances


def draw_dense_dis_single_target(h, w, dense_dis, dense_weight, ct_int, bbox):
    this_dense_weight = np.zeros_like(dense_weight)

    radius = gaussian_radius((math.ceil(h), math.ceil(w)))  # 向上取整
    radius = max(0, int(radius))  # 取正整数
    draw_gaussian(dense_weight, ct_int, radius)
    draw_gaussian(this_dense_weight, ct_int, radius)

    ys, xs = np.where(this_dense_weight > 0)
    num_points = xs.size
    for i in range(num_points):
        dists = get_fcos_dists(xs[i], ys[i], bbox)
        dense_dis[:, ys[i], xs[i]] = dists
        # # -----debug------
        # box = [xs[i] - dists[0], ys[i] - dists[1], xs[i] + dists[2], ys[i] + dists[3]]
        # vis = cv2.rectangle(np.zeros((512,512,3),dtype=np.uint8), (int(box[0]*4), int(box[1]*4)), (int(box[2]*4), int(box[3]*4)), (0, 0, 255), 1, lineType=cv2.LINE_AA)
        # cv2.namedWindow('gt_mask', 0)
        # cv2.namedWindow('dis_weight', 0)
        # cv2.imshow('gt_mask', vis)
        # cv2.imshow('dis_weight', dense_weight)
        # cv2.waitKey(1)


def get_pos_feature(pred, weight, target):
    batch, w, h = weight.shape
    device = weight.device
    x_range = torch.arange(
        0, w, 1, dtype=torch.float32, device=device)
    y_range = torch.arange(
        0, h, 1, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(y_range, x_range)
    x = x.unsqueeze(0).repeat(batch, 1, 1)
    y = y.unsqueeze(0).repeat(batch, 1, 1)
    flatten_points = torch.stack(
        (x.reshape(-1), y.reshape(-1)), dim=-1)

    flatten_dis_preds = pred.permute(0, 2, 3, 1).reshape(-1, 4)
    flatten_dis_targets = target.permute(0, 2, 3, 1).reshape(-1, 4)
    flatten_weight = weight.reshape(-1)

    pos_inds = flatten_weight.nonzero()
    num_pos = pos_inds.shape[0]
    pos_dis_preds = flatten_dis_preds[pos_inds].reshape(-1, 4)
    pos_dis_targets = flatten_dis_targets[pos_inds].reshape(-1, 4)
    pos_weights = flatten_weight[pos_inds].reshape(-1)
    pos_points = flatten_points[pos_inds].reshape(-1, 2)
    return num_pos, pos_dis_preds, pos_dis_targets, pos_weights, pos_points



def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    # print("heat pam size:\n", heat.size())
    return heat * keep


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


def ctfcos_decode(heat, dis, K=100):
    batch, cat, height, width = heat.size()
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)

    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)

    dis = _tranpose_and_gather_feat(dis, inds)
    dis = dis.view(batch, K, 4)
    xmin = xs - dis[:, :, 0].unsqueeze(-1)
    ymin = ys - dis[:, :, 1].unsqueeze(-1)
    xmax = xs + dis[:, :, 2].unsqueeze(-1)
    ymax = ys + dis[:, :, 3].unsqueeze(-1)


    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    detections = torch.cat([xmin, ymin, xmax, ymax, scores, clses], dim=2)  # 4 +1 +1

    return detections


