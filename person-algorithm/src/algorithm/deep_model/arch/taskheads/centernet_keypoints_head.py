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

from econn.models.operators.conv import act_dict
from econn.utils.visualize import add_coco_bbox, add_coco_hp
from econn.utils.debugger import Debugger
from econn.utils.transforms import get_affine_transform, affine_transform
from econn.models.loss.centernet_loss import CenterNetFocalLoss as FocalLoss
from econn.models.loss.centernet_loss import RegL1Loss
from econn.models.loss.centernet_loss import RegWeightedL1Loss
from econn.models.loss.centernet_loss import _tranpose_and_gather_feat, _gather_feat
from econn.models.taskheads.centernet_head import _sigmoid, gaussian_radius, draw_gaussian, gaussian2D


# debugger = Debugger(['person'])
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


class CenterNetKeypointsHead(nn.Module):
    def __init__(self,
                 input_channel,
                 num_classes,
                 head_conv,
                 strides,
                 loss_cfg,
                 norm_wh=False,
                 activation='ReLU'
                 ):
        super(CenterNetKeypointsHead, self).__init__()
        self.max_objs = 128
        self.num_classes = num_classes
        self.strides = strides
        self.encoder = partial(encoder, num_classes=num_classes, stride=strides[0],
                               norm_wh=norm_wh)  # encoder必须指向一个静态方法

        self.loss_cfg = loss_cfg
        self.norm_wh = norm_wh
        self.loss_class = FocalLoss()
        self.loss_reg = RegL1Loss()
        self.loss_hp = RegWeightedL1Loss()
        self.loss_hp_offset = RegL1Loss()
        self.loss_hm_hp = FocalLoss()

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

        self.hp = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, 34,
                      kernel_size=1, stride=1,
                      padding=0, bias=True)
        )
        fill_fc_weights(self.hp)

        self.hp_offset = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, 2,
                      kernel_size=1, stride=1,
                      padding=0, bias=True)
        )
        fill_fc_weights(self.hp_offset)

        self.hm_hp = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            act_dict[activation],
            nn.Conv2d(head_conv, 17,
                      kernel_size=1, stride=1,
                      padding=0, bias=True)
        )
        fill_fc_weights(self.hm_hp)
        self.hm_hp[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        x = x[0]
        out = (self.heatmap(x), self.wh(x), self.offset(x), self.hp(x), self.hp_offset(x), self.hm_hp(x))
        return out

    def loss(self,
             preds,
             gt_meta,
             ):
        pred_hm, pred_wh, pred_offset, pred_hp, pred_hp_offset, pred_hm_hp = preds
        pred_hm = _sigmoid(pred_hm)
        pred_hm_hp = _sigmoid(pred_hm_hp)
        b, c, h, w = pred_hm.shape
        # gt_hm, gt_wh, gt_reg, gt_reg_mask, gt_ind = self.encode(pred_hm, pred_wh, pred_offset, gt_meta)
        device = pred_hm.device
        gt_hm = gt_meta['encoded_gt']['gt_hm'].to(device)
        gt_wh = gt_meta['encoded_gt']['gt_wh'].to(device)
        gt_reg = gt_meta['encoded_gt']['gt_reg'].to(device)
        gt_hp = gt_meta['encoded_gt']['gt_hp'].to(device)
        gt_hp_offset = gt_meta['encoded_gt']['gt_hp_offset'].to(device)
        gt_hm_hp = gt_meta['encoded_gt']['gt_hm_hp'].to(device)
        gt_reg_mask = gt_meta['encoded_gt']['gt_reg_mask'].to(device)
        gt_ind = gt_meta['encoded_gt']['gt_ind'].to(device)
        gt_hp_mask = gt_meta['encoded_gt']['gt_hp_mask'].to(device)
        gt_hm_hp_ind = gt_meta['encoded_gt']['gt_hm_hp_ind'].to(device)
        gt_hm_hp_mask = gt_meta['encoded_gt']['gt_hm_hp_mask'].to(device)

        loss_cls = self.loss_class(pred_hm, gt_hm) * self.loss_cfg.loss_hm.weight
        loss_wh = self.loss_reg(pred_wh, gt_reg_mask, gt_ind, gt_wh) * self.loss_cfg.loss_wh.weight
        if self.norm_wh:
            loss_wh *= math.sqrt(w * h)
        loss_offset = self.loss_reg(pred_offset, gt_reg_mask, gt_ind, gt_reg) * self.loss_cfg.loss_offset.weight

        loss_hp = self.loss_hp(pred_hp, gt_hp_mask, gt_ind, gt_hp) * self.loss_cfg.loss_hp.weight
        loss_hp_offset = self.loss_hp_offset(pred_hp_offset, gt_hm_hp_mask, gt_hm_hp_ind,
                                             gt_hp_offset) * self.loss_cfg.loss_hp_offset.weight
        loss_hm_hp = self.loss_hm_hp(pred_hm_hp, gt_hm_hp) * self.loss_cfg.loss_hm_hp.weight
        loss = loss_cls + loss_wh + loss_offset + loss_hp + loss_hp_offset + loss_hm_hp
        loss_states = dict(loss=loss,
                           hm_loss=loss_cls,
                           wh_loss=loss_wh,
                           offset_loss=loss_offset,
                           hp_loss=loss_hp,
                           hp_offset_loss=loss_hp_offset,
                           hm_hp_loss=loss_hm_hp
                           )

        return loss, loss_states

    def decode(self, preds, meta, resize_keep_ratio):
        pred_hm, pred_wh, pred_offset, pred_hp, pred_hp_offset, pred_hm_hp = preds
        pred_hm = _sigmoid(pred_hm)
        pred_hm_hp = _sigmoid(pred_hm_hp)
        dets = multi_pose_decode(pred_hm, pred_wh, pred_offset, pred_hp, pred_hp_offset, pred_hm_hp, self.norm_wh)
        dets = dets.detach().cpu().numpy()
        img_height = meta['img_info']['height'].cpu().numpy() \
            if isinstance(meta['img_info']['height'], torch.Tensor) else meta['img_info']['height']
        img_width = meta['img_info']['width'].cpu().numpy() \
            if isinstance(meta['img_info']['width'], torch.Tensor) else meta['img_info']['width']
        c = np.array([img_width / 2., img_height / 2.], dtype=np.float32).reshape(-1, 2)
        s = np.array([img_width, img_height], dtype=np.float32).reshape(-1, 2)
        dets_out = multi_pose_post_process(dets.copy(),
                                           c,
                                           s,
                                           preds[0].shape[2],
                                           preds[0].shape[3],
                                           resize_keep_ratio=resize_keep_ratio)

        # # ------------debug----------------------
        # img = ((meta['img'][0].cpu().numpy().transpose(1, 2, 0) * std + mean)*255).astype(np.uint8)
        # pred = debugger.gen_colormap(pred_hm[0].detach().cpu().numpy())
        # debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(1))
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
                    add_coco_hp(img, bbox[5:39])
        return img


def centernet_single_img(num_classes,
                         num_target,
                         gt_labels,
                         gt_bboxes,
                         gt_keypoints,
                         hm_height,
                         hm_width,
                         norm_wh):
    hm = np.zeros((num_classes, hm_height, hm_width), dtype=np.float32)
    wh = np.zeros((128, 2), dtype=np.float32)
    reg = np.zeros((128, 2), dtype=np.float32)
    reg_mask = np.zeros((128), dtype=np.uint8)
    ind = np.zeros((128), dtype=np.int64)
    hp = np.zeros((128, 34), dtype=np.float32)
    hp_offset = np.zeros((128 * 17, 2), dtype=np.float32)
    hm_hp = np.zeros((17, hm_height, hm_width), dtype=np.float32)
    hp_mask = np.zeros((128, 34), dtype=np.uint8)
    hm_hp_ind = np.zeros((128 * 17), dtype=np.int64)
    hm_hp_mask = np.zeros((128 * 17), dtype=np.int64)
    for k in range(min(num_target, 128)):
        bbox = gt_bboxes[k]
        cls_id = gt_labels[k]
        # kps = np.array(gt_keypoints[k], np.float32).reshape(17,3)
        kps = gt_keypoints[k]

        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, hm_width - 1)  # 裁剪，把值限制在0～output_w - 1之间，防止越界
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, hm_height - 1)
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
            num_kps = kps[:, 2].sum()
            if num_kps == 0:
                hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                reg_mask[k] = 0
            # kps[:, 0] = np.clip(kps[:, 0], 0, hm_width - 1)
            # kps[:, 1] = np.clip(kps[:, 1], 0, hm_height - 1)
            for j in range(17):
                if kps[j, 2] > 0:
                    if kps[j, 0] > 0 and kps[j, 0] < hm_width and kps[j, 1] > 0 and kps[j, 1] < hm_height:
                        hp[k, j * 2: j * 2 + 2] = kps[j, :2] - ct_int
                        hp_mask[k, j * 2: j * 2 + 2] = 1
                        pt_int = kps[j, :2].astype(np.int32)
                        hp_offset[k * 17 + j] = kps[j, :2] - pt_int
                        hm_hp_ind[k * 17 + j] = pt_int[1] * hm_width + pt_int[0]
                        hm_hp_mask[k * 17 + j] = 1
                        draw_gaussian(hm_hp[j], pt_int, radius)
    return hm, wh, reg, reg_mask, ind, hp, hp_offset, hm_hp, hp_mask, hm_hp_ind, hm_hp_mask


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
    gt_keypoints = gt_meta['gt_keypoints']
    single_gt_labels = gt_labels
    single_gt_bboxes = gt_bboxes / stride  # 缩小到feature map大小
    single_gt_keypoints = gt_keypoints.reshape(-1, 17, 3)  # 缩小关键点尺寸至feature map 大小
    single_gt_keypoints[:, :, :2] = single_gt_keypoints[:, :, :2] / stride
    # print('\ngt_keypoints in head',single_gt_keypoints)

    hm, wh, reg, reg_mask, ind, hp, hp_offset, hm_hp, hp_mask, hm_hp_ind, hm_hp_mask = centernet_single_img(num_classes,
                                                                                                            len(
                                                                                                                single_gt_labels),
                                                                                                            single_gt_labels,
                                                                                                            single_gt_bboxes,
                                                                                                            single_gt_keypoints,
                                                                                                            hm_height,
                                                                                                            hm_width,
                                                                                                            norm_wh)
    encoded_gt = dict(gt_hm=torch.from_numpy(hm),
                      gt_wh=torch.from_numpy(wh),
                      gt_reg=torch.from_numpy(reg),
                      gt_reg_mask=torch.from_numpy(reg_mask),
                      gt_ind=torch.from_numpy(ind),
                      gt_hp=torch.from_numpy(hp),
                      gt_hp_offset=torch.from_numpy(hp_offset),
                      gt_hm_hp=torch.from_numpy(hm_hp),
                      gt_hp_mask=torch.from_numpy(hp_mask),
                      gt_hm_hp_ind=torch.from_numpy(hm_hp_ind),
                      gt_hm_hp_mask=torch.from_numpy(hm_hp_mask))
    gt_meta['encoded_gt'] = encoded_gt

    # # -------------debug-----------
    # img = ((gt_meta['img'].cpu().numpy().transpose(1, 2, 0) * std + mean)*255).astype(np.uint8)
    # bboxes = single_gt_bboxes
    # labels = single_gt_labels
    # keypoints = single_gt_keypoints
    # pred = debugger.gen_colormap(hm)
    # debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(1))
    # pred_hp_hm = debugger.gen_colormap(hm_hp)
    # debugger.add_blend_img(img, pred_hp_hm,'pred_hp_hm_{:.1f}'.format(1))
    # debugger.add_img(img)
    # for idx, bbox in enumerate(bboxes):
    #     debugger.add_coco_bbox(bbox*4, labels[idx])
    # for idx, keypoint in enumerate(keypoints):
    #     debugger.add_coco_hp(keypoint[:,:2] * 4)
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


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def multi_pose_decode(heat, wh, reg, kps, hp_offset, hm_hp, norm_wh):
    K = 100
    batch, cat, height, width = heat.size()
    num_joints = 17
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    # kps
    kps = _tranpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    # reg
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

    # hm_hp
    hm_hp = _nms(hm_hp)
    thresh = 0.1
    kps = kps.view(batch, K, num_joints, 2).permute(
        0, 2, 1, 3).contiguous()  # b x J x K x 2
    reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
    hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
    # hp_offset
    hp_offset = _tranpose_and_gather_feat(
        hp_offset, hm_inds.view(batch, -1))
    hp_offset = hp_offset.view(batch, num_joints, K, 2)
    hm_xs = hm_xs + hp_offset[:, :, :, 0]
    hm_ys = hm_ys + hp_offset[:, :, :, 1]

    mask = (hm_score > thresh).float()
    hm_score = (1 - mask) * -1 + mask * hm_score
    hm_ys = (1 - mask) * (-10000) + mask * hm_ys
    hm_xs = (1 - mask) * (-10000) + mask * hm_xs
    hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
        2).expand(batch, num_joints, K, K, 2)
    dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
    min_dist, min_ind = dist.min(dim=3)  # b x J x K
    hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
    min_dist = min_dist.unsqueeze(-1)
    min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
        batch, num_joints, K, 1, 2)
    hm_kps = hm_kps.gather(3, min_ind)
    hm_kps = hm_kps.view(batch, num_joints, K, 2)
    l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
    mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
           (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
           (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
    mask = (mask > 0).float().expand(batch, num_joints, K, 2)
    kps = (1 - mask) * hm_kps + mask * kps
    kps = kps.permute(0, 2, 1, 3).contiguous().view(
        batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)

    return detections


def multi_pose_post_process(dets, c, s, h, w, resize_keep_ratio):
    # dets: batch x max_dets x dim
    # return 1-based class det dict

    # 后处理，主要是把heatmap上预测的点，做仿射变换，变成输出大小下的点坐标
    ret = []
    for i in range(dets.shape[0]):
        bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h), resize_keep_ratio)
        pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h), resize_keep_ratio)
        top_preds = np.concatenate(
            [bbox.reshape(-1, 4), dets[i, :, 4:5],
             pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
        ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
    return ret
