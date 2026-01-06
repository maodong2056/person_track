"""
Create by Chengqi.Lv
2020/12/07
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from torchvision.ops.boxes import box_area

from ..loss.focal_loss import sigmoid_focal_loss_jit
from ...utils.bbox.iou_calculator import bbox_overlaps

from .fcos_head import fcos_post_process
from econn.utils.visualize import add_coco_bbox
from econn.utils.debugger import Debugger


# debugger = Debugger(['PuTongMen','ShaFa',
#                      'ChuangHu','ZhuoZi','ChaJi','DianShiGui',
#                      'DianShi','Chuang','ChuangTouGui',
#                      'MaTong', 'uchair'])

mean = np.array([0.40789654, 0.44719302, 0.47026115],
                dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.28863828, 0.27408164, 0.27809835],
               dtype=np.float32).reshape(1, 1, 3)


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.true_divide(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class One2OneCenterHead(nn.Module):
    def __init__(self,
                 input_channel,
                 num_classes,
                 head_conv,
                 strides,
                 loss,
                 class_weight,
                 giou_weight,
                 l1_weight,
                 ):
        super(One2OneCenterHead, self).__init__()
        self.input_channel = input_channel
        self.num_classes = num_classes
        self.head_conv = head_conv
        self.strides = strides

        self.class_weight = class_weight
        self.giou_weight = giou_weight
        self.l1_weight = l1_weight

        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        # self.loss_cls = FocalLoss(gamma=self.focal_loss_gamma,
        #                           alpha=self.focal_loss_alpha,
        #                           reduction="sum",
        #                           loss_weight=self.class_weight)

        self.loss_cfg = loss

        self._init_layers()

    def _init_layers(self):
        self.heatmap = nn.Sequential(
            nn.Conv2d(self.input_channel, self.head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, self.num_classes,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))

        fill_fc_weights(self.heatmap)
        self.heatmap[-1].bias.data.fill_(-4.595)

        self.dis = nn.Sequential(
            nn.Conv2d(self.input_channel, self.head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, 4,
                      kernel_size=1, stride=1,
                      padding=0, bias=True),
        )
        fill_fc_weights(self.dis)
        self.dis[-1].bias.data.fill_(2)

    @torch.no_grad()
    def locations(self, features, stride=4):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]
        device = features.device

        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        locations = locations.reshape(h, w, 2).permute(2, 0, 1)

        return locations

    def apply_ltrb(self, locations, pred_ltrb):
        """
        :param locations:  (1, 2, H, W)
        :param pred_ltrb:  (N, 4, H, W)
        """

        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[:, 0, :, :] = locations[:, 0, :, :] - pred_ltrb[:, 0, :, :]  # x1
        pred_boxes[:, 1, :, :] = locations[:, 1, :, :] - pred_ltrb[:, 1, :, :]  # y1
        pred_boxes[:, 2, :, :] = locations[:, 0, :, :] + pred_ltrb[:, 2, :, :]  # x2
        pred_boxes[:, 3, :, :] = locations[:, 1, :, :] + pred_ltrb[:, 3, :, :]  # y2

        return pred_boxes

    def forward(self, x):
        x = x[0]
        locations = self.locations(x)[None]
        pred_hm = self.heatmap(x)
        pred_dis = F.relu(self.dis(x))
        pred_box = self.apply_ltrb(locations, pred_dis)
        return pred_hm, pred_box

    def loss(self,
             preds,
             gt_meta,
             ):
        pred_hm, pred_box = preds
        device = pred_hm.device

        input_height, input_width = gt_meta['img'].shape[2:]

        mini_cost_idx = self.get_mini_cost_target_idx(pred_hm.clone().detach(), pred_box.clone().detach(), gt_meta)

        # 计算所有显卡上的平均目标数，用于loss归一化
        num_targets = sum(len(t) for t in gt_meta['gt_labels'])
        num_targets = torch.as_tensor([num_targets], dtype=torch.float, device=device)
        num_targets = reduce_mean(num_targets)
        num_targets = torch.clamp(num_targets, min=1).item()

        bs, k, h, w = pred_hm.shape
        src_logits = pred_hm.permute(0, 2, 3, 1).reshape(bs, h * w, k)
        src_boxes = pred_box.permute(0, 2, 3, 1).reshape(bs, h * w, 4)

        batch_idx = torch.cat([torch.full_like(src, i).to(device) for i, (src, _) in enumerate(mini_cost_idx)])
        src_idx = torch.cat([src.to(device) for (src, _) in mini_cost_idx])

        idx = (batch_idx, src_idx)

        target_classes_o = torch.cat([torch.from_numpy(t)[J] for t, (_, J) in zip(gt_meta['gt_labels'], mini_cost_idx)]).to(device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        # loss_cls = self.loss_cls(src_logits, labels) / num_targets
        loss_cls = self.class_weight * sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_targets

        src_boxes = src_boxes[idx]
        target_boxes = torch.cat([torch.from_numpy(t)[i] for t, (_, i) in zip(gt_meta['gt_bboxes'], mini_cost_idx)], dim=0).to(device)

        loss_giou = 1 - torch.diag(bbox_overlaps(src_boxes, target_boxes, mode="giou"))
        loss_giou = self.giou_weight * loss_giou.sum() / num_targets

        src_boxes_ = src_boxes / input_height
        target_boxes_ = target_boxes / input_height

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        loss_bbox = self.l1_weight * loss_bbox.sum() / num_targets

        loss = loss_cls + loss_bbox + loss_giou
        loss_states = dict(loss=loss,
                           loss_cls=loss_cls,
                           loss_giou=loss_giou,
                           bbox_loss=loss_bbox
                           )
        return loss, loss_states

    @torch.no_grad()
    def get_mini_cost_target_idx(self, pred_hm, pred_box, gt_meta):
        '''
        Mini Cost Target Match
        :param pred_hm:
        :param pred_dis:
        :param gt_meta:
        :return:
        '''
        bs, k, h, w = pred_hm.shape
        device = pred_hm.device

        input_height, input_width = gt_meta['img'].shape[2:]
        gt_bboxes = gt_meta['gt_bboxes']
        gt_labels = gt_meta['gt_labels']

        batch_out_prob = pred_hm.permute(0, 2, 3, 1).reshape(bs, h * w, k).sigmoid()  # [batch_size, num_queries, num_classes]
        batch_out_bbox = pred_box.permute(0, 2, 3, 1).reshape(bs, h * w, 4)  # [batch_size, num_queries, 4]

        indices = []
        for i in range(bs):
            tgt_ids = torch.from_numpy(gt_labels[i]).to(device)

            if tgt_ids.shape[0] == 0:
                indices.append(([], []))
                continue

            tgt_bbox = torch.from_numpy(gt_bboxes[i]).to(device)
            out_prob = batch_out_prob[i]
            out_bbox = batch_out_bbox[i]

            # Compute the classification cost.
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            # 归一化到相对坐标
            out_bbox_ = out_bbox / input_height  # TODO: 改成根据宽高归一化
            tgt_bbox_ = tgt_bbox / input_height
            cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)

            # Compute the giou cost betwen boxes
            # cost_giou = -self.giou(out_bbox, tgt_bbox)
            cost_giou = -bbox_overlaps(out_bbox, tgt_bbox, mode="giou")

            # Final cost matrix
            C = self.l1_weight * cost_bbox + self.class_weight * cost_class + self.giou_weight * cost_giou

            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(len(tgt_ids)).to(src_ind)
            indices.append((src_ind, tgt_ind))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def decode(self, preds, img_metas, resize_keep_ratio):
        pred_hm, pred_box = preds
        scores = torch.sigmoid(pred_hm)
        # # ------------debug----------------------
        # img = ((meta['img'][0].cpu().numpy().transpose(1, 2, 0) * std + mean)*255).astype(np.uint8)
        # # pred = debugger.gen_colormap(pred_hm[0].detach().cpu().numpy())
        # # debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(1))
        # debugger.add_img(cv2.resize(img, (1280,720)))
        # for label, bboxes in dets_out[0].items():
        #     for bbox in bboxes:
        #         if bbox[-1] > 0.3:
        #             debugger.add_coco_bbox(bbox[:-1], label-1, bbox[-1])
        # debugger.show_all_imgs(pause=False, expname='aaa')

        # # ------------Matplotlib debug----------------------
        # img = img_metas['img'][0].cpu().numpy().transpose(1, 2, 0) * std + mean
        # img = cv2.resize(img, (80, 80))
        # # plt.colorbar()
        # plt.cla()
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
        # plt.imshow(torch.max(scores[0].detach(), dim=0)[0].cpu().numpy(), cmap='jet', vmin=0, vmax=1, alpha=0.7)
        # plt.pause(0.01)

        img_height = img_metas['img_info']['height'].cpu().numpy() \
            if isinstance(img_metas['img_info']['height'], torch.Tensor) else img_metas['img_info']['height']
        img_width = img_metas['img_info']['width'].cpu().numpy() \
            if isinstance(img_metas['img_info']['width'], torch.Tensor) else img_metas['img_info']['width']
        img_shapes = np.array([img_width, img_height], dtype=np.float32).reshape(-1, 2)
        input_height, input_width = img_metas['img'].shape[2:]
        center = np.array([img_width / 2., img_height / 2.], dtype=np.float32).reshape(-1, 2)

        scores = scores.flatten(2)
        box_pred = pred_box.flatten(2)
        result_list = []
        for img_id, (scores_per_image, box_pred_per_image) in enumerate(zip(scores, box_pred)):
            img_shape = img_shapes[img_id]
            c = center[img_id]
            det_bboxes, det_labels = self.get_bboxes_single(scores_per_image, box_pred_per_image, K=100)
            dets = fcos_post_process(det_bboxes,
                                     det_labels,
                                     c,
                                     img_shape,
                                     input_height,
                                     input_width,
                                     self.num_classes,
                                     resize_keep_ratio)
            result_list.append(dets)



        # TODO:直接解码出json格式
        return result_list

    def show_result(self, img, dets, class_names, score_thres=0.3, show=True, save_path=None):
        dets = dets[0]
        for label in dets:
            for bbox in dets[label]:
                if bbox[4] > score_thres:
                    add_coco_bbox(img, bbox[:4], label - 1, bbox[4], class_names)
        return img

    def get_bboxes_single(self, cls_scores, bbox_preds, K=100):
        topk_score_cat, topk_inds_cat = torch.topk(cls_scores, k=K)
        topk_score, topk_inds = torch.topk(topk_score_cat.reshape(-1), k=K)
        topk_clses = topk_inds // K
        scores_per_image = topk_score.view(K, 1)
        labels_per_image = topk_clses
        topk_box_cat = bbox_preds[:, topk_inds_cat.reshape(-1)]
        topk_box = topk_box_cat[:, topk_inds]
        box_pred_per_image = topk_box.transpose(0, 1)
        det_bboxes = torch.cat([box_pred_per_image, scores_per_image], dim=1)

        return det_bboxes, labels_per_image

