"""
Create by Chengqi.Lv
2020/12/10
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

from ..loss.focal_loss import sigmoid_focal_loss_jit
from ...utils.bbox.iou_calculator import bbox_overlaps
from econn.models.operators.conv import ConvModule
from econn.models.operators.scale import Scale
from .fcos_head import fcos_post_process
from econn.utils.visualize import add_coco_bbox
from .anchor.anchor_target import multi_apply
from econn.models.operators.init_weights import normal_init
from econn.utils.bbox_nms import multiclass_nms
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


class One2OneFCOSHead(nn.Module):
    def __init__(self,
                 input_channel,
                 stacked_convs,
                 num_classes,
                 feat_channels,
                 strides,
                 loss,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 activation='ReLU',
                 use_nms=False
                 ):
        super(One2OneFCOSHead, self).__init__()
        self.in_channels = input_channel
        self.stacked_convs = stacked_convs
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.feat_channels = feat_channels
        self.strides = strides

        self.loss_cfg = loss
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.use_nms = use_nms

        self.class_weight = self.loss_cfg.loss_cls.loss_weight
        self.focal_loss_alpha = self.loss_cfg.loss_cls.alpha
        self.focal_loss_gamma = self.loss_cfg.loss_cls.gamma

        valid_iou = {'IoULoss': 'iou', 'GIoULoss': 'giou', 'EIoULoss': 'eiou'}
        assert self.loss_cfg.loss_iou.name in valid_iou
        self.iou_type = valid_iou[self.loss_cfg.loss_iou.name]
        self.iou_weight = self.loss_cfg.loss_iou.loss_weight

        self.l1_weight = self.loss_cfg.loss_bbox.loss_weight

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                    bias=self.norm_cfg is None))
        self.cls_conv = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.reg_conv = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = -4.595  # 用0.01的置信度初始化
        normal_init(self.cls_conv, std=0.01, bias=bias_cls)
        normal_init(self.reg_conv, std=0.01)
        print('Finish initialize One2One FCOS Head.')

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

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales, self.strides)

    def forward_single(self, x, scale, stride):
        cls_feature = x
        box_feature = x
        for cl in self.cls_convs:
            cls_feature = cl(cls_feature)
        for rg in self.reg_convs:
            box_feature = rg(box_feature)

        pred_hm = self.cls_conv(cls_feature)
        pred_dis = scale(self.reg_conv(box_feature)).float()
        pred_dis = F.relu(pred_dis)

        locations = self.locations(x, stride=stride)[None]
        pred_box = self.apply_ltrb(locations, pred_dis * stride)
        return pred_hm, pred_box

    def loss(self,
             preds,
             gt_meta,
             ):
        pred_hms, pred_boxs = preds
        device = pred_hms[0].device

        input_height, input_width = gt_meta['img'].shape[2:]

        all_level_hm = []
        all_level_box = []
        for pred_hm, pred_box in zip(pred_hms, pred_boxs):
            bs, k, h, w = pred_hm.shape
            device = pred_hm.device
            batch_out_hm = pred_hm.permute(0, 2, 3, 1).reshape(bs, h * w, k)  # [batch_size, num_queries, num_classes]
            batch_out_bbox = pred_box.permute(0, 2, 3, 1).reshape(bs, h * w, 4)  # [batch_size, num_queries, 4]
            all_level_hm.append(batch_out_hm)
            all_level_box.append(batch_out_bbox)

        all_level_hm = torch.cat(all_level_hm, dim=1)
        all_level_box = torch.cat(all_level_box, dim=1)

        mini_cost_idx = self.get_mini_cost_target_idx(all_level_hm.clone().detach(), all_level_box.clone().detach(), gt_meta)

        # 计算所有显卡上的平均目标数，用于loss归一化
        num_targets = sum(len(t) for t in gt_meta['gt_labels'])
        num_targets = torch.as_tensor([num_targets], dtype=torch.float, device=device)
        num_targets = reduce_mean(num_targets)
        num_targets = torch.clamp(num_targets, min=1).item()

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(mini_cost_idx)])
        src_idx = torch.cat([src for (src, _) in mini_cost_idx])
        idx = (batch_idx, src_idx)

        target_classes_o = torch.cat(
            [torch.from_numpy(t)[J] for t, (_, J) in zip(gt_meta['gt_labels'], mini_cost_idx)]).to(device)
        target_classes = torch.full(all_level_hm.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=device)
        target_classes[idx] = target_classes_o

        src_logits = all_level_hm.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        loss_cls = self.class_weight * sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_targets

        src_boxes = all_level_box[idx]
        target_boxes = torch.cat([torch.from_numpy(t)[i] for t, (_, i) in zip(gt_meta['gt_bboxes'], mini_cost_idx)],
                                 dim=0).to(device)

        loss_iou = 1 - bbox_overlaps(src_boxes, target_boxes, mode=self.iou_type, is_aligned=True)
        loss_iou = self.iou_weight * loss_iou.sum() / num_targets

        whwh = [input_width, input_height, input_width, input_height]
        image_size_out = torch.tensor(whwh, dtype=torch.float32, device=device).repeat(src_boxes.shape[0], 1)
        src_boxes_ = src_boxes / image_size_out
        image_size_tgt = torch.tensor(whwh, dtype=torch.float32, device=device).repeat(target_boxes.shape[0], 1)
        target_boxes_ = target_boxes / image_size_tgt

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        loss_bbox = self.l1_weight * loss_bbox.sum() / num_targets

        loss = loss_cls + loss_bbox + loss_iou
        loss_states = dict(loss=loss,
                           loss_cls=loss_cls,
                           loss_iou=loss_iou,
                           bbox_loss=loss_bbox
                           )
        return loss, loss_states

    @torch.no_grad()
    def get_mini_cost_target_idx(self, batch_out_hm, batch_out_bbox, gt_meta):
        batch_out_prob = batch_out_hm.sigmoid()
        bs = batch_out_prob.shape[0]
        device = batch_out_prob.device
        input_height, input_width = gt_meta['img'].shape[2:]
        gt_bboxes = gt_meta['gt_bboxes']
        gt_labels = gt_meta['gt_labels']

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
            whwh = [input_width, input_height, input_width, input_height]
            image_size_out = torch.tensor(whwh, dtype=torch.float32, device=device).repeat(out_bbox.shape[0], 1)
            out_bbox_ = out_bbox / image_size_out
            image_size_tgt = torch.tensor(whwh, dtype=torch.float32, device=device).repeat(tgt_bbox.shape[0], 1)
            tgt_bbox_ = tgt_bbox / image_size_tgt
            cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)

            # Compute the iou cost betwen boxes
            cost_iou = -bbox_overlaps(out_bbox, tgt_bbox, mode=self.iou_type)

            # Final cost matrix
            C = self.l1_weight * cost_bbox + self.class_weight * cost_class + self.iou_weight * cost_iou

            _, src_ind = torch.min(C, dim=0)
            tgt_ind = torch.arange(len(tgt_ids)).to(src_ind)
            indices.append((src_ind, tgt_ind))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def decode(self, preds, img_metas, resize_keep_ratio):
        pred_hms, pred_boxs = preds
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
        # input_height, input_width = img_metas['img'].shape[2:]
        # plt.cla()
        # num_level = len(self.strides)
        # for idx, (stride, hm) in enumerate(zip(self.strides, pred_hms)):
        #     img = cv2.resize(img, (int(input_height/stride), int(input_width/stride)))
        #     scores = hm.sigmoid()
        #
        #     plt.subplot(1, num_level, idx+1, title="stride={}".format(stride))
        #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
        #     plt.imshow(torch.max(scores[0].detach(), dim=0)[0].cpu().numpy(), cmap='jet', vmin=0, vmax=1, alpha=0.7)
        # plt.pause(0.01)

        img_height = img_metas['img_info']['height'].cpu().numpy() \
            if isinstance(img_metas['img_info']['height'], torch.Tensor) else img_metas['img_info']['height']
        img_width = img_metas['img_info']['width'].cpu().numpy() \
            if isinstance(img_metas['img_info']['width'], torch.Tensor) else img_metas['img_info']['width']
        img_shapes = np.array([img_width, img_height], dtype=np.float32).reshape(-1, 2)
        input_height, input_width = img_metas['img'].shape[2:]
        center = np.array([img_width / 2., img_height / 2.], dtype=np.float32).reshape(-1, 2)

        all_level_hm = []
        all_level_box = []
        for pred_hm, pred_box in zip(pred_hms, pred_boxs):
            batch_out_hm = pred_hm.flatten(2)
            batch_out_bbox = pred_box.flatten(2)
            all_level_hm.append(batch_out_hm)
            all_level_box.append(batch_out_bbox)

        scores = torch.cat(all_level_hm, dim=2).sigmoid()
        box_pred = torch.cat(all_level_box, dim=2)

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
        if not self.use_nms:
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

        else:
            cls_scores = cls_scores.transpose(0, 1)
            bbox_preds = bbox_preds.transpose(0, 1)

            padding = cls_scores.new_zeros(cls_scores.shape[0], 1)
            cls_scores = torch.cat([padding, cls_scores], dim=1)

            det_bboxes, det_labels = multiclass_nms(
                bbox_preds,
                cls_scores,
                score_thr=0.05,
                nms_cfg=dict(type='nms', iou_thr=0.6),
                max_num=100)

            return det_bboxes, det_labels
