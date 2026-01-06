"""
Create by Chengqi.Lv
2020/12/3
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import cv2

from econn.utils.visualize import add_coco_bbox
from econn.models.operators.scale import Scale
from econn.models.operators.conv import ConvModule
from econn.models.operators.init_weights import normal_init
from econn.utils.transforms import distance2bbox, bbox2distance
from econn.utils.bbox_nms import multiclass_nms
from econn.models.loss.varifocal_loss import VarifocalLoss
import econn.models.loss.iou_loss as iou_losses
from .anchor.anchor_target import multi_apply, images_to_levels, anchor_inside_flags, unmap
from .anchor.base_anchor_head import AnchorHead
from .assigner.atss_assigner import ATSSAssigner
from .sampler.pseudo_sampler import PseudoSampler
from .fcos_head import fcos_post_process


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.true_divide(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class VFLHead(AnchorHead):
    """
    Head of `VarifocalNet (VFNet): An IoU-aware Dense Object
    Detector.<https://arxiv.org/abs/2008.13367>
    """

    def __init__(self,
                 num_classes,
                 loss,
                 input_channel,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.loss_cfg = loss
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        use_sigmoid = True
        octave_scales = np.array(
            [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(VFLHead, self).__init__(
            num_classes, loss, use_sigmoid, input_channel, anchor_scales=anchor_scales, **kwargs)

        self.loss_vfl = VarifocalLoss(True, alpha=0.75, gamma=2.0, iou_weighted=True, loss_weight=1.0)
        box_loss_type = self.loss_cfg.loss_bbox.name
        assert box_loss_type in ['IoULoss', 'GIoULoss', 'CIoULoss', 'EIoULoss']
        self.loss_bbox = getattr(iou_losses, box_loss_type)(loss_weight=self.loss_cfg.loss_bbox.loss_weight)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
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
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.cls_conv = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.reg_conv = nn.Conv2d(
            self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.anchor_strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = -4.595  # 用0.01的置信度初始化
        normal_init(self.cls_conv, std=0.01, bias=bias_cls)
        normal_init(self.reg_conv, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.cls_conv(cls_feat)
        bbox_pred = scale(self.reg_conv(reg_feat)).float()
        bbox_pred = F.relu(bbox_pred)
        return cls_score, bbox_pred

    def anchor_center(self, anchors):
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=1)

    def iou_target(self, pred, target, eps=1e-7):
        # overlap
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        # union
        ap = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
        ag = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
        union = ap + ag - overlap + eps
        # IoU
        ious = overlap / union
        return ious

    def loss_single(self, anchors, cls_score, bbox_pred, labels,
                    label_weights, bbox_targets, stride, num_total_samples):

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        pos_inds = torch.nonzero(labels).squeeze(1)
        # # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        # bg_class_ind = self.num_classes
        # pos_inds = ((labels >= 0)
        #             & (labels < bg_class_ind)).nonzero().squeeze(1)
        iou_score_targets = torch.zeros_like(cls_score)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]

            norm_anchor_center = self.anchor_center(pos_anchors) / stride

            pos_bbox_pred_distance = pos_bbox_pred

            pos_decode_bbox_pred = distance2bbox(norm_anchor_center,
                                                 pos_bbox_pred_distance)
            pos_decode_bbox_targets = pos_bbox_targets / stride

            pos_label = (labels -1)[pos_inds].long()

            pos_ious = self.iou_target(pos_decode_bbox_pred.detach(), pos_decode_bbox_targets).clamp(min=1e-6)
            iou_score_targets[pos_inds, pos_label] = pos_ious.clone().detach()
            weight_targets = \
                cls_score.detach().sigmoid().max(dim=1)[0][pos_inds]

            # regression loss
            # print('pos_decode_bbox_pred:',pos_decode_bbox_pred.shape)
            # print('pos_decode_bbox_targets:',pos_decode_bbox_targets.shape)
            # print('weight_targets',weight_targets.shape)
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

        else:
            loss_bbox = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).cuda()

        loss_vfl = self.loss_vfl(cls_score, iou_score_targets.detach(), avg_factor=num_total_samples)

        return loss_vfl, loss_bbox, weight_targets.sum()

    def loss(self,
             preds,
             gt_meta
             ):
        cls_scores, bbox_preds = preds

        gt_bboxes = gt_meta['gt_bboxes']
        gt_labels = gt_meta['gt_labels']

        input_height, input_width = gt_meta['img'].shape[2:]
        img_shapes = [[input_height, input_width] for i in range(cls_scores[0].shape[0])]
        gt_bboxes_ignore = None

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_shapes, device=device)  #"img_shape":shape of the image input to the network as a tuple(h, w, c)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.gfl_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_shapes,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos).cuda()).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_vfl, losses_bbox, \
        avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.anchor_strides,
            num_total_samples=num_total_samples,
            )

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        if avg_factor <= 0:
            loss_vfl = torch.tensor(0, dtype=torch.float32).cuda()
            loss_bbox = torch.tensor(0, dtype=torch.float32).cuda()
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
            loss_vfl = sum(losses_vfl)
            loss_bbox = sum(losses_bbox)

        loss = loss_vfl + loss_bbox
        loss_states = dict(
            loss_vfl=loss_vfl,
            loss_bbox=loss_bbox)

        return loss, loss_states

    def decode(self, preds, meta, resize_keep_ratio):
        cls_scores, bbox_preds = preds
        # #------------debug----------------------
        # from econn.utils.debugger import Debugger
        # debugger = Debugger(['PuTongMen','ShaFa',
        #       'ChuangHu','ZhuoZi','ChaJi','DianShiGui',
        #       'DianShi','Chuang','ChuangTouGui',
        #       'MaTong', 'uchair'])
        #
        # mean = np.array([0.40789654, 0.44719302, 0.47026115],
        #                 dtype=np.float32).reshape(1, 1, 3)
        # std = np.array([0.28863828, 0.27408164, 0.27809835],
        #                dtype=np.float32).reshape(1, 1, 3)
        # img = ((meta['img'][0].cpu().numpy().transpose(1, 2, 0) * std + mean)*255).astype(np.uint8)
        # for idx, hm in enumerate(cls_scores):
        #     pred = debugger.gen_colormap(hm[0].sigmoid().detach().cpu().numpy())
        #     debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(idx))
        # debugger.show_all_imgs(pause=True, expname='aaa')
        # # ------------debug----------------------
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta, resize_keep_ratio)

        return result_list

    def show_result(self, img, dets, class_names, score_thres=0.3, show=True, save_path=None):
        dets = dets[0]
        for label in dets:
            for bbox in dets[label]:
                if bbox[4] > score_thres:
                    add_coco_bbox(img, bbox[:4], label - 1, bbox[4], class_names)
        return img

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   resize_keep_ratio,
                   rescale=False):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        num_classes = cls_scores[0].shape[1]
        device = cls_scores[0].device
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(
                cls_scores[i].size()[-2:],
                self.anchor_strides[i],
                device=device) for i in range(num_levels)
        ]

        img_height = img_metas['img_info']['height'].cpu().numpy() \
            if isinstance(img_metas['img_info']['height'], torch.Tensor) else img_metas['img_info']['height']
        img_width = img_metas['img_info']['width'].cpu().numpy() \
            if isinstance(img_metas['img_info']['width'], torch.Tensor) else img_metas['img_info']['width']
        img_shapes = np.array([img_width, img_height], dtype=np.float32).reshape(-1, 2)
        input_height, input_width = img_metas['img'].shape[2:]
        input_shape = [input_height, input_width]
        center = np.array([img_width / 2., img_height / 2.], dtype=np.float32).reshape(-1, 2)

        result_list = []
        for img_id in range(cls_scores[0].shape[0]):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_shapes[img_id]
            scale_factor = 1
            det_bboxes, det_labels = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, input_shape,
                                               scale_factor, rescale)
            c = center[img_id]
            dets = fcos_post_process(det_bboxes,
                                     det_labels,
                                     c,
                                     img_shape,
                                     input_height,
                                     input_width,
                                     num_classes,
                                     resize_keep_ratio)

            result_list.append(dets)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,  # input shape!!!!
                          scale_factor,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred, anchors in zip(
                self.anchor_strides, cls_scores, bbox_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).contiguous() * stride

            # nms_pre = cfg.get('nms_pre', -1)
            nms_pre = 1000
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(self.anchor_center(anchors), bbox_pred,
                                   max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)

        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr=0.05,
            nms_cfg=dict(type='nms', iou_thr=0.6),
            max_num=100)
        return det_bboxes, det_labels

    def gfl_target(self,
                   anchor_list,
                   valid_flag_list,
                   gt_bboxes_list,
                   img_shape_list,
                   gt_bboxes_ignore_list=None,
                   gt_labels_list=None,
                   label_channels=1,
                   unmap_outputs=True):
        """
        almost the same with anchor_target, with a little modification,
        here we need return the anchor
        """
        num_imgs = len(img_shape_list)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
            self.gfl_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_shape_list,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def gfl_target_single(self,
                          flat_anchors,
                          valid_flags,
                          num_level_anchors,
                          gt_bboxes,
                          gt_bboxes_ignore,
                          gt_labels,
                          img_shape,
                          label_channels=1,
                          unmap_outputs=True):
        device = flat_anchors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        num_gts = gt_labels.size(0)
        if num_gts > 0:
            gt_labels += 1

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_shape,
                                           allowed_border=-1)
        if not inside_flags.any():
            return (None,) * 6
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags.type(torch.bool), :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        bbox_assigner = ATSSAssigner(topk=9)
        assign_result = bbox_assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            # if cfg.pos_weight <= 0:
            #     label_weights[pos_inds] = 1.0
            # else:
            #     label_weights[pos_inds] = cfg.pos_weight
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            inside_flags = inside_flags.type(torch.bool)
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(labels, num_total_anchors, inside_flags)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside