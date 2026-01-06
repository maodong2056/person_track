"""
Create by Chengqi.Lv
2020/5/19
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import math
import time

from econn.utils.visualize import add_coco_bbox
from econn.utils.visualize import InstanceVisualizer
from econn.utils.debugger import Debugger
from econn.models.loss.centernet_loss import CenterNetFocalLoss as FocalLoss
from econn.models.loss.centernet_loss import _tranpose_and_gather_feat, _gather_feat
from econn.models.loss.dice_loss import dice_loss
from econn.utils.transforms import get_affine_transform
import pycocotools.mask as mask_util

debugger = Debugger(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
              'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
              'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
              'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
              'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'])

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
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    for m in up.modules():
        if isinstance(m, nn.ConvTranspose2d):
            w = m.weight.data
            f = math.ceil(w.size(2) / 2)
            c = (2 * f - 1 - f % 2) / (2. * f)
            for i in range(w.size(2)):
                for j in range(w.size(3)):
                    w[0, 0, i, j] = \
                        (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
            for c in range(1, w.size(0)):
                w[c, 0, :, :] = w[0, 0, :, :]
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


class CtCondInstHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 head_conv,
                 strides,
                 loss_cfg
                 ):
        super(CtCondInstHead, self).__init__()
        self.max_objs = 128
        self.num_classes = num_classes
        self.strides = strides
        self.encoder = partial(encoder, num_classes=num_classes, stride=strides[0])  # encoder必须指向一个静态方法

        # self.x_map = None
        # self.y_map = None

        self.loss_cfg = loss_cfg
        self.loss_class = FocalLoss()

        self.inplanes = in_channels[-1]
        self.deconv_channels = [256, 128, 128]
        self.deconv1 = self._make_deconv_layer(self.deconv_channels[0], 4)
        self.connect1 = nn.Conv2d(in_channels[2], self.deconv_channels[0], 1, 1, 0, bias=True)
        self.deconv2 = self._make_deconv_layer(self.deconv_channels[1], 4)
        self.connect2 = nn.Conv2d(in_channels[1], self.deconv_channels[1], 1, 1, 0, bias=True)
        self.deconv3 = self._make_deconv_layer(self.deconv_channels[2], 4)
        self.connect3 = nn.Conv2d(in_channels[0], self.deconv_channels[2], 1, 1, 0, bias=True)

        self.heatmap = nn.Sequential(
            nn.Conv2d(self.deconv_channels[-1], head_conv,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),

            nn.Conv2d(head_conv, head_conv,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),

            nn.Conv2d(head_conv, num_classes,
                      kernel_size=3, stride=1,
                      padding=1, bias=True))

        fill_fc_weights(self.heatmap)
        self.heatmap[-1].bias.data.fill_(-4.595)

        self.ctl = nn.Sequential(
            nn.Conv2d(self.deconv_channels[-1], 128,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 169,
                      kernel_size=3, stride=1,
                      padding=1, bias=True),
        )
        fill_fc_weights(self.ctl)

        self.mask = nn.Sequential(
            nn.Conv2d(self.deconv_channels[-1], 128,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 8,
                      kernel_size=3, stride=1,
                      padding=1, bias=True),
        )
        fill_fc_weights(self.mask)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters, num_kernels):

        layers = []
        kernel, padding, output_padding = \
            self._get_deconv_cfg(num_kernels)

        planes = num_filters

        fc = nn.Conv2d(self.inplanes, planes,
                       kernel_size=3, stride=1,
                       padding=1, dilation=1, bias=False)
        fill_fc_weights(fc)
        up = nn.ConvTranspose2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=kernel,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=False)
        fill_up_weights(up)

        layers.append(fc)
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        f1, f2, f3, f4 = x
        d1out = self.deconv1(f4) + self.connect1(f3)
        d2out = self.deconv2(d1out) + self.connect2(f2)
        d3out = self.deconv3(d2out) + self.connect3(f1)
        out = (self.heatmap(d3out), self.ctl(d3out), self.mask(d3out))
        return out

    def loss(self,
             preds,
             gt_meta,
             ):
        pred_hm, pred_ctl, mask_features = preds
        pred_hm = _sigmoid(pred_hm)
        N, C, H, W = pred_hm.shape
        _, __, mask_h, mask_w = mask_features.shape
        grid_x = torch.arange(mask_w).view(1, -1).float().repeat(mask_h, 1).cuda() / (mask_w - 1) * 2 - 1
        grid_y = torch.arange(mask_h).view(-1, 1).float().repeat(1, mask_w).cuda() / (mask_h - 1) * 2 - 1
        x_map = grid_x.view(1, 1, mask_h, mask_w).repeat(N, 1, 1, 1)
        y_map = grid_y.view(1, 1, mask_h, mask_w).repeat(N, 1, 1, 1)
        mask_features = torch.cat((mask_features, x_map, y_map), dim=1)

        device = pred_hm.device
        gt_masks = gt_meta['gt_masks']
        gt_hm = gt_meta['encoded_gt']['gt_hm'].to(device)
        gt_ind = gt_meta['encoded_gt']['gt_ind'].to(device)
        gt_valid_mask = gt_meta['encoded_gt']['gt_valid_mask'].to(device)

        loss_cls = self.loss_class(pred_hm, gt_hm) * self.loss_cfg.loss_hm.weight

        pred_ctl = pred_ctl.permute(0, 2, 3, 1)
        pred_ctl = pred_ctl.reshape(N, -1, 169)
        loss_mask = mask_features[0].new_tensor(0.0)
        batch_ins = 0
        for i in range(N):
            valid_mask_id = gt_valid_mask[i].nonzero().squeeze(dim=1)
            ind = gt_ind[i]
            ctl_ind = ind[valid_mask_id]
            masks = torch.from_numpy(gt_masks[i]).to(device).float()
            mask_feat = mask_features[i].unsqueeze(dim=0)
            ins_num = ctl_ind.shape[0]
            batch_ins += ins_num
            if ins_num>0:
                controllers = pred_ctl[i][ctl_ind]
                weights1 = controllers[:, :80].reshape(-1, 8, 10).reshape(-1, 10).unsqueeze(-1).unsqueeze(-1)
                bias1 = controllers[:, 80:88].flatten()
                weights2 = controllers[:, 88:152].reshape(-1, 8, 8).reshape(-1, 8).unsqueeze(-1).unsqueeze(-1)
                bias2 = controllers[:, 152:160].flatten()
                weights3 = controllers[:, 160:168].unsqueeze(-1).unsqueeze(-1)
                bias3 = controllers[:, 168:169].flatten()
                conv1 = F.conv2d(mask_feat, weights1, bias1).relu()
                conv2 = F.conv2d(conv1, weights2, bias2, groups=ins_num).relu()
                masks_per_image = F.conv2d(conv2, weights3, bias3, groups=ins_num)
                masks_per_image = aligned_bilinear(masks_per_image, factor=4)[0].sigmoid()
                for j in range(ins_num):
                    mask_target = masks[j]
                    mask_pred = masks_per_image[j]
                    loss_mask += dice_loss(mask_pred, mask_target)

        if batch_ins > 0:
            loss_mask = loss_mask / batch_ins * self.loss_cfg.loss_mask.weight

        loss = loss_cls + loss_mask
        loss_states = dict(loss=loss,
                           hm_loss=loss_cls,
                           mask_loss=loss_mask,
                           )
        return loss, loss_states

    def decode(self, preds, meta, resize_keep_ratio):
        # time1 = time.time()
        pred_hm, pred_ctl, mask_features = preds
        pred_hm = _sigmoid(pred_hm)
        # time2 = time.time()
        # if self.x_map is None:
        #     N, _, mask_h, mask_w = mask_features.shape
        #     grid_x = torch.arange(mask_w).view(1, -1).float().repeat(mask_h, 1).cuda() / (mask_w - 1) * 2 - 1
        #     grid_y = torch.arange(mask_h).view(-1, 1).float().repeat(1, mask_w).cuda() / (mask_h - 1) * 2 - 1
        #     self.x_map = grid_x.view(1, 1, mask_h, mask_w).repeat(N, 1, 1, 1)
        #     self.y_map = grid_y.view(1, 1, mask_h, mask_w).repeat(N, 1, 1, 1)
        # mask_features = torch.cat((mask_features, self.x_map, self.y_map), dim=1)

        N, _, mask_h, mask_w = mask_features.shape
        grid_x = torch.arange(mask_w).view(1, -1).float().repeat(mask_h, 1).cuda() / (mask_w - 1) * 2 - 1
        grid_y = torch.arange(mask_h).view(-1, 1).float().repeat(1, mask_w).cuda() / (mask_h - 1) * 2 - 1
        x_map = grid_x.view(1, 1, mask_h, mask_w).repeat(N, 1, 1, 1)
        y_map = grid_y.view(1, 1, mask_h, mask_w).repeat(N, 1, 1, 1)
        mask_features = torch.cat((mask_features, x_map, y_map), dim=1)

        # time3 = time.time()
        clses, scores, masks = ctcondinst_decode(pred_hm, pred_ctl, mask_features)
        # time4 = time.time()
        img_height = meta['img_info']['height'].cpu().numpy() \
            if isinstance(meta['img_info']['height'], torch.Tensor) else meta['img_info']['height']
        img_width = meta['img_info']['width'].cpu().numpy() \
            if isinstance(meta['img_info']['width'], torch.Tensor) else meta['img_info']['width']
        c = np.array([img_width / 2., img_height / 2.], dtype=np.float32).reshape(-1, 2)
        s = np.array([img_width, img_height], dtype=np.float32).reshape(-1, 2)

        # # ------------debug----------------------
        # img = ((meta['img'][0].cpu().numpy().transpose(1, 2, 0) * std + mean)*255).astype(np.uint8)
        # pred = debugger.gen_colormap(pred_hm[0].detach().cpu().numpy())
        # debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(1))
        # # debugger.add_img(img)
        #
        # debugger.show_all_imgs(pause=True, expname='aaa')
        # time5 = time.time()
        dets_out = ctcondinst_post_process(clses,
                                           scores,
                                           masks,
                                           c,
                                           s,
                                           mask_features.shape[2]*4,
                                           mask_features.shape[3]*4,
                                           pred_hm.shape[1],
                                           resize_keep_ratio=resize_keep_ratio)
        # time6 = time.time()
        # print('sigmoid:',time2-time1, ' coord:', time3-time2, ' decode:',time4-time3, ' post:', time6-time4)

        # TODO:直接解码出json格式
        return dets_out

    def show_result(self, img, dets, class_names, score_thres=0.3, show=True, save_path=None):
        dets = dets[0]
        visualizer = InstanceVisualizer(img, dets, class_names, score_thres)
        result = visualizer.overlay_instance(alpha=0.3)
        return result


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
    hm, valid_mask, ind = ctcondinst_single_img(num_classes,
                                                len(single_gt_labels),
                                                single_gt_labels,
                                                single_gt_bboxes,
                                                hm_height,
                                                hm_width)
    encoded_gt = dict(gt_hm=torch.from_numpy(hm),
                      gt_valid_mask=torch.from_numpy(valid_mask),
                      gt_ind=torch.from_numpy(ind))
    gt_meta['encoded_gt'] = encoded_gt
    return gt_meta


def ctcondinst_single_img(num_classes,
                          num_target,
                          gt_labels,
                          gt_bboxes,
                          hm_height,
                          hm_width):
    hm = np.zeros((num_classes, hm_height, hm_width), dtype=np.float32)
    ind = np.zeros((128), dtype=np.int64)
    valid_mask = np.zeros((128), dtype=np.uint8)

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
            draw_gaussian(hm[cls_id], ct_int, radius)  # 画heatmap的高斯点
            ind[k] = ct_int[1] * hm_width + ct_int[0]  # 索引
            valid_mask[k] = 1
    return hm, valid_mask, ind


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


def ctcondinst_decode(heat, ctl, mask_features, K=20):
    batch, cat, height, width = heat.size()
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)

    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)

    ctls = _tranpose_and_gather_feat(ctl, inds)
    ctls = ctls.view(batch, K, 169)
    masks = []
    for i in range(batch):
        controllers = ctls[i]
        weights1 = controllers[:, :80].reshape(-1, 8, 10).reshape(-1, 10).unsqueeze(-1).unsqueeze(-1)
        bias1 = controllers[:, 80:88].flatten()
        weights2 = controllers[:, 88:152].reshape(-1, 8, 8).reshape(-1, 8).unsqueeze(-1).unsqueeze(-1)
        bias2 = controllers[:, 152:160].flatten()
        weights3 = controllers[:, 160:168].unsqueeze(-1).unsqueeze(-1)
        bias3 = controllers[:, 168:169].flatten()
        conv1 = F.conv2d(mask_features, weights1, bias1).relu()
        conv2 = F.conv2d(conv1, weights2, bias2, groups=K).relu()
        masks_per_image = F.conv2d(conv2, weights3, bias3, groups=K)
        masks_per_image = aligned_bilinear(masks_per_image, factor=4)[0].sigmoid()
        masks.append(masks_per_image.detach().cpu().numpy())

    clses = clses.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()

    return clses, scores, masks


def ctcondinst_post_process(clses, scores, masks, c, s, h, w, num_classes, resize_keep_ratio):

    ret = []
    for i in range(clses.shape[0]):
        top_preds = []
        trans = get_affine_transform(c[i], s[i], 0, (w, h), inv=1, resize_keep_ratio=resize_keep_ratio)
        for idx, cls in enumerate(clses[i].tolist()):
            score = scores[i][idx]
            inpsize_mask = (masks[i][idx] * 255).astype("uint8")
            rawsize_mask = cv2.warpAffine(inpsize_mask, trans, (int(s[i][0]), int(s[i][1])), flags=cv2.INTER_LINEAR)
            _, mask = cv2.threshold(rawsize_mask, thresh=127, maxval=1, type=cv2.THRESH_BINARY)
            rle = mask_util.encode(np.asfortranarray(mask))
            det = dict(label=cls + 1,
                       score=score,
                       mask=mask,
                       rle=rle
                       )
            top_preds.append(det)
        ret.append(top_preds)
    return ret


