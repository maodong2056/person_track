# @Time : 2021/2/24 10:19 
# @Author : Altair.Huazj
# @File : depthwise_embedding_keypoints_head.py 
# @Software: PyCharm
import torch
import torch.nn as nn
from econn.models.operators.conv import act_dict
from econn.models.operators.keypoint_generator import HeatmapGenerator, \
    ScaleAwareHeatmapGenerator, JointsGenerator
from econn.models.loss.embedding_loss import MultiLossFactory
from functools import partial
from econn.utils.transforms import get_affine_transform, affine_transform
import numpy as np
import cv2
from munkres import Munkres
from econn.utils.visualize import add_coco_hp_emb
import torch.nn.functional as F
BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation='ReLU'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = act_dict[activation]
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class EmbeddingKeypointHead(nn.Module):
    def __init__(self,
                 input_channel,
                 input_size,
                 num_classes,
                 res_blocks,
                 # head_conv,
                 strides,
                 # loss_cfg,
                 # norm_wh=False,
                 activation='ReLU'
                 ):
        super(EmbeddingKeypointHead, self).__init__()
        self.max_objs = 128
        self.input_size = input_size
        self.heatmap_generator = [HeatmapGenerator(int(input_size/strides[0]), num_classes),
                                  HeatmapGenerator(int(input_size/strides[0]*2), num_classes)]
        self.joints_generator = [JointsGenerator(self.max_objs, num_classes, int(input_size / strides[0]), True),
                                 JointsGenerator(self.max_objs, num_classes, int(input_size / strides[0] * 2), True)]
        self.encoder = partial(encoder, input_size = input_size,
                                        num_keypoints = num_classes,
                                        heatmap_generator=self.heatmap_generator,
                                        joints_generator=self.joints_generator)  # encoder必须指向一个静态方法
        self.num_classes = num_classes
        self.activation = activation
        self.output_0 = nn.Sequential(*[nn.Conv2d(input_channel, num_classes*2, 1, 1)])
        deconv_layer = self._make_deconv_layers(num_classes*2+input_channel, input_channel, res_blocks)
        self.output_1 = nn.Sequential(*[deconv_layer, nn.Conv2d(input_channel, num_classes, 1, 1)])
        self.init_weights()
        self.strides = strides
        self.muti_loss = MultiLossFactory(num_classes, 2, (True, True), (1., 1.),
                                     (True, False), (0.01, 0.01), (0.01, 0.01), ae_loss_type='exp')
        self.joint_order = [
            i - 1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
        ]
        self.ADJUST = True
        self.tag_threshold = 1
        self.detection_threshold = 0.1

    def _make_deconv_layers(self, input_channels, output_channels, res_block=4):
        deconv_layers = []
        deconv_layers.append(nn.Sequential(*[nn.ConvTranspose2d(input_channels, output_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                                            nn.BatchNorm2d(output_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                            act_dict[self.activation]]))
        for i in range(res_block):
            deconv_layers.append(BasicBlock(output_channels, output_channels, activation=self.activation))

        return nn.Sequential(*deconv_layers)

    def init_weights(self, pretrained='', verbose=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x[0]
        out_0 = self.output_0(x)
        x = torch.cat((x, out_0), 1)
        out_1 = self.output_1(x)
        out = [out_0,
               out_1] # tag, heatmap_1(/4), heatmap_2(/2)
        return out

    def loss(self,
             preds,
             gt_meta,
             ):
        target_list = gt_meta['target_list']
        target_list[0] = torch.tensor(np.stack(target_list[0], axis=0)).to(gt_meta['img'].device)
        target_list[1] = torch.tensor(np.stack(target_list[1], axis=0)).to(gt_meta['img'].device)
        mask_list = gt_meta['mask_list']
        mask_list[0] = torch.tensor(np.stack(mask_list[0], axis=0)).to(gt_meta['img'].device)
        mask_list[1] = torch.tensor(np.stack(mask_list[1], axis=0)).to(gt_meta['img'].device)
        joints_list = gt_meta['joints_list']
        joints_list[0] = torch.tensor(np.stack(joints_list[0], axis=0)).to(gt_meta['img'].device)
        joints_list[1] = torch.tensor(np.stack(joints_list[1], axis=0)).to(gt_meta['img'].device)
        heatmaps_losses, push_losses, pull_losses = self.muti_loss(preds, target_list, mask_list, joints_list)
        loss = 0
        heat = 0
        push = 0
        pull = 0
        for idx in range(2):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                heat += heatmaps_loss
                loss = loss + heatmaps_loss
                if push_losses[idx] is not None:
                    push_loss = push_losses[idx].mean(dim=0)
                    push += push_loss
                    loss = loss + push_loss
                if pull_losses[idx] is not None:
                    pull_loss = pull_losses[idx].mean(dim=0)
                    pull += pull_loss
                    loss = loss + pull_loss
        loss_states = dict(loss=loss,
                           hm_loss=heat,
                           push_loss=push,
                           pull_losses=pull,
                           )
        return loss, loss_states


    def match_by_tag(self,inp):
        tag_k, loc_k, val_k = inp
        default_ = np.zeros((self.num_classes, 3 + tag_k.shape[2]))

        joint_dict = {}
        tag_dict = {}
        for i in range(self.num_classes):
            idx = self.joint_order[i]

            tags = tag_k[idx]
            joints = np.concatenate(
                (loc_k[idx], val_k[idx, :, None], tags), 1
            )
            mask = joints[:, 2] > self.detection_threshold
            tags = tags[mask]
            joints = joints[mask]

            if joints.shape[0] == 0:
                continue

            if i == 0 or len(joint_dict) == 0:
                for tag, joint in zip(tags, joints):
                    key = tag[0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                    tag_dict[key] = [tag]
            else:
                grouped_keys = list(joint_dict.keys())[:self.max_objs]
                grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

                # if self.ignore_too_much \
                # and len(grouped_keys) == self.max_objs:
                #     continue

                diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
                diff_normed = np.linalg.norm(diff, ord=2, axis=2)
                diff_saved = np.copy(diff_normed)

                # if self.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

                num_added = diff.shape[0]
                num_grouped = diff.shape[1]

                if num_added > num_grouped:
                    diff_normed = np.concatenate(
                        (
                            diff_normed,
                            np.zeros((num_added, num_added-num_grouped))+1e10
                        ),
                        axis=1
                    )

                pairs = py_max_match(diff_normed)
                for row, col in pairs:
                    if (
                        row < num_added
                        and col < num_grouped
                        and diff_saved[row][col] < self.tag_threshold
                    ):
                        key = grouped_keys[col]
                        joint_dict[key][idx] = joints[row]
                        tag_dict[key].append(tags[row])
                    else:
                        key = tags[row][0]
                        joint_dict.setdefault(key, np.copy(default_))[idx] = \
                            joints[row]
                        tag_dict[key] = [tags[row]]

        ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
        return ans

    def match(self, tag_k, loc_k, val_k):
        match = lambda x: self.match_by_tag(x)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    def refine(self, det, tag, keypoints):
        """
        Given initial keypoint predictions, we identify missing joints
        :param det: numpy.ndarray of size (17, 128, 128)
        :param tag: numpy.ndarray of size (17, 128, 128) if not flip
        :param keypoints: numpy.ndarray of size (17, 4) if not flip, last dim is (x, y, det score, tag score)
        :return:
        """
        if len(tag.shape) == 3:
            # tag shape: (17, 128, 128, 1)
            tag = tag[:, :, :, None]

        tags = []
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(np.int32)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        ans = []

        for i in range(keypoints.shape[0]):
            # score of joints i at all position
            tmp = det[i, :, :]
            # distance of all tag values with mean tag of current detected people
            tt = (((tag[i, :, :] - prev_tag[None, None, :]) ** 2).sum(axis=2) ** 0.5)
            tmp2 = tmp - np.round(tt)

            # find maximum position
            y, x = np.unravel_index(np.argmax(tmp2), tmp.shape)
            xx = x
            yy = y
            # detection score at maximum position
            val = tmp[y, x]
            # offset by 0.5
            x += 0.5
            y += 0.5

            # add a quarter offset
            if tmp[yy, min(xx + 1, tmp.shape[1] - 1)] > tmp[yy, max(xx - 1, 0)]:
                x += 0.25
            else:
                x -= 0.25

            if tmp[min(yy + 1, tmp.shape[0] - 1), xx] > tmp[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(det.shape[0]):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                # if ans[i, 2] > 0.01 and keypoints[i, 2] == 0:
                    keypoints[i, :2] = ans[i, :2]
                    keypoints[i, 2] = ans[i, 2]

        return keypoints

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        #print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)] > tmp[xx, max(yy-1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy] > tmp[max(0, xx-1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[batch_id][people_id, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def _nms(self,heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def top_k(self, det, tag):
        # det = torch.Tensor(det, requires_grad=False)
        # tag = torch.Tensor(tag, requires_grad=False)

        det = self._nms(det)
        num_images = det.size(0)
        num_joints = det.size(1)
        h = det.size(2)
        w = det.size(3)
        det = det.view(num_images, num_joints, -1)
        val_k, ind = det.topk(self.max_objs, dim=2)

        tag = tag.view(tag.size(0), tag.size(1), w*h, -1)
        # if not self.tag_per_joint:
        #     tag = tag.expand(-1, self.num_joints, -1, -1)

        tag_k = torch.stack(
            [
                torch.gather(tag[:, :, :, i], 2, ind)
                for i in range(tag.size(3))
            ],
            dim=3
        )

        x = ind % w
        y = (ind // w).long()

        ind_k = torch.stack((x, y), dim=3)

        ans = {
            'tag_k': tag_k.cpu().numpy(),
            'loc_k': ind_k.cpu().numpy(),
            'val_k': val_k.cpu().numpy()
        }

        return ans
    def parse(self, det, tag):
        ans = self.match(**self.top_k(det, tag))

        if self.ADJUST:
            ans = self.adjust(ans, det)
        scores = [i[:, 2].mean() for i in ans[0]]
        if self.ADJUST:
            ans = ans[0]
            # for every detected person
            for i in range(len(ans)):
                det_numpy = det[0].cpu().numpy()
                tag_numpy = tag[0].cpu().numpy()
                # if not self.tag_per_joint:
                #     tag_numpy = np.tile(
                #         tag_numpy, (self.num_classes, 1, 1, 1)
                #     )
                ans[i] = self.refine(det_numpy, tag_numpy, ans[i])
            ans = [ans]

        return ans, scores


    def decode(self, preds, meta, resize_keep_ratio):
        output_1 = torch.nn.functional.interpolate(
            preds[0],
            size=(preds[-1].size(2), preds[-1].size(3)),
            mode='bilinear',
            align_corners=False
        )
        offset_feat=self.num_classes
        heat_map=output_1[:,:offset_feat]
        tag=output_1[:,offset_feat:]
        heat_map=(preds[1]+heat_map)/2
        heat_map = torch.nn.functional.interpolate(
            heat_map,
            size=(meta['img'].shape[2], meta['img'].shape[3]),
            mode='bilinear',
            align_corners=False
        )
        tag = torch.nn.functional.interpolate(
            tag,
            size=(meta['img'].shape[2], meta['img'].shape[3]),
            mode='bilinear',
            align_corners=False
        )
        heat_map=heat_map.detach()
        # print(heat_map.max())
        tag=tag.detach()
        grouped, scores = self.parse(heat_map, tag)
        # print(scores)
        img_height = meta['img_info']['height'].cpu().numpy() \
            if isinstance(meta['img_info']['height'], torch.Tensor) else meta['img_info']['height']
        img_width = meta['img_info']['width'].cpu().numpy() \
            if isinstance(meta['img_info']['width'], torch.Tensor) else meta['img_info']['width']
        c = np.array([img_width / 2., img_height / 2.], dtype=np.float32).reshape(-1, 2)
        s = np.array([img_width, img_height], dtype=np.float32).reshape(-1, 2)
        final_res = self.get_final_preds(grouped, scores, c, s, [heat_map.size(3), heat_map.size(2)], resize_keep_ratio)

        return final_res

    def get_final_preds(self, grouped_joints, scores, center, scale, heatmap_size, resize_keep_ratio):
        final_results = []
        for i, person in enumerate(grouped_joints[0]):
            # joints = np.zeros((person.shape[0], 3))
            joints = transform_preds(person, center[0], scale[0], heatmap_size, resize_keep_ratio)
            result = []
            for js in joints:
                result += [float(j) for j in list(js)]
            result += [float(scores[i])]
            final_results.append(result)  # x, y, conf, tag
        return [{np.ones(1, dtype=np.int32)[0]: final_results}]

    def show_result(self, img, dets, class_names, score_thres=0.3, show=True, save_path=None):
        if len(dets)!=0:
            dets = dets[0]
            for label in dets:
                for bbox in dets[label]:
                    add_coco_hp_emb(img, bbox, score_thres)
        return img

def encoder(gt_meta, input_size, num_keypoints, heatmap_generator, joints_generator):
    joints = gt_meta["gt_keypoints"].reshape(-1, num_keypoints, 3)
    mask = gt_meta["gt_masks"]
    mask_list = []
    joints_list = []
    target_list = []
    for heat,joint in zip(heatmap_generator,joints_generator):
        ratio = input_size/heat.output_res
        mask_list.append(cv2.resize(mask, (heat.output_res, heat.output_res)))
        j = joints.copy()
        j[:,:,:2] = j[:,:,:2]/ratio
        target_list.append(heat(j))
        joints_list.append(joint(j).astype(np.int32))
    gt_meta['target_list'] = target_list
    gt_meta['mask_list'] = mask_list
    gt_meta['joints_list'] = joints_list
    # meta = {'img': gt_meta['img'],
    #         'target_list': target_list,
    #         'mask_list': mask_list,
    #         'joints_list': joints_list}
    return gt_meta

def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

# def get_affine_transform(center,
#                          scale,
#                          rot,
#                          output_size,
#                          shift=np.array([0, 0], dtype=np.float32),
#                          inv=0):
#     if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
#         print(scale)
#         scale = np.array([scale, scale])
#
#     scale_tmp = scale * 200.0
#     src_w = scale_tmp[0]
#     dst_w = output_size[0]
#     dst_h = output_size[1]
#
#     rot_rad = np.pi * rot / 180
#     src_dir = get_dir([0, src_w * -0.5], rot_rad)
#     dst_dir = np.array([0, dst_w * -0.5], np.float32)
#
#     src = np.zeros((3, 2), dtype=np.float32)
#     dst = np.zeros((3, 2), dtype=np.float32)
#     src[0, :] = center + scale_tmp * shift
#     src[1, :] = center + src_dir + scale_tmp * shift
#     dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
#     dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
#
#     src[2:, :] = get_3rd_point(src[0, :], src[1, :])
#     dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
#
#     if inv:
#         trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
#     else:
#         trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
#
#     return trans
#
# def affine_transform(pt, t):
#     new_pt = np.array([pt[0], pt[1], 1.]).T
#     new_pt = np.dot(t, new_pt)
#     return new_pt[:2]

def transform_preds(coords, center, scale, output_size, resize_keep_ratio=False):
    # target_coords = np.zeros(coords.shape)
    target_coords = coords.copy()
    trans = get_affine_transform(center, scale, 0, output_size, inv=1, resize_keep_ratio=resize_keep_ratio)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

if __name__ == '__main__':
    tensor = torch.rand((1,32,128,128))
    model = EmbeddingKeypointHead(32, 17, 4, 4)
    res = model([tensor])
    print(res)
