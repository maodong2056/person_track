import torch
import torch.nn as nn
import numpy as np
import cv2
def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    # topk_ys   = torch.floor_divide(topk_inds, width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1,   1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def mot_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds

def face_decode(fheat, fwh, freg=None, flm=None,cat_spec_fwh=False, K=100):
    batch, cat, height, width = fheat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    fheat = _nms(fheat)

    fscores, finds, fclses, fys, fxs = _topk(fheat, K=K)

    if freg is not None:
        freg = tranpose_and_gather_feat(freg, finds)
        freg = freg.view(batch, K, 2)
        fxs_bbox = fxs.view(batch, K, 1) + freg[:, :, 0:1]
        fys_bbox = fys.view(batch, K, 1) + freg[:, :, 1:2]
    else:
        fxs_bbox = fxs.view(batch, K, 1) + 0.5
        fys_bbox = fys.view(batch, K, 1) + 0.5
    fwh = tranpose_and_gather_feat(fwh, finds)

    if flm is not None:
        flm = tranpose_and_gather_feat(flm, finds)
        flm[:, :, [0,2,4,6,8]] = fxs.view(batch, K, 1).expand_as(flm[:, :, [0,2,4,6,8]]) + flm[:, :, [0,2,4,6,8]]
        flm[:, :, [1,3,5,7,9]] = fys.view(batch, K, 1).expand_as(flm[:, :, [1,3,5,7,9]]) + flm[:, :, [1,3,5,7,9]]

    if cat_spec_fwh:
        fwh = fwh.view(batch, K, 1, 2)
        fclses_ind = fclses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        fwh = fwh.gather(2, fclses_ind).view(batch, K, 2)
    else:
        fwh = fwh.view(batch, K, 2)

    fclses = fclses.view(batch, K, 1).float()
    fscores = fscores.view(batch, K, 1)
    fboxes = torch.cat([fxs_bbox - fwh[..., 0:1] / 2,
                        fys_bbox - fwh[..., 1:2] / 2,
                        fxs_bbox + fwh[..., 0:1] / 2,
                        fys_bbox + fwh[..., 1:2] / 2], dim=2)
    fdetections = torch.cat([fboxes, fscores, fclses, flm], dim=2)

    return fdetections, finds

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size,landmark=False):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        if not landmark:
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
        else:
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
            target_coords[p, 2:4] = affine_transform(coords[p, 2:4], trans)
            target_coords[p, 4:6] = affine_transform(coords[p, 4:6], trans)
            target_coords[p, 6:8] = affine_transform(coords[p, 6:8], trans)
            target_coords[p, 8:10] = affine_transform(coords[p, 8:10], trans)
    return target_coords

# def ctdet_post_process(dets, c, s, h, w, num_classes):
#     # dets: batch x max_dets x dim
#     # return 1-based class det dict
#     ret = []
#     for i in range(dets.shape[0]):
#         top_preds = {}
#         dets[i, :, :2] = transform_preds(
#               dets[i, :, 0:2], c[i], s[i], (w, h))
#         dets[i, :, 2:4] = transform_preds(
#               dets[i, :, 2:4], c[i], s[i], (w, h))
#         classes = dets[i, :, -1]
#         for j in range(num_classes):
#             inds = (classes == j)
#             top_preds[j + 1] = np.concatenate([
#                 dets[i, inds, :4].astype(np.float32),
#                 dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
#             ret.append(top_preds)
#     return ret

def ctdet_post_process(dets, c, s, h, w, num_classes, landmark=False):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
              dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
              dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, 5]
        if landmark:
            dets[i, :, 6:] = transform_preds(
              dets[i, :, 6:], c[i], s[i], (w, h),landmark)
        if not landmark:
            for j in range(num_classes):
                inds = (classes == j)
                top_preds[j + 1] = np.concatenate([
                  dets[i, inds, :4].astype(np.float32),
                  dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
            ret.append(top_preds)
        else:
            for j in range(num_classes):
                inds = (classes == j)
                top_preds[j + 1] = np.concatenate([
                  dets[i, inds, :4].astype(np.float32),
                  dets[i, inds, 4:5].astype(np.float32),
                  dets[i, inds, 6:].astype(np.float32)], axis=1).tolist()
            ret.append(top_preds)
    return ret



