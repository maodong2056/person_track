"""
Create by Chengqi.Lv
2020/4/14
"""
import torch
import torch.nn as nn


def bbox_overlaps_diou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]  # 第一个框的个数
    cols = bboxes2.shape[0]  # 第二个框的个数
    dious = torch.zeros((rows, cols))  # 初始化dious变量
    if rows * cols == 0:  #
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True
    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2  # （x1max +x1min）/2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2  # (y1max+y1min)/2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # min((x1max,y1max ),(x2max,y2max)) ->返回较小一组
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # max((x1min,y1min ),(x2min,y2min))->返回较大的一组
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2  # 中心点距离
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)  # 外接矩对角线距离
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious


class DIOULoss(nn.Module):
    def __init__(self):
        super(DIOULoss, self).__init__()

    def forward(self, pred_bbox, target_bbox, weight, avg_factor):
        ious = bbox_overlaps_diou(pred_bbox, target_bbox)
        loss = 1 - ious
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss