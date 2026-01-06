import numpy as np

def iou_cal(bbox_1, bbox_2):
    """
    :param bbox_1: input(m, 4) (xc ,yc ,w, h)
    :param bbox_2: input(n, 4)
    :return: output(m, n)
    """
    bx_1 = np.array(bbox_1.copy())[:, np.newaxis, :]
    bx_2 = np.array(bbox_2.copy())[np.newaxis, :, :]


    bx_1 = np.c_[bx_1[..., [0, 1]] - bx_1[..., [2, 3]] / 2., bx_1[..., [0, 1]] + bx_1[..., [2, 3]] / 2.]
    bx_2 = np.c_[bx_2[..., [0, 1]] - bx_2[..., [2, 3]] / 2., bx_2[..., [0, 1]] + bx_2[..., [2, 3]] / 2.]
    area_bx_1 = (bx_1[..., 2] - bx_1[..., 0]) * (bx_1[..., 3] - bx_1[..., 1])
    area_bx_2 = (bx_2[..., 2] - bx_2[..., 0]) * (bx_2[..., 3] - bx_2[..., 1])

    tl = np.maximum(bx_1[..., :2], bx_2[..., :2])
    br = np.minimum(bx_1[..., 2:], bx_2[..., 2:])
    wh = np.maximum(0., br - tl)

    area_intersection = wh[..., 0] * wh[..., 1]
    return area_intersection / (area_bx_1 + area_bx_2 - area_intersection)

def iou_cal_xyxy(bbox_1, bbox_2):
    """
    :param bbox_1: input(m, 4) (xc ,yc ,w, h)
    :param bbox_2: input(n, 4)
    :return: output(m, n)
    """
    if len(bbox_1) == 0 or len(bbox_2) == 0:
        return np.zeros((len(bbox_1), len(bbox_2)))
    bx_1 = np.array(bbox_1.copy())[:, np.newaxis, :]
    bx_2 = np.array(bbox_2.copy())[np.newaxis, :, :]


    bx_1 = np.c_[bx_1[..., [0, 1]], bx_1[..., [2, 3]]]
    bx_2 = np.c_[bx_2[..., [0, 1]], bx_2[..., [2, 3]]]
    area_bx_1 = (bx_1[..., 2] - bx_1[..., 0]) * (bx_1[..., 3] - bx_1[..., 1])
    area_bx_2 = (bx_2[..., 2] - bx_2[..., 0]) * (bx_2[..., 3] - bx_2[..., 1])

    tl = np.maximum(bx_1[..., :2], bx_2[..., :2])
    br = np.minimum(bx_1[..., 2:], bx_2[..., 2:])
    wh = np.maximum(0., br - tl)

    area_intersection = wh[..., 0] * wh[..., 1]
    return area_intersection / (area_bx_1 + area_bx_2 - area_intersection)
