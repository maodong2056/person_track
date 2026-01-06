# @Time : 2021/3/17 15:09 
# @Author : Altair.Huazj
# @File : bbox.py 
# @Software: PyCharm
import numpy as np

def xyxy2xywh(bbox):
    outbox = bbox.copy()
    assert (outbox[:, 2] > outbox[:, 0]).all()
    assert (outbox[:, 3] > outbox[:, 1]).all()
    outbox[:, 2] = outbox[:, 2] - outbox[:, 0]
    outbox[:, 3] = outbox[:, 3] - outbox[:, 1]
    return outbox

def xywh2xyxy(bbox):
    outbox = bbox.copy()
    outbox[:, 2] = outbox[:, 2] + outbox[:, 0]
    outbox[:, 3] = outbox[:, 3] + outbox[:, 1]
    return outbox

def xyxy2xcycwh(bbox):
    bx_1 = np.array(bbox.copy())
    if bx_1.shape[0]!=0:
        w = bx_1[:, 2] - bx_1[:, 0]
        h = bx_1[:, 3] - bx_1[:, 1]
        xc = bx_1[:, 0] + w / 2.
        yc = bx_1[:, 1] + h / 2.
        bx_1[:, :4] = np.c_[xc, yc, w, h]
    return bx_1

def xcycwh2xyxy(bbox):
    bx_1 = np.array(bbox.copy())
    if bx_1.shape[0]!=0:
        x1 = bx_1[:, 0] - bx_1[:, 2] / 2.
        y1 = bx_1[:, 1] - bx_1[:, 3] / 2.
        x2 = bx_1[:, 0] + bx_1[:, 2] / 2.
        y2 = bx_1[:, 1] + bx_1[:, 3] / 2.
        bx_1[:, :4] = np.c_[x1, y1, x2, y2]
    return bx_1