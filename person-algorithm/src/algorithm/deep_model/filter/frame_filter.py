import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from .tracker import EmaFilter, OneEuroFilter
from src.utils import PersonItem, PartName, KeyPointType
from src.utils.iou_cal import iou_cal_xyxy

class FrameFilter(object):
    def __init__(self,
                 diff_thres=4.,
                 smooth_box=0.3,
                 mode='euro',
                 filter_bbox=True,
                 filter_fbox=True,
                 filter_lhbox=True,
                 filter_rhbox=True,  #filter boxes
                 filter_flm=True,
                 filter_lhlm=True,
                 filter_rhlm=True,   #filter landmark
                 filter_keypoint=True,
                 filter_3dkeypoint=False,
                 iou_thres=0.6,
                 use_id_match=True,
                 ):
        self.diff_thres = diff_thres
        self.alpha = smooth_box
        self.previous_result = None
        self.previous_image = None
        self.iou_thres = iou_thres
        self.use_id_match = use_id_match
        self.filter = EmaFilter if mode == "ema" else OneEuroFilter
        self.args = [self.alpha] if mode == "ema" else []
        self.box_filters = {}
        self.lm_filters = {}
        self.kp_filters = {}
        self.kp3d_filters = {}
        if filter_bbox:
            self.box_filters[PartName.body_part] = self.filter(*self.args)
        if filter_fbox:
            self.box_filters[PartName.face_part] = self.filter(*self.args)
        if filter_lhbox:
            self.box_filters[PartName.lefthand_part] = self.filter(*self.args)
        if filter_rhbox:
            self.box_filters[PartName.righthand_part] = self.filter(*self.args)
        if filter_flm:
            self.lm_filters[PartName.face_part] = self.filter(*self.args)
        if filter_lhlm:
            self.lm_filters[PartName.lefthand_part] = self.filter(*self.args)
        if filter_rhlm:
            self.lm_filters[PartName.righthand_part] = self.filter(*self.args)
        if filter_keypoint:
            self.kp_filters[PartName.body_part] = self.filter(*self.args)
        if filter_3dkeypoint:
            self.kp3d_filters[PartName.body_part] = self.filter(*self.args)


    def _get_match(self, inputs, previous_inputs):
        if self.use_id_match:
            inp_ids = [inp.get_track_id() for inp in inputs]
            pre_ids = [pre.get_track_id() for pre in previous_inputs]
            match = [(i_id, pre_ids.index(i_id)) for i_id in inp_ids if i_id in pre_ids]
        else:
            inp_box = np.array([inp.get_box(PartName.body_part) for inp in inputs])
            pre_box = np.array([pre.get_box(PartName.body_part) for pre in previous_inputs])
            iou = iou_cal_xyxy(inp_box, pre_box)
            cost_matrix = 1 - iou
            cost_matrix[iou < self.iou_thres] = 1001
            row_indices, col_indices = linear_assignment(cost_matrix)
            match, unmatched_inp, unmatched_pre = [], [], []
            for col, bbox in enumerate(pre_box):
                if col not in col_indices:
                    unmatched_pre.append(col)
            for row, fbox in enumerate(inp_box):
                if row not in row_indices:
                    unmatched_inp.append(row)
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] > 1000:
                    unmatched_pre.append(col)
                    unmatched_inp.append(row)
                else:
                    match.append((row, col))
        return match


    def get_output(self, image, inputs):
        # first input image
        if self.previous_image is None:
            self.previous_image = image
            self.previous_result = inputs
            return inputs
        if len(inputs) == 0:
            self.previous_image = image
            return inputs
        matched_idx = self._get_match(inputs, self.previous_result)
        inputs = self._smooth(image, inputs, self.previous_result, matched_idx)
        self.previous_result = inputs
        self.previous_image = image
        return inputs


    def _diff_frame_judge(self, now_frame, pre_frame, inp, pre):
        inp_box = inp.get_box(PartName.body_part).astype(np.int)
        pre_box = pre.get_box(PartName.body_part).astype(np.int)
        inp_box[[0, 2]] = np.clip(inp_box[[0, 2]], 0, pre_frame.shape[1])
        inp_box[[1, 3]] = np.clip(inp_box[[1, 3]], 0, pre_frame.shape[0])
        pre_box[[0, 2]] = np.clip(pre_box[[0, 2]], 0, pre_frame.shape[1])
        pre_box[[1, 3]] = np.clip(pre_box[[1, 3]], 0, pre_frame.shape[0])
        b_diff = cv2.absdiff(pre_frame[pre_box[1]:pre_box[3], pre_box[0]:pre_box[2], :],
                             now_frame[pre_box[1]:pre_box[3], pre_box[0]:pre_box[2], :])
        _diff = np.sum(b_diff) / (pre_box[3] - pre_box[1]) / (pre_box[2] - pre_box[0]) / 3
        if _diff > self.diff_thres:
            return True
        else:
            return False


    def _smooth(self, image, inputs, previous, matches):
        for m in matches:
            inp = inputs[m[0]]
            pre = previous[m[1]]
            if self._diff_frame_judge(image, self.previous_image, inp, pre):
                for partname in self.box_filters.keys():
                    box_filter = self.box_filters[partname]
                    inp_box = inp.get_box(partname)
                    pre_box = pre.get_box(partname)
                    if inp_box is not None and pre_box is not None:
                        new_box = box_filter(inp_box, pre_box)
                        inp.update_box(partname, new_box, inp.get_conf(partname))
                for partname in self.lm_filters.keys():
                    lm_filter = self.lm_filters[partname]
                    inp_lm = inp.get_lm(partname)
                    pre_lm = pre.get_lm(partname)
                    if inp_lm is not None and pre_lm is not None:
                        new_lm = lm_filter(inp_lm, pre_lm)
                        inp.update_lm(partname, new_lm)
                for partname in self.kp_filters.keys():
                    kp_filter = self.kp_filters[partname]
                    inp_kp = inp.get_keypoint(partname, KeyPointType.Kps2D)
                    pre_kp = pre.get_keypoint(partname, KeyPointType.Kps2D)
                    if inp_kp is not None and pre_kp is not None:
                        new_lm = kp_filter(inp_kp, pre_kp)
                        inp.update_keypoint(partname, new_lm, KeyPointType.Kps2D, update_type=False)
                for partname in self.kp3d_filters.keys():
                    kp3d_filter = self.kp3d_filters[partname]
                    inp_kp = inp.get_keypoint(partname, KeyPointType.Kps3D)
                    pre_kp = pre.get_keypoint(partname, KeyPointType.Kps3D)
                    if inp_kp is not None and pre_kp is not None:
                        new_lm = kp3d_filter(inp_kp, pre_kp)
                        inp.update_keypoint(partname, new_lm, KeyPointType.Kps3D, update_type=False)
            else:
                inp.update(pre)
        return inputs
