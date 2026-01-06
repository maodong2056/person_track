import numpy as np
import cv2
from src.algorithm.module.traceFilter.LK import EmaFilter, OneEuroFilter
from src.algorithm.module.track import TrackModule

class TrackFilter():
    def __init__(self, diff_thres=4., smooth_box=0.3, mode='euro'):
        self.diff_thres = diff_thres
        self.alpha = smooth_box
        self.track_box = None
        self.previous_image = None
        self.track_dict = {}
        if mode == 'ema':
            self.filter_body = EmaFilter(self.alpha)
            self.filter_face = EmaFilter(self.alpha)
            self.filter_lh = EmaFilter(self.alpha)
            self.filter_rh = EmaFilter(self.alpha)
            self.filter_face_lm = EmaFilter(self.alpha)
            self.filter_lh_lm = EmaFilter(self.alpha)
            self.filter_rh_lm = EmaFilter(self.alpha)
        else:
            self.filter_body = OneEuroFilter()
            self.filter_face = OneEuroFilter()
            self.filter_lh = OneEuroFilter()
            self.filter_rh = OneEuroFilter()
            self.filter_face_lm = OneEuroFilter()
            self.filter_lh_lm = OneEuroFilter()
            self.filter_rh_lm = OneEuroFilter()


    def get_smooth_output(self, track: TrackModule, detection_setting: dict, image: np.ndarray):
        if self.diff_frames_crop(self.previous_image, image, self.track_box) or len(self.track_box)==0:
            output = track.get_result(image, **detection_setting)
            self.previous_image = image.copy()
            output = self.smooth_output(self.track_dict, output)
        else:
            output = self.track_box
            self.previous_image = image.copy()
        # if len(self.track_box)
        self.track_box = output
        self.track_dict = {
            # output[i, -1]: output[i] for i in range(len(output))
            output[i]["id"]: output[i] for i in range(len(output))
        }
        return output

    def smooth_output(self, previuous_output: dict, now_output: np.ndarray):
        for i in range(len(now_output)):
            out = now_output[i]
            # id = out[-1]
            id = out["id"]
            if id in previuous_output.keys():
                out["body_box"] = self.smooth(self.filter_body, out["body_box"], previuous_output[id]["body_box"])
                face_filter = (out["face_box"] is not None) and (previuous_output[id]["face_box"] is not None)
                out["face_box"] = self.smooth(self.filter_face, out["face_box"], previuous_output[id]["face_box"]) if face_filter else out["face_box"]
                out["face_lm"] = self.smooth(self.filter_face_lm, out["face_lm"], previuous_output[id]["face_lm"]) if face_filter else out["face_lm"]
                lh_filter = (out["lh_box"] is not None) and (previuous_output[id]["lh_box"] is not None)
                out["lh_box"] = self.smooth(self.filter_lh, out["lh_box"], previuous_output[id]["lh_box"]) if lh_filter else out["lh_box"]
                out["lh_lm"] = self.smooth(self.filter_lh_lm, out["lh_lm"], previuous_output[id]["lh_lm"]) if lh_filter else out["lh_lm"]
                rh_filter = (out["rh_box"] is not None) and (previuous_output[id]["rh_box"] is not None)
                out["rh_box"] = self.smooth(self.filter_rh, out["rh_box"], previuous_output[id]["rh_box"]) if rh_filter else out["rh_box"]
                out["rh_lm"] = self.smooth(self.filter_rh_lm, out["rh_lm"], previuous_output[id]["rh_lm"]) if rh_filter else out["rh_lm"]
                # now_output[i][4: 8] = fbox.astype(np.int)
        return now_output

    def smooth(self, filter, now_box, previous_box):
        return filter(now_box, previous_box)


    def diff_frames(self, previous_frame, image):
        '''
        diff value for two value,
        determin if to excute the detection

        :param previous_frame:  RGB  array
        :param image:           RGB  array
        :return:                True or False
        '''
        if previous_frame is None:
            return True
        else:

            _diff = cv2.absdiff(previous_frame, image)

            diff = np.sum(_diff) / previous_frame.shape[0] / previous_frame.shape[1] / 3.
            #print(diff)
            if diff > self.diff_thres:
                return True
            else:
                return False

    def diff_frames_crop(self, previous_frame, image, output):
        '''
        diff value for two value,
        determin if to excute the detection

        :param previous_frame:  RGB  array
        :param image:           RGB  array
        :return:                True or False
        '''
        if previous_frame is None:
            return True
        else:
            _diff = []
            for i in range(len(output)):
                # bbox = output[i, :4]
                bbox = output[i]["body_box"].astype(np.int).copy()
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, previous_frame.shape[1])
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, previous_frame.shape[0])
                try:
                    b_diff = cv2.absdiff(previous_frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :],
                                       image[bbox[1]:bbox[3], bbox[0]:bbox[2], :])
                    # cv2.imshow("prev", previous_frame[bbox[1]:bbox[3],bbox[0]:bbox[2], :])
                    # cv2.imshow("now", image[bbox[1]:bbox[3], bbox[0]:bbox[2], :])
                    _diff.append(np.sum(b_diff) / (bbox[3]-bbox[1])/(bbox[2]-bbox[0])/3)
            # _diff = cv2.absdiff(previous_frame, image)
                except:
                    print(1)

            # diff = np.sum(_diff) / previous_frame.shape[0] / previous_frame.shape[1] / 3.
            #print(_diff)
            if len(_diff)!=0:
                diff = max(_diff)
            else:
                diff = 1000
            if diff > self.diff_thres:
                return True
            else:
                return False