import numpy as np

from src.algorithm.deepModel.deepSort.sort.nn_matching import NearestNeighborDistanceMetric
from src.algorithm.deepModel.deepSort.sort.preprocessing import non_max_suppression
from src.algorithm.deepModel.deepSort.sort.detection import Detection, FaceDetection, PersonDetection
from src.algorithm.deepModel.deepSort.sort.tracker import Tracker
from src.algorithm.deepModel.deepSort.sort.tracker_single import Tracker as TrackerSingle

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, max_dist=0.2, single_mode=False, *args, **kwargs):
        self.min_confidence = 0.3
        self.nms_max_overlap = 0.5


        self.max_cosine_distance = max_dist
        self.nn_budget = 100
        self.metric = NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric) if not single_mode else TrackerSingle(self.metric)
        self.single_mode = single_mode

    def update_body(self, bbox_xyxy, confidences, features, img_size, btag=None):
        self.height, self.width = img_size[:2]
        # generate detections
        # features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        detections = [Detection(bbox_tlwh[i], confidences[i], features[i],
                                tag=None if btag is None else btag[i]) for i, _ in enumerate(bbox_tlwh)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        # outputs = []
        # for track in self.tracker.tracks:
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue
        #     box = track.to_tlwh()
        #     x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
        #     track_id = track.track_id
        #     outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        # if len(outputs) > 0:
        #     outputs = np.stack(outputs,axis=0)
        # return outputs

    def update_face(self, fbbox_xyxy, confidences, landmarks, ftag=None, tag_match=False, tag_thres=0.5):
        bbox_tlwh = self._xyxy_to_tlwh(fbbox_xyxy)
        fdetections = [FaceDetection(bbox_tlwh[i], confidences[i], landmarks[i],
                                     tag=None if ftag is None else ftag[i]) for i, _ in enumerate(bbox_tlwh)]
        self.tracker.face_update(fdetections, tag_match, tag_thres)

    def update_person(self, res, img_size):
        self.height, self.width = img_size[:2]
        person_detection = [PersonDetection(res[i]) for i, _ in enumerate(res)]
        self.tracker.predict_person()
        self.tracker.update_person(person_detection)

    def update_all(self, bbox_xyxy, bconf, features,
                   fbbox_xyxy, fconf, landmarks,
                   btag=None, ftag=None, tag_match=False, tag_thres=0.5,
                   img_size=(720, 1280, 3)):
        self.update_body(bbox_xyxy, bconf, features, img_size, btag)
        self.update_face(fbbox_xyxy, fconf, landmarks, ftag, tag_match, tag_thres)
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            output = []
            box = track.to_tlwh()
            # body_conf = track.body_conf
            bbox = self._tlwh_to_xyxy(box)
            output += bbox
            if track.is_face_confirmed():
                fbox = track.face_bbox.to_tlbr()
                # face_conf = [track.face_bbox.confidence]
                flm = track.face_bbox.landmark
            else:
                fbox = [-1, -1, -1, -1]
                # face_conf = [-1]
                flm = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            output += list(fbox) + list(flm)
            track_id = track.track_id
            output += [track_id]
            outputs.append(np.array(output, dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def update_notag(self, res, img_size=(720, 1280, 3)):
        self.update_person(res, img_size)
        outputs = []
        if not self.single_mode:
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                # output = []
                # bbox, fbox = track.to_tlwh_person()
                # body_conf = track.body_conf
                # bbox = self._tlwh_to_xyxy(bbox)
                outputs.append(track.to_person())
                # output += bbox
                # if track.is_face_confirmed():
                #     fbox = self._tlwh_to_xyxy(fbox)
                #     # face_conf = [track.face_bbox.confidence]
                #     flm = track.face_landmark
                # else:
                #     fbox = [-1, -1, -1, -1]
                #     # face_conf = [-1]
                #     flm = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                # output += list(fbox) + list(flm)
                # track_id = track.track_id
                # output += [track_id]
                # outputs.append(np.array(output, dtype=np.int))
        else:
            if self.tracker.track is not None:
                track = self.tracker.track
                if not (not track.is_confirmed() or track.time_since_update > 1):
                    # output = []
                    outputs.append(track.to_person())
                    # bbox, fbox = track.to_tlwh_person()
                    # body_conf = track.body_conf
                    # bbox = self._tlwh_to_xyxy(bbox)
                    # output += bbox
                    # if track.is_face_confirmed():
                    #     fbox = self._tlwh_to_xyxy(fbox)
                        # face_conf = [track.face_bbox.confidence]
                        # flm = track.face_landmark
                    # else:
                    #     fbox = [-1, -1, -1, -1]
                        # face_conf = [-1]
                        # flm = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                    # output += list(fbox) + list(flm)
                    # track_id = track.track_id
                    # output += [track_id]
                    # outputs.append(np.array(output, dtype=np.int))
        # if len(outputs) > 0:
        #     outputs = np.stack(outputs, axis=0)
        return outputs
    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        bbox_xywh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_xywh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_xywh

    @staticmethod
    def _xyxy_to_tlwh(bbox_xyxy):
        bbox_xyxy[:,2] = bbox_xyxy[:,2] - bbox_xyxy[:,0]
        bbox_xyxy[:,3] = bbox_xyxy[:,3] - bbox_xyxy[:,1]
        return bbox_xyxy

    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def clear(self):
        # del self.tracker
        # self.tracker = Tracker(self.metric)
        self.tracker.clear()
    # def _get_features(self, bbox_xywh, ori_img):
    #     im_crops = []
    #     for box in bbox_xywh:
    #         x1,y1,x2,y2 = self._xywh_to_xyxy(box)
    #         im = ori_img[y1:y2,x1:x2]
    #         im_crops.append(im)
    #     if im_crops:
    #         features = self.extractor(im_crops)
    #     else:
    #         features = np.array([])
    #     return features


