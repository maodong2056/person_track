# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature, tag=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.tag = float(tag) if tag is not None else None
        # self.name = str(name) if name is not None else None
        # self.score = float(name_score) if name_score is not None else None

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

class FaceDetection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, landmark, tag=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.landmark = np.asarray(landmark, dtype=np.float)
        self.tag = float(tag) if tag is not None else None

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class PersonDetection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, person, tag=None):
        self.body_tlwh = np.asarray(self.xyxy_to_tlbr(person["body_box"]), dtype=np.float)
        self.body_confidence = float(person["body_conf"])
        self.body_tag = float(person["body_tag"])
        self.body_feature = np.asarray(person["reid"], dtype=np.float32)
        self.face_tlwh = np.asarray(self.xyxy_to_tlbr(person["face_box"]), dtype=np.float) \
            if person["face_box"] is not None else None
        self.face_confidence = float(person["face_conf"])\
            if person["face_conf"] is not None else None
        self.face_landmark = np.asarray(person["face_lm"], dtype=np.float)\
            if person["face_lm"] is not None else None
        self.face_tag = float(person["face_tag"])\
            if person["face_tag"] is not None else None
        self.det_res = person

    def xyxy_to_tlbr(self, bbox_xyxy):
        bbox_xyxy[2] = bbox_xyxy[2] - bbox_xyxy[0]
        bbox_xyxy[3] = bbox_xyxy[3] - bbox_xyxy[1]
        return bbox_xyxy

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        body_ret = self.body_tlwh.copy()
        body_ret[2:] += body_ret[:2]
        if self.face_tlwh is not None:
            face_ret = self.face_tlwh.copy()
            face_ret[2:] += face_ret[:2]
        else:
            face_ret = None
        return body_ret, face_ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        body_ret = self.body_tlwh.copy()
        body_ret[:2] += body_ret[2:] / 2
        body_ret[2] /= body_ret[3]
        if self.face_tlwh is not None:
            face_ret = self.face_tlwh.copy()
            face_ret[:2] += face_ret[2:] / 2
            face_ret[2] /= face_ret[3]
        else:
            face_ret = None
        return body_ret, face_ret