# vim: expandtab:ts=4:sw=4
import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, tag=None, face_mean=None, face_covariance=None, face_tag=None,
                 face_landmark=None, detection=None):
        self.mean = mean
        self.covariance = covariance
        self.face_mean = face_mean
        self.face_covariance = face_covariance

        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative

        self.face_state = 0
        self.face_since_update = 1
        self.body_conf = 0
        # self.name = None
        # self.name_score = None
        # self.name_body = None
        # self.name_body_score = None
        self.features = []
        self.tag = float(tag) if tag is not None else None
        self.face_tag = float(face_tag) if face_tag is not None else None
        self.face_bbox = None
        self.face_landmark = face_landmark
        self.face_conf = 0.
        self.detection = detection
        if self.detection is not None:
            self.detection.det_res["id"] = self.track_id
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlwh_person(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        if self.face_mean is not None:
            face_ret = self.face_mean[:4].copy()
            face_ret[2] *= face_ret[3]
            face_ret[:2] -= face_ret[2:] / 2
        else:
            face_ret = None
        return ret, face_ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def to_tlbr_person(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret, face_ret = self.to_tlwh_person()
        # ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        if face_ret is not None:
            face_ret[2:] = face_ret[:2] + face_ret[2:]
        else:
            face_ret = None
        return ret, face_ret

    def to_person(self):
        bbox, fbox = self.to_tlwh_person()
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        self.detection.det_res["body_box"] = bbox
        if self.detection.det_res["face_box"] is not None:
            fbox[2] = fbox[0] + fbox[2]
            fbox[3] = fbox[1] + fbox[3]
            self.detection.det_res["face_box"] = fbox
        self.detection.det_res["id"] = self.track_id
        return self.detection.det_res

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self.face_state = 0


    def predict_person(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        if self.face_mean is not None:
            self.face_mean, self.face_covariance = \
                kf.predict(self.face_mean, self.face_covariance)
        self.age += 1
        self.time_since_update += 1
        self.face_state = 0
        self.face_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.body_feature)
        self.tag = detection.body_tag
        self.body_conf = detection.body_confidence

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        # if self.name_body_score is not None:
        #     if detection.score < self.name_body_score:
        #         self.name_body = detection.name
        #         self.name_body_score = detection.score
        # else:
        #     self.name_body = detection.name
        #     self.name_body_score = detection.score

    def update_person(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        body_ret, face_ret = detection.to_xyah()
        self.detection = detection
        #####################update body####################################
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, body_ret)
        self.features.append(detection.body_feature)
        self.tag = detection.body_tag
        self.body_conf = detection.body_confidence
        self.hits += 1
        self.time_since_update = 0
        #####################update face####################################
        if face_ret is not None:
            self.face_since_update = 0
            self.face_state = 1
            self.face_landmark = detection.face_landmark
            self.face_tag = detection.face_tag
            self.face_conf = detection.face_confidence
        if self.face_mean is not None:
            if face_ret is not None:
                self.face_mean, self.face_covariance = kf.update(
                    self.face_mean, self.face_covariance, face_ret)
            else:
                if self.face_since_update > 5:
                    self.face_mean, self.face_covariance = None, None
                    self.face_landmark = None
                    self.face_tag = None
                    self.face_conf = None
        else:
            if face_ret is not None:
                self.face_mean, self.face_covariance = kf.initiate(face_ret)

        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def face_update(self, fdetection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.face_bbox = fdetection
        self.face_state = 1
        # if self.name_score is not None:
        #     if fdetection.score < self.name_score:
        #         self.name = fdetection.name
        #         self.name_score = fdetection.score
        # else:
        #     self.name = fdetection.name
        #     self.name_score = fdetection.score
        # self.hits += 1
        # self.time_since_update = 0
        # self.hits += 1
        # self.time_since_update = 0
        # if self.state == TrackState.Tentative and self.hits >= self._n_init:
        #     self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def is_face_confirmed(self):
        return self.face_state == 1

    # def is_have_face(self):
    #     return self.name is not None
    #
    # def is_name_confirm(self):
    #     return self.name == self.name_body
