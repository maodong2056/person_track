import numpy as np
class PersonState:
    Exist = 1
    NExist = 0

class TrackingState:
    Tracking = 1
    UnTracking = 0

class PartName:
    body_part = "body"
    face_part = "face"
    lefthand_part = "lh"
    righthand_part = "rh"

class KeyPointType:
    Kps2D = 0
    Kps3D = 1

class PersonItem(object):
    def __init__(self, body_box: np.array, body_conf: float):
        self._body_box = np.array(body_box, dtype=np.float32)
        self._body_conf = float(body_conf)
        self._body_state = PersonState.Exist
        self._face_box = None
        self._face_conf = 0
        self._face_lm = None
        self._face_state = PersonState.NExist
        self._lh_box = None
        self._lh_conf = 0
        self._lh_lm = None
        self._lh_state = PersonState.NExist
        self._rh_box = None
        self._rh_conf = 0
        self._rh_lm = None
        self._rh_state = PersonState.NExist
        self._body_keypoint = None
        self._body_keypoint_3d = None
        self._body_keypoint_state = PersonState.NExist
        self._body_keypoint_type = KeyPointType.Kps2D
        self._lh_keypoint = None
        self._lh_keypoint_3d = None
        self._lh_keypoint_state = PersonState.NExist
        self._lh_keypoint_type = KeyPointType.Kps2D
        self._rh_keypoint = None
        self._rh_keypoint_3d = None
        self._rh_keypoint_state = PersonState.NExist
        self._rh_keypoint_type = KeyPointType.Kps2D
        self._body_feature = None
        self._face_feature = None
        self._track_id = None
        self._tracking_state = TrackingState.UnTracking
        self._name = None

    def update_box(self, part_name: str, part_data: np.array, part_conf: float):
        try:
            exec("self._{}_box = part_data.astype(np.float32)".format(part_name))
            exec("self._{}_state = PersonState.Exist".format(part_name))
            exec("self._{}_conf = float(part_conf)".format(part_name))
            return 1
        except:
            return 0

    def update_lm(self, part_name: str, part_data: np.array):
        try:
            part_data = part_data.reshape(-1, 2).astype(np.float32)
            exec("assert self._{}_state == PersonState.Exist".format(part_name))
            exec("self._{}_lm = part_data".format(part_name))
            return 1
        except:
            return 0

    def update_keypoint(self, part_name: str, part_data: np.array, data_type: int, update_type=True):
        try:
            exec("assert self._{}_state == PersonState.Exist".format(part_name))
            exec("self._{}_keypoint_state = PersonState.Exist".format(part_name))
            if update_type:
                exec("self._{}_keypoint_type = data_type".format(part_name))
            if data_type == KeyPointType.Kps2D:
                exec("self._{}_keypoint = part_data.astype(np.float32)".format(part_name))
            else:
                exec("self._{}_keypoint_3d = part_data.astype(np.float32)".format(part_name))
            return 1
        except:
            return 0

    def update_feature(self, part_name: str, part_data: np.array):
        try:
            exec("self._{}_feature = part_data.astype(np.float32)".format(part_name))
            return 1
        except:
            return 0

    def update_trackid(self, track_id: int):
        self._track_id = track_id

    def update_trackState(self, state: int):
        self._tracking_state = state

    def update_name(self, name: str):
        self._name = name

    def get_box(self, part_name: str):
        try:
            exec("assert self._{}_state == PersonState.Exist".format(part_name))
            res = eval("self._{}_box".format(part_name))
            return res
        except:
            return None

    def get_conf(self, part_name: str):
        try:
            exec("assert self._{}_state == PersonState.Exist".format(part_name))
            res = eval("self._{}_conf".format(part_name))
            return res
        except:
            return None

    def get_lm(self, part_name: str):
        try:
            exec("assert self._{}_state == PersonState.Exist".format(part_name))
            res = eval("self._{}_lm".format(part_name))
            return res
        except:
            return None

    def get_keypoint(self, part_name: str, keypoint_type=None):
        try:
            exec("assert self._{}_keypoint_state == PersonState.Exist".format(part_name))
            kp_type = keypoint_type if keypoint_type is not None else self._body_keypoint_type
            if kp_type == KeyPointType.Kps2D:
                res = eval("self._{}_keypoint".format(part_name))
            else:
                res = eval("self._{}_keypoint_3d".format(part_name))
            return res
        except:
            return None

    def get_keypoint_type(self, part_name: str):
        try:
            exec("assert self._{}_keypoint_state == PersonState.Exist".format(part_name))
            res = eval("self._{}_keypoint_type".format(part_name))
            return res
        except:
            return None

    def get_feature(self, part_name: str):
        try:
            exec("assert self._{}_state == PersonState.Exist".format(part_name))
            exec("assert self._{}_feature is not None".format(part_name))
            res = eval("self._{}_feature".format(part_name))
            return res
        except:
            return None

    def get_track_id(self):
        try:
            assert self._body_state == PersonState.Exist
            assert self._track_id is not None
            res = self._track_id
            return res
        except:
            return None

    def get_track_state(self):
        try:
            res = (self._tracking_state == TrackingState.Tracking)
            return res
        except:
            return None
    
    def get_part_state(self, part_name: str, keypoint=False):
        res = eval("self._{}_state".format(part_name)) if keypoint == False else \
              eval("self._{}_keypoint_state".format(part_name))
        if res == PersonState.Exist:
            return True
        else:
            return False

    def get_name(self):
        return self._name

    def update(self, person_item):
        self._body_box = person_item._body_box
        self._body_conf = person_item._body_conf
        self._body_state = person_item._body_state
        self._face_box = person_item._face_box
        self._face_conf = person_item._face_conf
        self._face_lm = person_item._face_lm
        self._face_state = person_item._face_state
        self._lh_box = person_item._lh_box
        self._lh_conf = person_item._lh_conf
        self._lh_lm = person_item._lh_lm
        self._lh_state = person_item._lh_state
        self._rh_box = person_item._rh_box
        self._rh_conf = person_item._rh_conf
        self._rh_lm = person_item._rh_lm
        self._rh_state = person_item._rh_state
        self._body_keypoint = person_item._body_keypoint
        self._body_keypoint_3d = person_item._body_keypoint_3d
        self._body_keypoint_state = person_item._body_keypoint_state
        self._body_keypoint_type = person_item._body_keypoint_type
        self._lh_keypoint = person_item._lh_keypoint
        self._lh_keypoint_3d = person_item._lh_keypoint_3d
        self._lh_keypoint_state = person_item._lh_keypoint_state
        self._lh_keypoint_type = person_item._lh_keypoint_type
        self._rh_keypoint = person_item._rh_keypoint
        self._rh_keypoint_3d = person_item._rh_keypoint_3d
        self._rh_keypoint_state = person_item._rh_keypoint_state
        self._rh_keypoint_type = person_item._rh_keypoint_type
        self._body_feature = person_item._body_feature
        self._face_feature = person_item._face_feature
        self._track_id = person_item._track_id
        self._tracking_state = person_item._tracking_state

def person_wrapper(cls):
    def inner(person_item, *args, **kwargs):
        new_cls = cls(person_item,
                      *args,
                      **kwargs)
        super(cls, new_cls).__init__(person_item.get_box(PartName.body_part),
                                     person_item.get_conf(PartName.body_part))
        new_cls.update(person_item)
        return new_cls
    return inner

if __name__ == '__main__':
    person_a = PersonItem(np.array([1,2,3,4]), 0.2)
    person_a.update_box(PartName.face_part, np.array([1, 5, 2, 7]), 0.3)
    print(person_a.get_box(PartName.body_part))

