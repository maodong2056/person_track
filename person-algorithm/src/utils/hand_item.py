import numpy as np
class HandState:
    Exist = 1
    NExist = 0

class TrackingState:
    Tracking = 1
    UnTracking = 0

class HandPartName:
    # body_part = "body"
    # face_part = "face"
    # lefthand_part = "lh"
    # righthand_part = "rh"
    hand_part = 'hand'

class KeyPointType:
    Kps2D = 0
    Kps3D = 1

class HandItem(object):
    def __init__(self, hand_box: np.array, hand_conf: float):
        #added by zh,2022.03.31.单纯手的检测模型不知道左手还是右手
        self._hand_box = np.array(hand_box, dtype=np.float32)
        self._hand_conf = float(hand_conf)
        self._hand_lm = None
        self._hand_state = HandState.Exist

        self._hand_keypoint = None
        self._hand_keypoint_3d = None
        self._hand_keypoint_state = HandState.NExist
        self._hand_keypoint_type = KeyPointType.Kps2D

        self._hand_roi = None #added
        self._hand_tracking_state = TrackingState.UnTracking


    def update_roi(self, part_name: str, part_data: np.array):
            try:
                exec("assert self._{}_state == HandState.Exist".format(part_name))
                exec("self._{}_roi = part_data".format(part_name))
                return 1
            except:
                return 0

    def update_box(self, part_name: str, part_data: np.array, part_conf: float):
        try:
            exec("self._{}_box = part_data.astype(np.float32)".format(part_name))
            exec("self._{}_state = HandState.Exist".format(part_name))
            exec("self._{}_conf = float(part_conf)".format(part_name))
            return 1
        except:
            return 0

    def update_lm(self, part_name: str, part_data: np.array):
        try:
            part_data = part_data.reshape(-1, 2).astype(np.float32)
            exec("assert self._{}_state == HandState.Exist".format(part_name))
            exec("self._{}_lm = part_data".format(part_name))
            return 1
        except:
            return 0

    def update_keypoint(self, part_name: str, part_data: np.array, data_type: int, update_type=True):
        try:
            exec("assert self._{}_state == HandState.Exist".format(part_name))
            exec("self._{}_keypoint_state = HandState.Exist".format(part_name))
            if update_type:
                exec("self._{}_keypoint_type = data_type".format(part_name))
            if data_type == KeyPointType.Kps2D:
                exec("self._{}_keypoint = part_data.astype(np.float32)".format(part_name))
            else:
                exec("self._{}_keypoint_3d = part_data.astype(np.float32)".format(part_name))
            return 1
        except:
            return 0

    # def update_feature(self, part_data: np.array):
    #     self._feature = part_data

    # def update_trackid(self, track_id: int):
    #     self._track_id = track_id
    #
    def update_trackState(self, state: int):
        self._hand_tracking_state = state

    # def update_name(self, name: str):
    #     self._name = name

    def get_box(self, part_name: str):
        try:
            exec("assert self._{}_state == HandState.Exist".format(part_name))
            res = eval("self._{}_box".format(part_name))
            return res
        except:
            return None

    def get_roi(self, part_name: str):
        try:
            exec("assert self._{}_state == HandState.Exist".format(part_name))
            res = eval("self._{}_roi".format(part_name))
            return res
        except:
            return None

    def get_conf(self, part_name: str):
        try:
            exec("assert self._{}_state == HandState.Exist".format(part_name))
            res = eval("self._{}_conf".format(part_name))
            return res
        except:
            return None

    def get_lm(self, part_name: str):
        try:
            exec("assert self._{}_state == HandState.Exist".format(part_name))
            res = eval("self._{}_lm".format(part_name))
            return res
        except:
            return None

    def get_keypoint(self, part_name: str, keypoint_type=None):
        try:
            exec("assert self._{}_keypoint_state == HandState.Exist".format(part_name))
            kp_type = keypoint_type if keypoint_type is not None else self._hand_keypoint_type
            if kp_type == KeyPointType.Kps2D:
                res = eval("self._{}_keypoint".format(part_name))
            else:
                res = eval("self._{}_keypoint_3d".format(part_name))
            return res
        except:
            return None

    def get_keypoint_type(self, part_name: str):
        try:
            exec("assert self._{}_keypoint_state == HandState.Exist".format(part_name))
            res = eval("self._{}_keypoint_type".format(part_name))
            return res
        except:
            return None

    def get_track_state(self):
        try:
            res = (self._hand_tracking_state == TrackingState.Tracking)
            return res
        except:
            return None
    
    def get_part_state(self, part_name: str, keypoint=False):
        res = eval("self._{}_state".format(part_name)) if keypoint == False else \
              eval("self._{}_keypoint_state".format(part_name))
        if res == HandState.Exist:
            return True
        else:
            return False

    # def get_name(self):
    #     return self._name

    def update(self, hand_item):

        #-added------------------------------------------------------
        self._hand_box = hand_item._hand_box
        self._hand_conf = hand_item._hand_conf
        self._hand_lm = hand_item._hand_lm
        self._hand_state = hand_item._hand_state

        self._hand_keypoint = hand_item._hand_keypoint
        self._hand_keypoint_3d = hand_item._hand_keypoint_3d
        self._hand_keypoint_state = hand_item._hand_keypoint_state
        self._hand_keypoint_type = hand_item._hand_keypoint_type

        self._hand_roi = hand_item._hand_roi

def hand_wrapper(cls):
    def inner(hand_item, *args, **kwargs):
        new_cls = cls(hand_item,
                      *args,
                      **kwargs)
        super(cls, new_cls).__init__(hand_item.get_box(HandPartName.hand_part),
                                     hand_item.get_conf(HandPartName.hand_part))
        new_cls.update(hand_item)
        return new_cls
    return inner

if __name__ == '__main__':
    hand_a = HandItem(np.array([1,2,3,4]), 0.2)
    hand_a.update_box(HandPartName.hand_part, np.array([1, 5, 2, 7]), 0.3)
    print(hand_a.get_box(HandPartName.hand_part))

