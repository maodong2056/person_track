import numpy as np
from src.utils import PersonItem, PartName
from src.utils.point_utils import tanAngle, disAndAngle
from src.utils.person_item import person_wrapper

class PersonReidFlag:
    changed = 1
    unchanged = 0

class PersonReidIdx:
    init_feature = 0
    middle_feature = 1
    edge_feature = 2



@person_wrapper
class PersonPointItem(PersonItem):
    def __init__(self, person_item,
                       joints_name):
        self.L = 1580
        self.joints_name = joints_name
        self.right_wrist_idx = self.joints_name.index("R_Wrist")
        self.left_wrist_idx = self.joints_name.index("L_Wrist")
        self.root_idx = self.joints_name.index("Pelvis")
        self.nose_idx = self.joints_name.index("Nose")

    def calculate(self):
        assert self._body_keypoint_3d is not None
        p_nose = self._body_keypoint_3d[self.nose_idx][:3].copy()
        p_hand = self._body_keypoint_3d[self.right_wrist_idx][:3].copy()
        p_nose[1] = -p_nose[1]
        p_hand[1] = -p_hand[1]
        direction = p_nose[0] - p_hand[0]
        ground = -np.max(self._body_keypoint_3d[:, 1])
        p_foot = p_nose.copy()
        p_foot[1] = ground
        _, _, tan_A = tanAngle(p_nose, p_hand, p_foot)
        radius_ = self.L * tan_A
        p_center_z = p_nose.copy()
        p_center_z[1] = 0
        p_center_z[2] = self._body_keypoint_3d[self.root_idx][2]
        p_camera_z = p_nose.copy()
        # p_center_z[0] = 0
        p_camera_z[1] = 0
        p_camera_z[2] = 0
        p_hand_z = p_hand.copy()
        p_hand_z[1] = 0
        root_dis = self._body_keypoint_3d[self.root_idx][2].copy()
        moving_dist, moving_direction = disAndAngle(p_center_z, p_camera_z, p_hand_z, radius_, root_dis, bias=self._body_keypoint_3d[self.root_idx][0])
        if direction < 0:
            moving_direction = -moving_direction
        return radius_, moving_dist, moving_direction


