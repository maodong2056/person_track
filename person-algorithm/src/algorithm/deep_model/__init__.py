from .detection import get_body_det_model
from .keypoint import get_pose_2d_model
from .keypoint import get_pose_3d_model
from .recognition import get_body_rec_model
from .recognition import get_face_rec_model
from .track import get_track_model
from .detection import get_hand_det_model
from .keypoint import get_hand_2d_model

class MODEL_TASK:
    TASK_BODY = "body_det"
    TASK_2DKPS = "pose_2d"
    TASK_3DKPS = "pose_3d"
    TASK_REID = "body_rec"
    TASK_FACEREC = "face_rec"
    TASK_TRACK = "track"
    TASK_HANDDET = "hand_det"
    TASK_HAND2DKPS = "hand_2d"

def get_model(TASK_NAME, *args, **kwargs):
    return eval("get_{}_model".format(TASK_NAME))(*args, **kwargs)
