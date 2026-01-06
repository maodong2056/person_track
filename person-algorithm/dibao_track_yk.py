import argparse
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rospy

from src import Video
from src.utils import PartName
from src.algorithm.api import PersonDetection, PersonMutiTrack
from src.algorithm.api import PersonFollow, PersonTrackerState
from src.algorithm.api.person_follow.person_item_track import PersonFollowItem
from src.utils import draw_person_bbox
from prediction.msg import Pose as FollowPose


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    person_height_m: float = 1.7
    desired_distance_m: float = 0.9
    max_forward_m: float = 0.7
    max_back_m: float = 0.4
    max_lateral_m: float = 0.4
    min_bbox_px: float = 20.0
    min_distance_m: float = 0.3
    max_distance_m: float = 3.5


person_det = PersonDetection("user/settings/model/detection/body_detection/centernet_lite_8down_ratio_old.json")
person_track = PersonMutiTrack("user/settings/model/track/deep_sort.json")
person_follow = PersonFollow(
    "user/settings/model/recognition/body_recognition/pcb_reid_mobilev2.json",
    person_item=PersonFollowItem,
)


def build_target_from_bbox(bbox, camera: CameraModel):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    bbox_h = max(y2 - y1, camera.min_bbox_px)

    distance_m = camera.person_height_m * camera.fy / bbox_h
    distance_m = float(np.clip(distance_m, camera.min_distance_m, camera.max_distance_m))
    lateral_m = (cx - camera.cx) * distance_m / camera.fx
    lateral_m = float(np.clip(lateral_m, -camera.max_lateral_m, camera.max_lateral_m))

    forward_error = distance_m - camera.desired_distance_m
    forward_m = float(np.clip(forward_error, -camera.max_back_m, camera.max_forward_m))
    return forward_m, -lateral_m


def build_follow_pose(forward, lateral):
    pose = FollowPose()
    pose.header.stamp = rospy.Time.now()
    pose.x = forward
    pose.y = lateral
    pose.theta = math.atan2(lateral, max(forward, 1e-3))
    return pose


def build_camera_model(args):
    fx = args.fx if args.fx > 0 else args.width * 0.6
    fy = args.fy if args.fy > 0 else args.height * 0.6
    cx = args.cx if args.cx > 0 else args.width / 2.0
    cy = args.cy if args.cy > 0 else args.height / 2.0
    return CameraModel(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        person_height_m=args.person_height,
        desired_distance_m=args.desired_distance,
        max_forward_m=args.max_forward,
        max_back_m=args.max_back,
        max_lateral_m=args.max_lateral,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Person follow tracking")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    parser.add_argument("--person-height", type=float, default=1.7)
    parser.add_argument("--desired-distance", type=float, default=0.9)
    parser.add_argument("--max-forward", type=float, default=0.7)
    parser.add_argument("--max-back", type=float, default=0.4)
    parser.add_argument("--max-lateral", type=float, default=0.4)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--ros", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.ros:
        rospy.init_node("person_follow", anonymous=True)
        target_pub = rospy.Publisher("/person_follow/target", FollowPose, queue_size=10)
    else:
        target_pub = None

    camera_mode = args.camera or not args.video
    video_source = args.camera_index if camera_mode else args.video
    video = Video(camera_mode, video_source, video_width=args.width, video_height=args.height)
    camera_model = build_camera_model(args)

    tracking_state = PersonTrackerState.Uninit
    ret = True
    while ret:
        ret, input_image = video.capOneFrame()
        if not ret:
            break

        tracking_target = None
        output = person_det.get_output(input_image)
        input_image = draw_person_bbox(input_image, output)
        output = person_track.get_output(input_image, output)

        if output:
            if tracking_state == PersonTrackerState.Uninit:
                track_ret = person_follow.init_track(input_image, person_items=output)
                if track_ret:
                    tracking_state = person_follow.get_state()
                    tracking_target = person_follow.get_tracking_person()
            else:
                tracking_state = person_follow.update_track(input_image, output)
                tracking_target = person_follow.get_tracking_person()

        if tracking_target is None:
            person_follow.mark_nodetected()
        else:
            bbox = tracking_target.get_box(PartName.body_part).tolist()
            forward, lateral = build_target_from_bbox(bbox, camera_model)
            if target_pub is not None:
                target_pub.publish(build_follow_pose(forward, lateral))
            input_image = draw_person_bbox(input_image, [tracking_target], draw_tracking=True, draw_conf=True)

        if args.show:
            cv2.imshow("follow", input_image)
            if cv2.waitKey(1) == 27:
                break

        if args.ros and rospy.is_shutdown():
            break

    if args.show:
        cv2.destroyAllWindows()
    time.sleep(0.1)


if __name__ == "__main__":
    main()
