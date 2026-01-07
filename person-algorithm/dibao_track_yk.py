import argparse
import time
from dataclasses import dataclass

import cv2
import rospy

from src import Video
from src.utils import PartName
from src.algorithm.api import PersonDetection, PersonMutiTrack
from src.algorithm.api import PersonFollow, PersonTrackerState
from src.algorithm.api.person_follow.person_item_track import PersonFollowItem
from src.utils import draw_person_bbox
from prediction.msg import Pose as FollowPose


@dataclass
class FollowConfig:
    min_ratio: float = 0.05
    max_ratio: float = 0.8


person_det = PersonDetection("user/settings/model/detection/body_detection/centernet_lite_8down_ratio_old.json")
person_track = PersonMutiTrack("user/settings/model/track/deep_sort.json")
person_follow = PersonFollow(
    "user/settings/model/recognition/body_recognition/pcb_reid_mobilev2.json",
    person_item=PersonFollowItem,
)


def build_target_from_bbox(bbox, image_shape, config: FollowConfig):
    x1, y1, x2, y2 = bbox
    image_h, image_w = image_shape[:2]
    cx = (x1 + x2) / 2.0
    bbox_h = max(y2 - y1, 1.0)

    ratio = bbox_h / max(image_h, 1.0)
    ratio = min(max(ratio, config.min_ratio), config.max_ratio)

    offset = (cx - image_w / 2.0) / (image_w / 2.0)
    offset = min(max(offset, -1.0), 1.0)
    return float(ratio), float(offset)


def build_follow_pose(forward, lateral):
    pose = FollowPose()
    pose.header.stamp = rospy.Time.now()
    pose.x = forward
    pose.y = lateral
    pose.theta = 0.0
    return pose


def parse_args():
    parser = argparse.ArgumentParser(description="Person follow tracking")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--ros", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = FollowConfig()

    if args.ros:
        rospy.init_node("person_follow", anonymous=True)
        target_pub = rospy.Publisher("/person_follow/target", FollowPose, queue_size=10)
    else:
        target_pub = None

    camera_mode = args.camera or not args.video
    video_source = args.camera_index if camera_mode else args.video
    video = Video(camera_mode, video_source, video_width=args.width, video_height=args.height)

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
            forward, lateral = build_target_from_bbox(bbox, input_image.shape, config)
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
