import math
import time

import numpy as np
import rospy

from deebot_ros import Deebot, DeebotControlParam
from prediction.msg import Pose as FollowPose


class FollowController:
    def __init__(self):
        rospy.init_node("follow_control")
        self.deebot = Deebot(sim=False, keyboard_control=False, init_node=False)
        self.deebot.set_param(
            DeebotControlParam(
                l_k=0.05,
                lfc=0.4,
                kp_v=0.5,
                kp_s=1.5,
                max_speed=0.35,
                wheel_l=0.23,
                stop_s=0.02,
                stop_dyaw=8,
                back_dyaw=100,
                k_rot_scale=0.8,
                rot_back_dyaw=80,
                max_over_s=0.2,
                min_v=0.1,
                min_w=0.2,
            )
        )
        self.last_target = None
        self.last_target_time = None
        self.target_timeout = rospy.Duration(0.6)
        rospy.Subscriber("/person_follow/target", FollowPose, self.target_callback)
        rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def target_callback(self, msg: FollowPose):
        self.last_target = msg
        self.last_target_time = msg.header.stamp if msg.header.stamp else rospy.Time.now()
        self.publish_target(msg)

    def timer_callback(self, event):
        if self.last_target_time is None:
            return
        if rospy.Time.now() - self.last_target_time > self.target_timeout:
            self.publish_stop()

    def publish_target(self, msg: FollowPose):
        forward = msg.x
        lateral = msg.y
        yaw = math.atan2(lateral, max(forward, 1e-3))
        path = np.array(
            [
                [forward * 0.5, lateral * 0.5, yaw],
                [forward, lateral, yaw],
            ],
            dtype=float,
        )
        ts = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
        self.deebot.pub_path(path, ts)

    def publish_stop(self):
        self.last_target_time = None
        path = np.array([[0.0, 0.0, 0.0]], dtype=float)
        self.deebot.pub_path(path, time.time())


if __name__ == "__main__":
    FollowController()
    rospy.spin()
