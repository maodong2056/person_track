import rospy

from prediction.msg import Pose as FollowPose
from wheel.msg import SetWheelSpeed


class FollowController:
    def __init__(self):
        rospy.init_node("follow_control")
        self.wheel_pub = rospy.Publisher("wheel/SetWheelSpeed", SetWheelSpeed, queue_size=10)
        self.wheel_l = rospy.get_param("~wheel_l", 0.23)
        self.forward_speed = rospy.get_param("~forward_speed", 0.18)
        self.turn_gain = rospy.get_param("~turn_gain", 0.8)
        self.turn_deadband = rospy.get_param("~turn_deadband", 0.08)
        self.max_turn_speed = rospy.get_param("~max_turn_speed", 0.6)
        self.search_turn_speed = rospy.get_param("~search_turn_speed", 0.5)
        self.desired_ratio = rospy.get_param("~desired_ratio", 0.35)
        self.min_ratio = rospy.get_param("~min_ratio", 0.05)
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
            self.publish_search()

    def publish_target(self, msg: FollowPose):
        ratio = msg.x
        offset = msg.y
        if ratio < self.min_ratio:
            self.publish_search()
            return
        if abs(offset) < self.turn_deadband:
            offset = 0.0

        target_forward = 0.0 if ratio >= self.desired_ratio else self.forward_speed
        target_turn = self.turn_gain * offset
        target_turn = max(-self.max_turn_speed, min(self.max_turn_speed, target_turn))
        self.publish_wheel_speed(target_forward, target_turn)

    def publish_search(self):
        self.last_target_time = None
        self.publish_wheel_speed(0.0, self.search_turn_speed)

    def publish_wheel_speed(self, linear_v, angular_w):
        left = linear_v - angular_w * self.wheel_l * 0.5
        right = linear_v + angular_w * self.wheel_l * 0.5
        msg = SetWheelSpeed()
        msg.speed_type = SetWheelSpeed.SPEED_TYPE_PHYSICAL
        msg.speed = [left * 1000.0, right * 1000.0]
        self.wheel_pub.publish(msg)


if __name__ == "__main__":
    FollowController()
    rospy.spin()
