#!/usr/bin/env python
import rospy
import cv2
from std_msgs.msg import Header
from aiplan.msg import AiPlan
from prediction.msg import Pose as DeebotPose
from prediction.msg import PredictPose
from geometry_msgs.msg import Twist
from wheel.msg import SetWheelSpeed
from onOffInfo.msg import OnOffInfo
from rangeDet.msg import RangeDetect

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped,Quaternion
from geometry_msgs.msg import TransformStamped
import tf

from GdcImg.msg import GdcImg
from pynput.keyboard import Listener, Key
import numpy as np 
import math
import pyautogui
import copy

WHEEL_BASE = 0.23

class DeebotSim():
    def __init__(self) -> None:
        # 订阅 /odom 话题
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        # 发布 /prediction/PredictPose 话题
        self.predict_pose_pub = rospy.Publisher('/prediction/PredictPose', PredictPose, queue_size=10)
        
        # 订阅 /wheel/SetWheelSpeed 话题
        rospy.Subscriber("/wheel/SetWheelSpeed", SetWheelSpeed, self.set_wheel_speed_callback)
        # 发布 /cmd_vel 话题
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    def set_wheel_speed_callback(self,data):
        if data.speed_type == SetWheelSpeed.SPEED_TYPE_PHYSICAL:
            # 计算线速度和角速度
            v_left = data.speed[0] / 1000.0  # 将速度从 mm/s 转换为 m/s
            v_right = data.speed[1] / 1000.0  # 将速度从 mm/s 转换为 m/s

            linear_speed = (v_left + v_right) / 2.0
            angular_speed = (v_right - v_left) / WHEEL_BASE

            # 创建Twist消息
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = linear_speed
            cmd_vel_msg.angular.z = angular_speed

            # 发布cmd_vel消息
            self.cmd_vel_pub.publish(cmd_vel_msg)

    def odom_callback(self, data):
        # 从Odometry消息中提取位置信息
        pose = DeebotPose()
        pose.header = data.header
        pose.x = data.pose.pose.position.x * 1000
        pose.y = data.pose.pose.position.y * 1000
        pose.theta = 2 * math.atan2(data.pose.pose.orientation.z, data.pose.pose.orientation.w)
        
        # 创建PredictPose消息
        predict_pose_msg = PredictPose()
        predict_pose_msg.predictPose = pose
        predict_pose_msg.pose = pose  # 这里简单示例使用相同的pose，你可以根据需求更改
        
        # 发布PredictPose消息
        self.predict_pose_pub.publish(predict_pose_msg)

class KeboardSpeed:
    def __init__(self) -> None:
        self.speed = 0.0
        self.angular_speed = 0.0
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()
    def on_press(self, key):
        da = 0.03
        dw = 0.1
        change = True
        if key == Key.up:
            self.speed += da
        elif key == Key.down:
            self.speed -= da
        elif key == Key.left:
            self.angular_speed += dw
        elif key == Key.right:
            self.angular_speed -= dw
        else:
            change=False
        if change:rospy.loginfo(f"v:{self.speed},w:{self.angular_speed}")

class DeebotVis:
    def __init__(self,frame_id):
        self.frame_id = frame_id
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.target_pub = rospy.Publisher('/ai_control/target', PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher('/ts_trajectory_path', Path, queue_size=10)
        # 订阅 /prediction/PredictPose 话题
        self.odom_sub = rospy.Subscriber("/prediction/PredictPose", PredictPose, self.predict_pose_callback)
        self.control_target_sub = rospy.Subscriber('ai_control/head', DeebotPose, self.head_callback)

    def head_callback(self, msg):
        self.target_pub.publish(self.deebotpose_to_posestamped(msg))
    
    def predict_pose_callback(self, data):
        # 从PredictPose消息中提取位姿
        predict_pose:DeebotPose = data.predictPose

        # 获取当前时间
        current_time = data.predictPose.header.stamp
        #rospy.loginfo(predict_pose.x,predict_pose.y,predict_pose.theta,current_time)

        # 平移
        translation = (
            predict_pose.x/1000,
            predict_pose.y/1000,
            0.0
        )

        # 旋转
        quat = tf.transformations.quaternion_from_euler(0, 0, predict_pose.theta)
        rotation = (
            quat[0],
            quat[1],
            quat[2],
            quat[3]
        )

        # 发布TF变换
        self.tf_broadcaster.sendTransform(
            translation,
            rotation,
            current_time,
            "base_footprint",
            "odom"
        )

    def plan_vis(self, plan_msg:AiPlan):
        path = Path()
        path.header.stamp = plan_msg.ts_odom.header.stamp
        path.header.frame_id = self.frame_id

        for pose in plan_msg.ts_trajectory:
            path.poses.append(self.deebotpose_to_posestamped(pose))

        self.path_pub.publish(path)

    def deebotpose_to_posestamped(self, pose:DeebotPose):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = pose.header.stamp
        pose_stamped.header.frame_id = self.frame_id
        pose_stamped.pose.position.x = pose.x
        pose_stamped.pose.position.y = pose.y
        pose_stamped.pose.position.z = 0
        # 假设 pose.theta 代表绕z轴的旋转
        pose_stamped.pose.orientation = self.yaw_to_quaternion(pose.theta)
        return pose_stamped

    def yaw_to_quaternion(self, yaw):
        from tf.transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, yaw)
        return Quaternion(*q)

class DeebotControlParam:
    def __init__(self,
                 l_k=0.05,                # 目标点距离速度增益
                 lfc=0.4,                 # 目标点基础距离
                 kp_v=0.5,                # 速度比例增益
                 kp_s=1.5,                # 距离比例增益
                 max_speed=0.3,           # [m/s] 最大速度
                 wheel_l=0.23,            # [m] 左右轮之间的距离
                 stop_s=0.01,             # [m] 到达终点时的停止距离
                 stop_dyaw=5,             # [deg] 旋转停止角度
                 back_dyaw=100,           # [deg] 后退角度，当机器人与目标点连线夹角超过此值时将后退
                 k_rot_scale=0.9,         # w = k_rot_scale * dyaw / 0.5 PI
                 rot_back_dyaw=80,        # [deg] 如果规划角度（最近路径点）与机器人的角度夹角大于此值，则进行旋转
                 max_over_s=0.2,          # [m] 最大额外行驶距离，即计划行驶距离加上此值，若超过则认为已到达并进行旋转
                 min_v=0.1,               # [m/s] 最小线速度
                 min_w=0.2):              # [rad/s] 最小角速度
        self.l_k = l_k
        self.lfc = lfc
        self.kp_v = kp_v
        self.kp_s = kp_s
        self.max_speed = max_speed
        self.wheel_l = wheel_l
        self.stop_s = stop_s
        self.stop_dyaw = stop_dyaw
        self.back_dyaw = back_dyaw
        self.k_rot_scale = k_rot_scale
        self.rot_back_dyaw = rot_back_dyaw
        self.max_over_s = max_over_s
        self.min_v = min_v
        self.min_w = min_w

class Deebot:
    def __init__(self,sim=False,keyboard_control = False ,init_node=False):
        if init_node:
            rospy.init_node('ai_plan')
        self.frame_id = 'base_footprint'

        if sim: self.deebot_sim = DeebotSim()
        self.vis = DeebotVis(self.frame_id)

        self.odom_list = []
        self.img_msg = None
        self.img = None
        self.bump_states = [None] * 4  # 撞板状态缓存
        self.edge_states = {}  # 沿边数据缓存，键是type，值是对应的values数组
        self.sub_odom = rospy.Subscriber('/prediction/PredictPose', PredictPose, self.deebotodom_callback) 
        self.sub_img = rospy.Subscriber('media/gdc_img', GdcImg, self.img_callback)
 
        self.sub_bump = rospy.Subscriber('onOffInfo/OnOffInfo', OnOffInfo, self.bump_callback)  # 订阅撞板信息
        self.sub_edge = rospy.Subscriber('rangeDet/RangeDetect', RangeDetect, self.edge_callback)  # 订阅沿边信息
        self.pub = rospy.Publisher('/ai/plan', AiPlan, queue_size=10)
        if keyboard_control:
            self.keyboard_speed = KeboardSpeed()
            self.timer = rospy.Timer(rospy.Duration(10), self.timer_callback)

        self.param = DeebotControlParam()

    def set_param(self,param:DeebotControlParam):
        self.param = param

    def __delattr__(self):
        cv2.destroyAllWindows()

    def deebotodom_callback(self, msg):
        self.odom_list.append(msg)
        #rospy.loginfo(msg)
        if len(self.odom_list) > 1000:
            self.odom_list.pop(0)

    def img_callback(self, msg:GdcImg):
        self.img_msg = msg
        self.img = np.frombuffer(msg.img, dtype=np.uint8).reshape((msg.h, msg.w, int(msg.imgSize/(msg.h*msg.w))))
        #rospy.loginfo(msg.h, msg.w,)
        # 将图像从BGR格式转换为RGB格式
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        if 1: 
            cv2.imshow('GdcImg', self.img)
            cv2.waitKey(1)
        if 1: return
        # 截取屏幕
        screenshot = pyautogui.screenshot()

        # 将截屏转换为numpy数组
        screenshot = cv2.cvtColor(np.array(screenshot) , cv2.COLOR_BGR2RGB)
        # 确保截屏和self.img大小一致，否则需要调整大小
        if screenshot.shape != self.img.shape:
            # 调整截屏大小与self.img一致
            screenshot = cv2.resize(screenshot, (self.img.shape[1], self.img.shape[0]))

        # 拼接图像
        concatenated_img = np.concatenate((self.img, screenshot), axis=1)

        # 显示拼接后的图像
        cv2.imshow('Concatenated Image', concatenated_img)
        cv2.waitKey(1)
    
    def bump_callback(self, msg: OnOffInfo):
        for sensor_value in msg.values:
            if sensor_value.type == sensor_value.TYPE_BUMP:
                state = sensor_value.value
                for i in range(4):
                    state_val = 1 if state & (0x01 << i) else 0
                    if self.bump_states[i] != state_val:
                        self.bump_states[i] = state_val
                        # 这里可以添加处理撞板状态变化的逻辑
                        rospy.loginfo(f"[bump_states change] {self.bump_states}")
    def edge_callback(self, msg: RangeDetect):
        for v in msg.rangeDets:
            self.edge_states[v.type.type] = v.values
            # 这里可以添加处理沿边数据变化的逻辑

    def get_bump_states(self):
        return self.bump_states

    def get_edge_states(self):
        return self.edge_states
    
    def get_odom(self, ts):
        # 计算时间差并找到最小值的索引
        time_diff = [abs(odom.predictPose.header.stamp.to_sec() - ts) for odom in self.odom_list]
        min_index = time_diff.index(min(time_diff))
        
        # 返回时间差最小的odom对象
        return self.odom_list[min_index]
    
    def get_img(self):
        if self.img is None:
            return None ,None
        return self.img,self.img_msg.ts
    
    def createAiPlan(self,ts):
        ts_odom:PredictPose = self.get_odom(ts)
        plan_msg = AiPlan()
        plan_msg.ts = ts
        plan_msg.ts_odom = ts_odom.predictPose
        plan_msg.l_k = self.param.l_k
        plan_msg.lfc = self.param.lfc
        plan_msg.kp_v = self.param.kp_v
        plan_msg.kp_s = self.param.kp_s
        plan_msg.max_speed = self.param.max_speed
        plan_msg.wheel_l = self.param.wheel_l
        plan_msg.stop_s = self.param.stop_s
        plan_msg.stop_dyaw = self.param.stop_dyaw
        plan_msg.back_dyaw = self.param.back_dyaw
        plan_msg.rot_back_dyaw = self.param.rot_back_dyaw
        plan_msg.max_over_s = self.param.max_over_s
        plan_msg.min_v = self.param.min_v
        plan_msg.min_w = self.param.min_w
        return plan_msg
    
    def pub_path(self,path,ts):
        if self.img is None:
            rospy.loginfo("[pub_path] wait for img")
            return 
        plan_msg:AiPlan = self.createAiPlan(ts)
        for point in path:
            pose = DeebotPose() 
            # 更新位置和朝向，基于前一个点
            pose.x = point[0]
            pose.y = point[1]
            pose.theta = point[2]
            pose.header.stamp = plan_msg.ts_odom.header.stamp
            
            # 将计算出的点加入轨迹中
            plan_msg.ts_trajectory.append(pose)
        self.pub.publish(plan_msg)
        self.vis.plan_vis(plan_msg)


    def timer_callback(self, event):
        if self.img_msg is None:
            rospy.loginfo("[timer_callback] wait for img")
            ts = rospy.Time.now().to_sec() - 0.1
            #return
        else:
            self.img_msg:GdcImg
            ts = self.img_msg.ts
        if len(self.odom_list) > 0: 
            plan_msg = self.createAiPlan(ts)
            # 初始点为当前的odom点
            pose = DeebotPose(x=0,y=0,theta=0)
            prev_pose = copy.deepcopy(pose) 
            back = False
            if not back and  self.keyboard_speed.speed < 0:
                pose.theta += math.pi
            plan_msg.ts_trajectory.append(pose)
            
            for i in range(5):
                pose = DeebotPose()
                dt = 0.2  # 时间间隔
                
                # 更新位置和朝向，基于前一个点
                pose.x = prev_pose.x + self.keyboard_speed.speed * dt * math.cos(prev_pose.theta)
                pose.y = prev_pose.y + self.keyboard_speed.speed * dt * math.sin(prev_pose.theta)
                pose.theta = prev_pose.theta + self.keyboard_speed.angular_speed * dt
                # 更新prev_pose为当前计算出的点
                prev_pose = copy.deepcopy(pose) 
                if not back and   self.keyboard_speed.speed < 0:
                    pose.theta += math.pi

                # 将计算出的点加入轨迹中
                plan_msg.ts_trajectory.append(pose)

            
            self.pub.publish(plan_msg)
            self.vis.plan_vis(plan_msg)


if __name__ == '__main__': 
    Deebot(sim=False,keyboard_control=True,init_node=True)
    rospy.spin()
    cv2.destroyAllWindows()
