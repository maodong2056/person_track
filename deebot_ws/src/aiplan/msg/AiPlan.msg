float32 ts                              # 时间戳ts
prediction/Pose ts_odom                 # ts时刻的里程计定位
prediction/Pose[] ts_trajectory         # 基于ts时刻图像的规划轨迹

# 路径跟踪参数
float32 l_k             # look forward gain
float32 lfc             # look-ahead distance
float32 kp_v            # speed propotional gain
float32 kp_s            # distance propotional gain 
float32 max_speed       # [m/s]最大速度
float32 wheel_l         # [m] 左右轮距
      
float32 k_rot_scale     # w = k_rot_scale * dyaw / 0.5 PI
float32 stop_s          # [m] 终点停止距离
float32 stop_dyaw       # [deg] 旋转停止距离

float32 back_dyaw       # [deg] 后退角度，机器与目标点连线夹角超过back_dyaw，则后退
float32 rot_back_dyaw   # [deg] 触发后退时，如果规划角度（最近路径点）与机器角度夹角大于rot_dyaw，则旋转

float32 max_over_s      # [m] 最大路程= S(plan) + max_over_s, 超过就认为到达，进行旋转
float32 min_v           # [m/s] 最小线速度
float32 min_w           # [rad/s] 最小角速度