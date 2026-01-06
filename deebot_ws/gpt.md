debot_odom_nav项目 
这是一个ros1项目，需要创建多个ros包

## 创建消息定义包4个

### 1). prediction 包有2个消息
   1. Pose.msg:
        ```
      Header header
      float32 x
      float32 y
      float32 theta
        ```
   2. PredictPose.msg:
        ```
      Pose predictPose
      Pose pose
        ```

      里程计数据发布 topic:  `prediction/PredictPose`

### 2).wheel 包有1个消息
1. SetWheelSpeed.msg
    
    ```
    uint8 SPEED_TYPE_DRIVING = 0 # type enum
    uint8 SPEED_TYPE_PHYSICAL = 1
    uint8 speed_type    # SPEED_TYPE
    float32[] speed 
    ```
    轮子速度控制 topic:  `wheel/SetWheelSpeed`

    speed[0]左轮 speed[1]右轮
    模式选择SPEED_TYPE_PHYSICAL，单位mm/s

###  3).  GdcImg 包有1个消息
1. GdcImg.msg:
   ```
    uint64 ts
    uint64 imgSize
    uint64 w
    uint64 h
    uint8[589824] img
    ```
    图像发布 topic:  `media/gdc_img`

    存储方式：
    ```
    #len_w 图片宽度
    #len_h 图片高度
    for(int h=0;h<len_h;h++)
    {
      for(int w=0;w<len_w;w++)
      {
        # 开始取(w,h)像素的rgb值
        B = img[(h*len_w + w)*3 + 0] 
        G = img[(h*len_w + w)*3 + 1] 
        R = img[(h*len_w + w)*3 + 2] 
      }
    }
    ```

###  4).  aiplan 包有1个消息

1. AiPlan.msg:
    ```
    # Ai ts时刻的路径规划结果
    float32 ts                              # 时间戳ts
    prediction/Pose ts_odom                 # ts时刻的里程计定位
    prediction/Pose2[] ts_trajectory               # 基于ts时刻图像的规划轨迹

    # 路径跟踪参数
    float32 l_k             # look forward gain
    float32 lfc             # look-ahead distance
    float32 kp_v            # speed propotional gain
    float32 kp_s            # distance propotional gain 
    float32 stop_s          # 停车距离
    float32 max_speed       # 最大速度    
    float32 wheel_l;        # 左右轮距
    ```
    ai发布规划轨迹 topic:  `/ai/plan`

## python实现键盘模拟Ai轨迹发布
  1. 创建ros1 python 包，代码封装在类中
  2. 订阅里程计数据，缓存在odom_list队列中，只缓存1000帧,
  3. 订阅图片数据，按照GdcImg.msg的定义读取图片缓存在img中，并使用opencv显示
  4. from pynput.keyboard import Listener, Key 监听键盘数据控制速度，w：加速，s:停止，x：减速，a:角速度加，d:角速度减
  5. 创建定时器，定时发生‘/ai/plan’，轨迹为时间间隔为0.2的v,w的5帧推测轨迹


## C++实现Ai轨迹跟踪控制

  已有算法头文件
  ```C
  #ifndef AI_ODOM_NAV_H
  #define AI_ODOM_NAV_H

  # define M_PI 3.14159265358979323846 
  # define M_PI_2		1.57079632679489661923	/* pi/2 */
  typedef struct {
      double x;
      double y;
      double yaw; // 方向
      double v; // 速度
  } Odom;

  typedef struct {
      double x;
      double y;
  } Point2D;

  typedef struct {
      Odom odom;
      Point2D trajectory[5];
  } Trajectory;

  typedef struct {
      double l_k ;            // look forward gain
      double lfc ;          // look-ahead distance
      double kp_v ;         // speed propotional gain
      double kp_s ;         // dist propotional gain
      double stop_s ;       // [m] 单车模型前后轮距离
      double max_speed;     // 最大速度
      double wheel_l;       // 左右轮距
  } ContrlParam;

  Point2D pure_pursuit(Odom current_odom, Odom ts_odom, Point2D ts_trajectory[5], ContrlParam param, double dt, double *linear_left, double *linear_right);

  #endif // AI_ODOM_NAV_H
  ```

  1. 创建ros1 C++ 包，代码封装在类中
  2. 订阅里程计数据，填充成员变量 `Odom current_odom`,速度填充为0
  3. ai规划数据，填充成员变量 `Odom ts_odom; Point2D ts_trajectory[5]; ContrlParam param`
  4. 创建定时器10hz循环调用pure_pursuit,输出左右轮速度v_l,v_r,发布在轮子速度控制 topic上
 
输出包创建命令
直接输出各个包的代码



tail -f /tmp/log/ros.log.00* | grep battery
sudo ifmetric wlp2s0f0 50


sudo mkdir -p /tmp/gazebo-nobody-rtshaderlibcache/
sudo chmod 777 /tmp/gazebo-nobody-rtshaderlibcache/

source /usr/share/gazebo-11/setup.bash 


X_TOOL=log /usr/bin/xspace_tool -o switch -n wheel.txt -p 'deeboot' -i 0 -s 
X_TOOL=log /usr/bin/xspace_tool -o level -n wheel.txt -p 'deeboot' -l 0
X_TOOL=log /usr/bin/xspace_tool -o info -n wheel.txt -p 'deeboot' -l 0
