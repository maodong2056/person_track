
#ifndef AI_ODOM_NAV_H
#define AI_ODOM_NAV_H
#include <vector>

#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923 /* pi/2 */
#define TO_RAD 0.017453292519943295
typedef struct
{
    double x;
    double y;
    double yaw; // 方向
    double v;   // 速度
    double ts;
} Odom;

typedef struct
{
    double x;
    double y;
    double yaw; // 方向
} Point2D;

typedef struct
{
    Odom odom;
    std::vector<Point2D> trajectory;
    double S;
    double now_s;
    Odom last_odom;
    bool s_arrive;
    bool a_arrive;
} Trajectory;

typedef struct
{
    double l_k;             // look forward gain
    double lfc;             // look-ahead distance
    double kp_v;            // speed propotional gain
    double kp_s;            // dist propotional gain
    double max_speed;       // [m/s]最大速度
    double wheel_l;         // [m] 左右轮距

    double k_rot_scale;     // w = k_rot_scale * dyaw / 0.5 PI
      
    double stop_s;          // [m] 终点停止距离
    double stop_dyaw;       // [deg] 旋转停止距离

    double back_dyaw;       // [deg] 后退角度，机器与目标点连线夹角超过back_dyaw，则后退
    double rot_back_dyaw;   // [deg] 触发后退时，如果规划角度（最近路径点）与机器角度夹角大于rot_dyaw，则旋转

    double max_over_s;      // [m] 最大路程= S(plan) + max_over_s, 超过就认为到达，进行旋转
    double min_v;           // [m/s] 最小线速度
    double min_w;           // [rad/s] 最小角速度
} ContrlParam;

Point2D pure_pursuit(Odom current_odom, Trajectory& plan, ContrlParam param, double dt, double *linear_left, double *linear_right);

#endif // AI_ODOM_NAV_H