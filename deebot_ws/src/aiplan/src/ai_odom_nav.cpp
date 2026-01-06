#include "aiplan/ai_odom_nav.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <ros/ros.h>

// double k = {0.1};     // look forward gain
// double Lfc = 1.0;     // look-ahead distance
// double Kp_v = 1.0;    // speed propotional gain
// double Kp_s = 0.5;    // dist propotional gain
// double stop_s = 0.05; // [m] 单车模型前后轮距离

double normalize_angle(double angle)
{
    const double result = fmod(angle + M_PI, 2.0 * M_PI);
    if (result <= 0.0)
        return result + M_PI;
    return result - M_PI;
}
double PIDControl(double Kp, double target, double current)
{
    double v = Kp * (target - current);
    return v;
}
double clip(double v, double min, double max)
{
    if (v > max)
        return max;
    if (v < min)
        return min;
    return v;
}
double setMin(double& target,double min,double scale=1){
    if(target==0.0) return target;
    if (target > 0 && target < min)
        target = min*scale;
    if (target < 0 && target > -min)
        target = -min*scale;
}
Point2D getPointBaseOdom(Point2D p, Odom state){
    Point2D prime;
    double x_trans = state.x - p.x;
    double y_trans = state.y - p.y;

    prime.x = x_trans * cos(state.yaw) + y_trans * sin(state.yaw);
    prime.y = -x_trans * sin(state.yaw) + y_trans * cos(state.yaw);
    return prime;
}

Point2D pure_pursuit(Odom current_odom, Trajectory& plan, ContrlParam param, double dt, double *linear_left, double *linear_right)
{
    static double last_w = 0.0;
    static bool in_roll = false;
    static bool in_roll_back = false;
    if (plan.trajectory.empty())
        return Point2D({0, 0});
    // 将当前odom转换到ts时刻的odom
    double dx = current_odom.x - plan.odom.x;
    double dy = current_odom.y - plan.odom.y;
    double dyaw = current_odom.yaw - plan.odom.yaw;

    double cos_yaw = cos(-plan.odom.yaw);
    double sin_yaw = sin(-plan.odom.yaw);

    double x_robot = cos_yaw * dx - sin_yaw * dy;
    double y_robot = sin_yaw * dx + cos_yaw * dy;
    Odom state = {x_robot, y_robot, dyaw, current_odom.v};

    Point2D goal = plan.trajectory.back();
    Point2D goal_base_state = getPointBaseOdom(goal,state);
    // double goal_dist = sqrt(pow(goal.y - state.y, 2) + pow(goal.x - state.x, 2));
    double goal_dist = sqrt(pow(goal_base_state.x, 2) + pow(goal_base_state.y / 4.0, 2));
    bool over_max_s = plan.now_s > fmin(plan.S + param.max_over_s,plan.S * 1.2) ;
    ROS_INFO("[goal checker] goal_diff:%f-%f,scale_goal_dist:%f,plan.S:%f,plan.now_s:%f,over_max_s:%d\n",goal_base_state.x,goal_base_state.y,goal_dist,plan.S,plan.now_s,over_max_s);
    if (over_max_s || goal_dist <  param.stop_s * (plan.s_arrive ? 2 : 1))
    {
        plan.s_arrive = true;
        double diff_yaw = normalize_angle(goal.yaw - state.yaw);

        if (abs(diff_yaw) < param.stop_dyaw * TO_RAD * (plan.a_arrive ? 2 : 1))// angle_arrive 则放宽到达条件
        {
            plan.s_arrive = true;
            ROS_INFO("[goal checker] angle arrive and stop rot\n");
            in_roll = false;
            last_w = 0;

            *linear_left = 0;
            *linear_right = 0;
        }
        else
        { // 到达原地旋转
            plan.s_arrive = false;
            in_roll = true;
            double target_v = 0, target_w = 0;
            double keep_yaw = (30 * TO_RAD);
            if (last_w < 0)
                keep_yaw = -keep_yaw;
            if (last_w == 0)
                keep_yaw = 0;
            if (diff_yaw + keep_yaw > 0)
                target_w = 0.5 * abs(diff_yaw) / M_PI;
            else
                target_w = -0.5 * abs(diff_yaw) / M_PI;
            setMin(target_w,param.min_w);
            last_w = target_w;
            ROS_INFO("[goal checker][goal rot] pose arrive and rot and dyaw:%f, flag in_roll: %d, rot_w: %f \n",diff_yaw, in_roll,target_w);

            *linear_left = target_v - target_w * param.wheel_l * 0.5;
            *linear_right = target_v + target_w * param.wheel_l * 0.5;
        }
        //     target_speed = 0;
        //     target_w = 0.5;
        //     if(norm_alpha<0)
        //         target_w = -target_w;

        return goal;
    }
    plan.s_arrive = false;

    // 寻找前视点
    int min_idx = 0;
    double min_dist = 1e20;
    for (size_t i = 0; i < plan.trajectory.size(); ++i)
    {
        double dx = plan.trajectory[i].x - state.x;
        double dy = plan.trajectory[i].y - state.y;
        double distance = sqrt(dx * dx + dy * dy);
        if (distance <= min_dist)
        {
            min_dist = distance;
            min_idx = i;
        }
    }
    Point2D near = plan.trajectory[min_idx];

    double ld = 0.0001;
    double Lf = param.l_k * state.v + param.lfc;
    int target_index = min_idx;
    while (Lf > ld && (target_index + 1) < plan.trajectory.size())
    {
        double dx = plan.trajectory[target_index + 1].x - plan.trajectory[target_index].x;
        double dy = plan.trajectory[target_index + 1].y - plan.trajectory[target_index].y;
        ld += sqrt(dx * dx + dy * dy);
        target_index += 1;
    }
    if (target_index > 9)
        target_index = 9;
    Point2D target = plan.trajectory[target_index];
    ld = sqrt(pow(target.y - state.y, 2) + pow(target.x - state.x, 2));

    double alpha_raw = atan2(target.y - state.y, target.x - state.x) - state.yaw;

    double alpha = alpha_raw;
    if (state.v < 0) // back
        alpha = atan2(target.y - state.y, target.x - state.x) - (state.yaw + M_PI);

    // 单车模型
    // double delta = atan2(2.0 * L * sin(alpha) / /*ld*/ Lf, 1.0);
    // double target_w = state.v / L * tan(delta) ;
    ld = fmax(ld,0.08);
    // 差速模型
    double R = ld / (2.0 * sin(alpha));
    double target_w = fabs(state.v) / R;

    double p_speed = PIDControl(param.kp_s, goal_dist, 0);
    double target_speed = clip(p_speed, -param.max_speed, param.max_speed);

    double norm_alpha_raw = normalize_angle(alpha_raw);
    if (in_roll_back || fabs(norm_alpha_raw) > param.back_dyaw * TO_RAD) // 目标点在后方
    {
        target_speed = -target_speed; // 后退
        double near_diff_yaw = normalize_angle(near.yaw - state.yaw);
        if (abs(near_diff_yaw) > (in_roll_back ?  param.stop_dyaw :  param.rot_back_dyaw) * TO_RAD // 与路径指导方向小于60，继续后退
                                                                   // && goal_dist > param.stop_s * 3
            )                                                      // 原地旋转
        {
            in_roll_back = true;
            target_speed = 0;
            double keep_yaw = (30 * TO_RAD);
            if (last_w < 0)
                keep_yaw = -keep_yaw;
            if (last_w == 0)
                keep_yaw = 0;
            if (norm_alpha_raw + keep_yaw > 0)
                target_w = 0.5 * abs(norm_alpha_raw) / M_PI_2;
            else
                target_w = -0.5 * abs(norm_alpha_raw) / M_PI_2;
            setMin(target_w,param.min_w);
            ROS_INFO("[back checker][in rot back] back dyaw:%f, flag in_roll_back: %d, rot_w:%f, path dyaw:%f\n",near_diff_yaw, in_roll_back,target_w,near_diff_yaw);
        }
        else
        { 
            ROS_INFO("[back checker][in back] back dyaw:%f, flag in_roll_back: %d\n",near_diff_yaw, in_roll_back);
            in_roll_back = false;
        }
    }
    else
    {

        in_roll_back = false;
    }

    double a = PIDControl(param.kp_v, target_speed, state.v);
    double target_dv = a * dt;
    // setMin(target_dv,param.min_v,1.1);
    double target_v = state.v + target_dv;

    // 输出计算结果
    ROS_INFO("--- [pure_pursuit]\nmin_idx:%d,target_index:%d,flag in_roll_back: %d\nx:%f, y:%f, yaw:%f, v:%f\nLf:%f,ld:%f, alpha:%f, R:%f\ntarget_speed:%f,a:%f,target_dv:%f, target_v:%f, target_w:%f\n state.yaw:%f - near.yaw:%f---\n",
           min_idx, target_index, in_roll_back, state.x, state.y, state.yaw, state.v, Lf, ld, alpha_raw, R,target_speed, a, target_dv,target_v, target_w, state.yaw, near.yaw);

    if(in_roll_back && fabs(target_v)<=param.min_v) target_v=0;
    setMin(target_v,param.min_v);
    setMin(target_w,param.min_w);
    last_w = target_w;

    *linear_left = target_v - target_w * param.wheel_l * 0.5;
    *linear_right = target_v + target_w * param.wheel_l * 0.5;
    return target;
}
