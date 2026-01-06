#include <ros/ros.h>
#include <wheel/SetWheelSpeed.h>
#include <prediction/Pose.h>
#include <prediction/PredictPose.h>
#include <aiplan/AiPlan.h>
#include <cmath>
#include "aiplan/ai_odom_nav.h"

class AIControl
{
public:
    AIControl(double fps)
    {
        dt = 1.0/fps;
        odom_sub = nh.subscribe("/prediction/PredictPose", 1000, &AIControl::odomCallback, this);
        ai_plan_sub = nh.subscribe("/ai/plan", 1000, &AIControl::aiPlanCallback, this);
        wheel_pub = nh.advertise<wheel::SetWheelSpeed>("wheel/SetWheelSpeed", 1000);
        head_pub = nh.advertise<prediction::Pose>("ai_control/head", 1000);
        timer = nh.createTimer(ros::Duration(dt), &AIControl::timerCallback, this);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber odom_sub;
    ros::Subscriber ai_plan_sub;
    ros::Publisher wheel_pub;
    ros::Publisher head_pub;
    ros::Timer timer;
    double dt;
    bool get_plan{false};

    Odom current_odom;
    Odom ts_odom;
    Trajectory plan;
    ContrlParam param;
    double output_v{0};

    void odomCallback(const prediction::PredictPose::ConstPtr &msg)
    {
        current_odom.x = msg->predictPose.x/1000.;
        current_odom.y = msg->predictPose.y/1000.;
        current_odom.yaw = msg->predictPose.theta;
        current_odom.v = output_v; // 暂时使用输出的速度
        current_odom.ts = msg->predictPose.header.stamp.toSec();
    }

    void aiPlanCallback(const aiplan::AiPlan::ConstPtr &msg)
    {
        get_plan=true;
        ts_odom.x = msg->ts_odom.x/1000.0;
        ts_odom.y = msg->ts_odom.y/1000.0;
        ts_odom.yaw = msg->ts_odom.theta;
        ts_odom.v = 0; // 忽略，没有使用
        ts_odom.ts = msg->ts_odom.header.stamp.toSec();

        plan.odom = ts_odom;
        plan.S = 0;
        plan.now_s = 0;
        plan.s_arrive = false;
        plan.a_arrive = false;
        plan.last_odom = current_odom;
        plan.trajectory.resize(msg->ts_trajectory.size());
        for (int i = 0; i < msg->ts_trajectory.size(); ++i)
        {
            plan.trajectory[i].x = msg->ts_trajectory[i].x;
            plan.trajectory[i].y = msg->ts_trajectory[i].y;
            plan.trajectory[i].yaw = msg->ts_trajectory[i].theta;

            if(i>0){
                double dx = plan.trajectory[i].x - plan.trajectory[i - 1].x;
                double dy = plan.trajectory[i].y - plan.trajectory[i - 1].y;
                plan.S += sqrt(dx * dx + dy * dy);
            }
        }

        param.l_k = msg->l_k;
        param.lfc = msg->lfc;
        param.kp_v = msg->kp_v;
        param.kp_s = msg->kp_s;
        param.max_speed = msg->max_speed;
        param.wheel_l = msg->wheel_l;
        param.stop_s = msg->stop_s;
        param.stop_dyaw = msg->stop_dyaw;
        param.back_dyaw = msg->back_dyaw;
        param.k_rot_scale = msg->k_rot_scale;
        param.rot_back_dyaw = msg->rot_back_dyaw;
        param.max_over_s = msg->max_over_s;
        param.min_v = msg->min_v;
        param.min_w = msg->min_w;
        ROS_INFO("[ros] get plan,ts:%f\n",ts_odom.ts);
    }

    void timerCallback(const ros::TimerEvent &)
    {
        if(!get_plan || plan.a_arrive) return;
        double dx = current_odom.x - plan.last_odom.x;
        double dy = current_odom.y - plan.last_odom.y;
        plan.now_s += sqrt(dx * dx + dy * dy); 
        plan.last_odom = current_odom;
        double linear_left, linear_right;
        Point2D head_point = pure_pursuit(current_odom, plan, param, dt, &linear_left, &linear_right);

        wheel::SetWheelSpeed speed_msg;
        speed_msg.speed_type = speed_msg.SPEED_TYPE_PHYSICAL;
        speed_msg.speed.push_back(linear_left*1000);
        speed_msg.speed.push_back(linear_right*1000);
        wheel_pub.publish(speed_msg);
        output_v = (linear_left + linear_right) / 2;

        prediction::Pose head;
        head.x = head_point.x;
        head.y = head_point.y;
        head.theta = 0;
        head.header.frame_id="base_footprint";
        head.header.stamp= ros::Time(ts_odom.ts);
        head_pub.publish(head);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ai_control");
    double fps = 50;
    AIControl ai_control(fps);
    ros::spin();
    return 0;
}
