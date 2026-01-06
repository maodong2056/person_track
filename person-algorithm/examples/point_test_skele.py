# @Time : 2021/3/17 14:57
# @Author : Altair.Huazj
# @File : demo.py
# @Software: PyCharm
import json
import cv2
import os
import datetime
import numpy as np
import math
from src import DeepDetModel, Video
from src import DeepKPModel, TrackModule
from src import TrackFilter, LandmarkFilter
from src import Deep3DKPModel, Deep3DKPLiteModel, Deep3DKPSkeleModel
from src.utils import draw_hkp, draw_hkp_simple, get_3d_depth_bybox, vis_keypoints, vis_3d_multiple_skeleton_conf_toimage
from src.utils.draw_3dkps import vis_3d_multiple_skeleton_conf_addpoint_toimage
import matplotlib.pyplot as plt
from src.utils.draw_3dkps import fig2data
from src.utils.pose_utils import pixel2cam, cam2pixel
from src.utils.point_utils import disAndAngle, tanAngle
colours = np.random.rand(32, 3)*255
L = 1500
# load config file##########################################################################
with open("../user/lib/setting_trail.json") as f:
    config = json.load(f)
# gpu setting
gpu_setting = config["gpu_setting"]
############################################################################################

# det&track algorithm#######################################################################
my_trackModule = TrackModule(**config["deepModel"]["deepDetection"]["structure"],
                             **gpu_setting,
                             **{"single_mode": True})
det_setting = config["deepModel"]["deepDetection"]["parameters"]
det_load_ret = my_trackModule.load_model(config["deepModel"]["deepDetection"]["model"])
############################################################################################

# 3Dkeypoint algorithm########################################################################
my_3dkp_model = Deep3DKPModel(**config["deepModel"]["deep3DKeypoint"]["structure"],
                    **gpu_setting)
tdkp_load_ret = my_3dkp_model.load_model(config["deepModel"]["deep3DKeypoint"]["model"])

my_3dkp_skele_model = Deep3DKPSkeleModel(**config["deepModel"]["deep3DKeypointSkeleton"]["structure"],
                    **gpu_setting)
tdkp_skele_load_ret = my_3dkp_skele_model.load_model(config["deepModel"]["deep3DKeypointSkeleton"]["model"])
############################################################################################

# filter trace##############################################################################
my_trace = TrackFilter(diff_thres=7.5)
############################################################################################
data_root = "data_point2wash_everywhere"
tag = "point_everywhere"
# load images###############################################################################
root_dir = os.path.join(r"D:\project\arch\realsense_data", data_root)
txt_list = os.path.join(root_dir, "list.txt")
with open(txt_list, "r") as f:
    images_ids = f.readlines()
i = datetime.datetime.now()
video_name = "../user/output/3dpose_{}_{}_{}_{}_{}.avi".format(data_root, tag, i.year, i.month, i.day, i.hour)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15
size = (2144, 720)
raw_video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
############################################################################################

#save dir###################################################################################
save_dir = os.path.join(r"D:\project\arch\realsense_res", "{}_{}_{}_{}_{}_{}".
                        format(i.year, i.month, i.day, i.hour, data_root, tag))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
############################################################################################


# load model check##########################################################################
if (not det_load_ret) or (not tdkp_load_ret):
    print("Load Model Error!")
    exit()
############################################################################################
ret = True
kps = []
depth_scale = 0.0010000000474974513
point_pos = np.array([[329, 454, 353, 531]])
# skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
skeleton = ((0,7),(7, 8), (8, 9), (8, 10), (10, 11), (11, 12), (8, 13), (13, 14), (14, 15), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
def vis_3d_multiple_skeleton_conf_whole_toimage(kpt_3d, point_p, point_p_pred,
                                                radius, direction, pointdis,
                                                conf_thres=0.2, filename=None):
    fig = plt.figure(figsize=(12, 10), dpi=72)
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    ax2.plot(radius, c=colors[0])
    ax3.plot(direction, c=colors[5])
    ax4.plot(pointdis, c=colors[13])
    # ax1.scatter(point_p[0, 0], point_p[0, 2], -point_p[0, 1], c=np.array([0., 0., 1.]), marker='o')
    ax1.scatter(point_p_pred[0, 0], point_p_pred[0, 2], -point_p_pred[0, 1], c=np.array([1., 0., 1.]), marker='x')
    for l in range(len(skeleton)):
        i1 = skeleton[l][0]
        i2 = skeleton[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n, i1, 0], kpt_3d[n, i2, 0]])
            y = np.array([kpt_3d[n, i1, 1], kpt_3d[n, i2, 1]])
            z = np.array([kpt_3d[n, i1, 2], kpt_3d[n, i2, 2]])

            if kpt_3d[n, i1, 3] > conf_thres and kpt_3d[n, i2, 3] > conf_thres:
                ax1.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d[n, i1, 3] > conf_thres:
                ax1.scatter(kpt_3d[n, i1, 0], kpt_3d[n, i1, 2], -kpt_3d[n, i1, 1], c=colors[l], marker='o')
            if kpt_3d[n, i2, 3] > conf_thres:
                ax1.scatter(kpt_3d[n, i2, 0], kpt_3d[n, i2, 2], -kpt_3d[n, i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax1.set_title('3D vis')
    else:
        ax1.set_title(filename)
    ax2.set_title('radius')
    ax3.set_title('direction degree')
    ax4.set_title('point dis')
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Z Label')
    ax1.set_zlabel('Y Label')
    ax1.set_xlim(-500, 2000)
    ax1.set_ylim(1000, 5000)
    ax1.set_zlim(-750, 1200)
    # ax.legend()

    image = fig2data(fig)
    plt.close(fig)
    return image

radius = []
direction = []
pointdis = []
# cv2.namedWindow('win', 0)
for ids in images_ids:
    frame = cv2.imread(os.path.join(root_dir, ids.strip()+".jpg"))
    depth_image = np.load(os.path.join(root_dir, ids.strip()+".npy"))
    # output = my_trace.get_smooth_output(my_trackModule, det_setting, frame)
    output = my_trackModule.get_result(frame, **det_setting)
    if len(output)!=0:
        # bbox = output_track[:, :4].astype(np.int32)
        bbox = np.array([out["body_box"] for out in output]).astype(np.int32)
        # fbox = output_track[:, 4: 8].astype(np.int32)
        # flm = output_track[:, 8:18].astype(np.int32)
        id = np.array([out["id"] for out in output]).astype(np.uint8)
        point_dis = get_3d_depth_bybox(depth_image, point_pos, depth_scale, percent=20)
        point_p = np.array([[point_pos[0, 0] + (point_pos[0, 2] - point_pos[0, 0])/2.,
                   point_pos[0, 1] + (point_pos[0, 3] - point_pos[0, 1])/2.,
                   point_dis[0]]])
        point_p = pixel2cam(point_p, my_3dkp_model.focal, my_3dkp_model.princpt)
        # human = my_kp.get_output(frame, bbox[0, :][None, :])
        try:
            root_depth_list = get_3d_depth_bybox(depth_image, bbox, depth_scale, percent=20)
        except:
            print(1)
        kp_2d, kp_3d = my_3dkp_model.get_output(frame, bbox[0, :][None, :], root_depth_list)
        skele_3d = my_3dkp_skele_model.get_output(frame, kp_2d[:,:,:2], root_depth_list)
        kp_3d = np.concatenate([skele_3d, kp_3d[:,[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16],3:4]], axis=2)
        kp = kp_2d[0, 16].astype(int)
        # depth_six = depth_image[kp[1], kp[0]]
        color = (int(colours[id[0] % 32, 0]), int(colours[id[0] % 32, 1]), int(colours[id[0] % 32, 2]))
        cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])),
                      (int(bbox[0][2]), int(bbox[0][3])),
                                            color, 2)
        frame = vis_keypoints(frame, kp_2d[0], kp_thresh=-0.1)
        # td_image = vis_3d_multiple_skeleton_conf_addpoint_toimage(kp_3d, point_p,
        #                          conf_thres=0.05,
        #                          filename='output_pose_3d (x,y,z: camera-centered. mm.)')
        # p_shoulder = kp_3d[0, 9][:3].copy()
        p_shoulder = (kp_3d[0, 9][:3].copy() + kp_3d[0, 8][:3].copy())/2.
        p_hand = kp_3d[0, 15][:3].copy()
        p_shoulder[1] = -p_shoulder[1]
        p_hand[1] = -p_hand[1]
        p_foot = p_shoulder.copy()
        p_foot[1] = 0
        _, _, tan_A = tanAngle(p_shoulder, p_hand, p_foot)
        radius_ = L * tan_A
        print("radius,", radius_)
        p_center_z = p_shoulder.copy()
        p_center_z[1] = 0
        p_camera_z = p_shoulder.copy()
        p_center_z[0] = 0
        p_center_z[1] = 0
        p_hand_z = p_hand.copy()
        p_hand_z[1] = 0
        root_dis = root_depth_list[0]
        moving_dist, moving_direction = disAndAngle(p_center_z, p_camera_z, p_hand_z, radius_, root_dis, bias=-450)
        # moving_direction += 10
        cup_center_z = math.sin((90 - moving_direction) * np.pi / 180.) * moving_dist
        cup_center_x = p_hand[0] - math.cos((90 - moving_direction) * np.pi / 180.) * moving_dist
        cup_pred = np.array([[cup_center_x, 750, cup_center_z]])
        radius.append(radius_)
        direction.append(moving_direction)
        pointdis.append(moving_dist)
        radius = radius[-200:]
        direction = direction[-200:]
        pointdis = pointdis[-200:]
        td_image = vis_3d_multiple_skeleton_conf_whole_toimage(kp_3d, point_p, cup_pred,
                                                               radius, direction, pointdis,
                                                               conf_thres=-0.1,
                                                               filename='output_pose_3d (x,y,z: camera-centered. mm.)')
        print("cup_dis", cup_center_x, cup_center_z)
        print(moving_dist, moving_direction)
        cup_pred_2d = cam2pixel(cup_pred, my_3dkp_model.focal, my_3dkp_model.princpt)
        # fig = plt.figure()
        # kps_sixteen = kp_3d[0, 16]
        # kps.append(depth_six)
        # kps = kps[-500:]
        # plt.plot(kps)
        # image = fig2data(fig)
        # plt.close(fig)
        # cv2.ellipse(frame, (int(cup_pred_2d[0][0]), int(cup_pred_2d[0][1])), (20, 10), 0, 0, 360, (0, 0, 255), thickness=5)
        cat_image = np.hstack((frame, td_image))
        cv2.imshow("win", cat_image)
        # cv2.imshow("win_depth", td_image)
        cv2.imwrite(os.path.join(save_dir, ids.strip()+".jpg"), frame)
        cv2.imwrite(os.path.join(save_dir, ids.strip() + "_d.jpg"), td_image)
        # cv2.imshow("dis", image)
        raw_video_writer.write(cat_image)
        cv2.waitKey(1)
        # print(1)s