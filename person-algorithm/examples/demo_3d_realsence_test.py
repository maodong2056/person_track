# @Time : 2021/3/17 14:57
# @Author : Altair.Huazj
# @File : demo.py
# @Software: PyCharm
import json
import cv2
import numpy as np
from src import DeepDetModel, Video
from src import DeepKPModel, TrackModule
from src import TrackFilter, LandmarkFilter
from src import RealSenceVideo
from src import Deep3DKPModel, Deep3DKPLiteModel
from src.utils import draw_hkp, draw_hkp_simple, get_3d_depth_bybox, vis_keypoints, \
    vis_3d_multiple_skeleton_conf_toimage, get_cam_depth_bybox
import matplotlib.pyplot as plt
from src.utils.draw_3dkps import fig2data
colours = np.random.rand(32, 3)*255

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
my_3dkp_model = Deep3DKPLiteModel(**config["deepModel"]["deep3DKeypoint"]["structure"],
                    **gpu_setting)
tdkp_load_ret = my_3dkp_model.load_model(config["deepModel"]["deep3DKeypoint"]["model"])
############################################################################################

# filter trace##############################################################################
my_trace = TrackFilter(diff_thres=7.5)
############################################################################################

# create video fourcc#######################################################################
input_video = "../2.mp4"
camera_mode = False
video_name = "../pose_3d_det.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15
size = (1280, 720)
raw_video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
my_video_cap = RealSenceVideo(video_width=640, video_height=480, video_fps=fps)
############################################################################################

# load model check##########################################################################
if (not det_load_ret) or (not tdkp_load_ret):
    print("Load Model Error!")
    exit()
############################################################################################
ret = True
kps = []
# cv2.namedWindow('win', 0)
xs = []
ys = []
zs = []
rotate = []
while ret:
    depth_image, frame = my_video_cap.capOneFrame()
    if depth_image is None:
        break
    # frame = cv2.imread('0.jpg')
    output_track = my_trace.get_smooth_output(my_trackModule, det_setting, frame)
    if len(output_track)!=0:
        # bbox = output_track[:, :4].astype(np.int32)
        bbox = np.array([out["body_box"] for out in output_track]).astype(np.int32)
        # fbox = output_track[:, 4: 8].astype(np.int32)
        # flm = output_track[:, 8:18].astype(np.int32)
        id = np.array([out["id"] for out in output_track]).astype(np.uint8)
        # human = my_kp.get_output(frame, bbox[0, :][None, :])
        try:
            root_depth_list = get_cam_depth_bybox(depth_image, bbox, my_3dkp_model.focal, my_3dkp_model.princpt,
                                                  my_video_cap.depth_scale, percent=20)
            # print(root_depth_list)
            xs.append(root_depth_list[0])
            ys.append(root_depth_list[1])
            zs.append(root_depth_list[2])
            rotate.append(root_depth_list[3])
        except:
            print(1)
        # kp_2d, kp_3d = my_3dkp_model.get_output(frame, bbox[0, :][None, :], root_depth_list)
        # kp = kp_2d[0, 16].astype(int)
        # # depth_six = depth_image[kp[1], kp[0]]
        color = (int(colours[id[0] % 32, 0]), int(colours[id[0] % 32, 1]), int(colours[id[0] % 32, 2]))
        cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])),
                      (int(bbox[0][2]), int(bbox[0][3])),
                                            color, 2)
        xs = xs[-500:]
        ys = ys[-500:]
        zs = zs[-500:]
        rotate = rotate[-500:]
        fig = plt.figure(figsize=(8, 8), dpi=90)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.plot(xs, c=(1.,0.,0.))
        ax2.plot(ys, c=(1.,1.,0.))
        ax3.plot(zs, c=(1.,0.,1.))
        ax4.plot(rotate, c=(0.,0.,1.))
        image = fig2data(fig)
        plt.close(fig)
        # frame = vis_keypoints(frame, kp_2d[0], kp_thresh=0.05)
        # td_image = vis_3d_multiple_skeleton_conf_toimage(kp_3d,
        #                          conf_thres=0.05,
        #                          filename='output_pose_3d (x,y,z: camera-centered. mm.)')
        # fig = plt.figure()
        # kps_sixteen = kp_3d[0, 16]
        # kps.append(depth_six)
        # kps = kps[-500:]
        # plt.plot(kps)
        # image = fig2data(fig)
        # plt.close(fig)
        # cat_image = np.hstack((frame, td_image))
        cv2.imshow("win", frame)
        cv2.imshow("para", image)
        # cv2.imshow("dis", image)
        cv2.waitKey(1)
        # print(1)