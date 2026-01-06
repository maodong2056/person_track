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
from src import DeepRootModel
from src import Deep3DKPModel
from src.utils import draw_hkp, draw_hkp_simple, get_3d_depth_bybox, vis_keypoints, vis_3d_multiple_skeleton_conf_toimage

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


# 3DRootNet algorithm########################################################################
my_3droot_model  = DeepRootModel(**config["deepModel"]["deep3dRootNet"]["structure"],
                                 **gpu_setting)
my_3droot_model.load_model(config["deepModel"]["deep3dRootNet"]["model"])
########################################################################


# 3Dkeypoint algorithm########################################################################
my_3dkp_model = Deep3DKPModel(**config["deepModel"]["deep3DKeypoint"]["structure"],
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
# my_video_cap = RealSenceVideo(video_width=640, video_height=480, video_fps=fps)
my_video_cap = Video(camera_mode, input_video)
############################################################################################

# load model check##########################################################################
if (not det_load_ret) or (not tdkp_load_ret):
    print("Load Model Error!")
    exit()
############################################################################################
ret = True

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
            root_depth_list_real = get_3d_depth_bybox(depth_image, bbox, my_video_cap.depth_scale, percent=20)

            root_depth_list = my_3droot_model.get_output(frame, bbox[0, :][None, :])

            print('realsense dist',root_depth_list_real)
            print('rootnet_dist',root_depth_list)
        except:
            print(1)
        kp_2d, kp_3d = my_3dkp_model.get_output(frame, bbox[0, :][None, :], root_depth_list)
        color = (int(colours[id[0] % 32, 0]), int(colours[id[0] % 32, 1]), int(colours[id[0] % 32, 2]))
        cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])),
                      (int(bbox[0][2]), int(bbox[0][3])),
                                            color, 2)
        frame = vis_keypoints(frame, kp_2d[0], kp_thresh=0.25)
        td_image = vis_3d_multiple_skeleton_conf_toimage(kp_3d,
                                 conf_thres=0.25,
                                 filename='output_pose_3d (x,y,z: camera-centered. mm.)')
        cat_image = np.hstack((frame, td_image))
        cv2.imshow("win", cat_image)
        cv2.waitKey(1)
        # print(1)