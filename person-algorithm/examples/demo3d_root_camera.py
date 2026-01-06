# @Time : 2021/3/17 14:57
# @Author : Altair.Huazj
# @File : demo.py
# @Software: PyCharm
import json
import cv2
import os
import glob
import numpy as np
from src import DeepDetModel, Video
from src import DeepKPModel, TrackModule
from src import TrackFilter, LandmarkFilter
from src import RealSenceVideo
from src import DeepRootModel
from src import Deep3DKPModel
from src.utils import draw_hkp, draw_hkp_simple, get_3d_depth_bybox, vis_keypoints, vis_root_3d_multiple_skeleton_conf_toimage
import  time
from examples.utils import trans_kps

'''

focal = [150, 150] # x-axis, y-axis
princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis

'''
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
                             **{"single_mode": False})
det_setting = config["deepModel"]["deepDetection"]["parameters"]
det_load_ret = my_trackModule.load_model(config["deepModel"]["deepDetection"]["model"])
############################################################################################


# 3DRootNet algorithm########################################################################
my_3droot_model  = DeepRootModel(**config["deepModel"]["deep3dRootNet"]["structure"],
                                 **gpu_setting)
my_3droot_model.load_model(config["deepModel"]["deep3dRootNet"]["model"])
########################################################################


# 3Dkeypoint algorithm########################################################################
my_3dkp_model = Deep3DKPModel(**config["deepModel"]["deep3DKeypoint_deeboot"]["structure"],
                    **gpu_setting)
tdkp_load_ret = my_3dkp_model.load_model(config["deepModel"]["deep3DKeypoint"]["model"])
############################################################################################

# filter trace##############################################################################
my_trace = TrackFilter(diff_thres=0)
############################################################################################

# create video fourcc#######################################################################
input_videos = ["../2.mp4"]
# input_videos = glob.glob(r"D:\share\dataset\3d_kps_dataset\collections\temp2hzj\*.mp4")
# input_video = r"C:\Users\Altair\Desktop\ToAltair,20211208\trainVideo\cyr\1_Sit\WIN_20210815_15_27_43_Pro.mp4"
camera_mode = False

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
size = (1280, 720)

#my_video = Video(camera_mode, 0)


fps = 29
# size = (640, 480)
i_size = (640, 480)

# out = cv2.VideoWriter('3d_out.mp4', fourcc, 25, (1280,360))


############################################################################################

# load model check##########################################################################
if (not det_load_ret) or (not tdkp_load_ret):
    print("Load Model Error!")
    exit()
############################################################################################


all_time = []
# cv2.namedWindow('win',0)
kp_np = []
frame_np = []
idx = 0
for input_video in input_videos:
    my_video = Video(camera_mode, input_video, video_width=size[0], video_height=size[1])
    ret = True
    while ret:
        ret, frame = my_video.capOneFrame()
        if not ret:
            continue
        # frame = cv2.imread('0.jpg')
        frame_np.append(frame)
        T1 = time.time()
        output_track = my_trace.get_smooth_output(my_trackModule, det_setting, frame)
        if len(output_track)!=0:
            # bbox = output_track[:, :4].astype(np.int32)
            bbox = np.array([out["body_box"] for out in output_track]).astype(np.int32)
            bbox_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
            max_box_idx = bbox_area.argmax()
            # fbox = output_track[:, 4: 8].astype(np.int32)
            # flm = output_track[:, 8:18].astype(np.int32
            id = np.array([out["id"] for out in output_track]).astype(np.uint8)
            # human = my_kp.get_output(frame, bbox[0, :][None, :])
            try:


                root_depth_list = my_3droot_model.get_output(frame, bbox[max_box_idx, :][None, :])

                #print('rootnet_dist',root_depth_list)
            except:
                root_depth_list = [1]
                print(1)
            kp_2d, kp_3d = my_3dkp_model.get_output(frame, bbox[max_box_idx, :][None, :], root_depth_list)


            T2 = time.time()

            all_time.append((T2-T1)*1000)

            if len(all_time)==120:
                print('spend time ========',np.mean(all_time))
            color = (int(colours[id[max_box_idx] % 32, 0]), int(colours[id[max_box_idx] % 32, 1]), int(colours[id[max_box_idx] % 32, 2]))
            cv2.rectangle(frame, (int(bbox[max_box_idx][0]), int(bbox[max_box_idx][1])),
                          (int(bbox[max_box_idx][2]), int(bbox[max_box_idx][3])),
                                                color, 2)
            frame = vis_keypoints(frame, kp_2d[0], kp_thresh=-1)


            frame =  cv2.resize(frame,(640,480))
            # kp_3d[:, :, :3] -= kp_3d[:, 10, :3]
            td_image = vis_root_3d_multiple_skeleton_conf_toimage(kp_3d,
                                     conf_thres=-1,
                                     filename='output pose 3d origin')
                                     #filename='output_pose_3d origin(x,y,z: camera-centered. mm.)')
            cat_image = np.hstack((frame, td_image))
            # cv2.imwrite(r"C:\Users\Altair\Desktop\temp2hzj\res\{:0>6d}.jpg".format(idx), frame)
            # cv2.imwrite(r"C:\Users\Altair\Desktop\temp2hzj\res\{:0>6d}_d.jpg".format(idx), td_image)
            #out.write(frame)
            cat_image = cv2.resize(cat_image, (1280, 360))
            #out.write(cat_image)
            cv2.imshow("win", cat_image)
            #out.write(cat_image)

            idx += 1
            tran_kps = trans_kps(kp_2d[0], (256, 256))
            img = np.zeros((256, 256, 3))
            img = vis_keypoints(img, tran_kps, kp_thresh=-1)
            cv2.imshow("vis", img)
            cv2.waitKey(1)
            # kps = kp_2d[:, [10, 9, 8, 11, 14, 12, 15,
            #                 13, 16, 7, 0, 4, 1, 5, 2, 6, 3], :]
            # kps = kps - kps[:, 10:11, :]
            # new_kps = np.zeros((1, 34))
            # new_kps[0, :17] = kps[0, :, 0]
            # new_kps[0, 17:] = kps[0, :, 1]
            # new_kps[0, :17] -= new_kps[0, 10:11]
            # new_kps[0, 17:] -= new_kps[0, 27:28]
            kps = kp_2d[:, [0, 4, 5, 6, 1, 2, 3,
                            7, 8, 10, 14, 15, 16, 11, 12, 13], :]

            kp_np.append(kps[0])

            # print(1)

            #out.write(cat_image)
            # print(1)

np.save(r"D:\share\dataset\3d_kps_dataset\collections\temp2hzj\data_input.npy", np.array(kp_np))
# np.save(r"C:\Users\Altair\Desktop\temp2hzj\frame.npy", np.array(frame_np))
