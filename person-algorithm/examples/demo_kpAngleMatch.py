# @Time : 2021/7/13 11:20
# @Author : Leia.Ma
# @File : demo.py
# @Software: PyCharm

import copy
import json
import time

import cv2
import numpy as np

from src import DeepDetModel, Video
from src import TrackModule
from src import TrackFilter, LandmarkFilter
from src import DeepKPModel, DeepSiamDRKPModel
from src.utils import draw_hkp_with_angle_wrong

time_now = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())[5:16]

# -----------------------Load config-------------------------------------------
with open("../user/lib/setting_trail.json") as f:
    config = json.load(f)
gpu_setting = config["gpu_setting"]

# ----------------------- Only Detection-------------------------------------------
# my_det = DeepDetModel(**config["deepModel"]["deepDetection"]["structure"], **gpu_setting)
# my_det_load_ret = my_det.load_model(config["deepModel"]["deepDetection"]["model"])

# -----------------------Dection & Track--------------------------------------------
my_trackModule = TrackModule(**config["deepModel"]["deepDetection"]["structure"],
                             **gpu_setting,
                             **{"single_mode": False})
det_load_ret = my_trackModule.load_model(config["deepModel"]["deepDetection"]["model"])
det_setting = config["deepModel"]["deepDetection"]["parameters"]

# ---------------------------Filter-------------------------------------------
my_trace = TrackFilter()
my_kp_trace = LandmarkFilter()

# ---------------------------Keypoint-------------------------------------------
my_kp = DeepSiamDRKPModel(**config["deepModel"]["deepKeypoint"]["structure"],
                    **gpu_setting)
kp_load_ret = my_kp.load_model(config["deepModel"]["deepKeypoint"]["model"])

# -----------------------Check load model ------------------------------------
if (not det_load_ret) or (not kp_load_ret):
    print("Load Model Error!")
    exit()

input_video = '../Demo_0707_ori.avi'
camera_mode = False

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 15

record_path = 'out/poseMatch_{}.avi'.format(time_now)
video_writer = cv2.VideoWriter(filename=record_path, fourcc=fourcc, fps=fps, frameSize=(1280, 720))

if camera_mode:
    input_video = '0'
    ori_record_path = 'out/poseMatch_ori_{}.avi'.format(time_now)
    ori_video_writer = cv2.VideoWriter(filename=ori_record_path, fourcc=fourcc, fps=fps, frameSize=(1280, 720))

my_video = Video(camera_mode, input_video, video_width=1280, video_height=720)

colours = np.random.rand(32, 3) * 255
ret = True
frame_idx = -1

angle_thre = 30
point_list = [[7, 9, 11], [6, 8, 10], [9, 7, 13], [8, 6, 12],
              [7, 13, 12], [6, 12, 13], [12, 13, 15], [13, 12, 14], [13, 15, 17], [12, 14, 16]]


def GetCrossAngle(point1, point2, point3):
    arr_0 = np.array([(point2[0] - point1[0]), (point2[1] - point1[1])])
    arr_1 = np.array([(point2[0] - point3[0]), (point2[1] - point3[1])])
    angle = clockwise_angle(arr_0, arr_1)
    return angle


def cos_angle(v1, v2):
    dot = float(v1.dot(v2))
    det = np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2))
    cos_value = dot / det
    angle = 180 - (np.arccos(cos_value) * (180 / np.pi))
    angle = round(angle, 1)
    return angle


def clockwise_angle(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    theta = np.arctan2(det, dot)
    theta = theta if theta > 0 else 2 * np.pi + theta
    angle = 180 - theta * 180 / np.pi
    angle = round(angle, 1)
    return angle


score_list_ori = []

while ret:
    ret, frame = my_video.capOneFrame()
    # frame = cv2.imread('testPic.jpg')
    frame_idx += 1

    try:
        # -------------detection only----------------
        # out_cur = my_det.get_output(frame)
        # bbox_1 = out_cur[1][0]['body_box'].astype(np.int32)
        # kp_1 = my_kp.get_output(frame, bbox_1[None, :])
        # bbox_2 = out_cur[1][1]['body_box'].astype(np.int32)
        # kp_2 = my_kp.get_output(frame, bbox_2[None, :])

        output_track = my_trace.get_smooth_output(my_trackModule, det_setting, frame)
        kp_now = my_kp_trace.get_smooth_landmark(my_kp, output_track, frame)

        if kp_now.shape[0] >= 2:
            Angle_list1 = []
            Angle_list2 = []
            Dist = []
            wrong_point = []
            wrong_skeleton = []
            kp_1 = kp_now[0]
            kp_2 = kp_now[1]

            for p in point_list:
                a = GetCrossAngle(kp_1[p[0] - 1], kp_1[p[1] - 1], kp_1[p[2] - 1])
                Angle_list1.append(a)

                b = GetCrossAngle(kp_2[p[0] - 1], kp_2[p[1] - 1], kp_2[p[2] - 1])
                Angle_list2.append(b)

                Dist.append(a - b)
                if np.abs(a - b) >= angle_thre:
                    wrong_point.append(p[1])
                    wrong_skeleton.append([p[1], p[0]])
                    wrong_skeleton.append([p[1], p[2]])
            print(Dist)

            if camera_mode:
                frame_ori = copy.deepcopy(frame)
                ori_video_writer.write(frame_ori)

            # frame = draw_hkp_simple(frame, kp_1[0])
            # frame = draw_hkp_simple(frame, kp_2[0])

            frame = draw_hkp_with_angle_wrong(frame, kp_1, wrong_point, wrong_skeleton)
            frame = draw_hkp_with_angle_wrong(frame, kp_2, wrong_point, wrong_skeleton)

            bbox_1 = output_track[0]['body_box']
            bbox_2 = output_track[1]['body_box']

            cv2.rectangle(frame, (int(bbox_1[0]), int(bbox_1[1])), (int(bbox_1[2]), int(bbox_1[3])), [255, 255, 0], 2)
            cv2.rectangle(frame, (int(bbox_2[0]), int(bbox_2[1])), (int(bbox_2[2]), int(bbox_2[3])), [0, 255, 255], 2)

            cv2.imshow("frame", frame)
            video_writer.write(frame)
            cv2.waitKey(1)

    except:
        print('Error!')
