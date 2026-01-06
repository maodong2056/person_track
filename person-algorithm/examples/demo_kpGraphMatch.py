# @Time : 2021/5/25 14:57
# @Author : Leia.Ma
# @File : demo.py 
# @Software: PyCharm
import torch
from src import DeepDetModel, Video
from src import DeepKPModel, DeepSiamDRKPModel
from src import DeepGraphKPModel
from src import TrackModule
from src.utils import draw_hkp, draw_hkp_simple
from src import TrackFilter, LandmarkFilter
import json
import cv2
import numpy as np
import copy

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

graph_model = '../user/model/model_kpGraph_epoch210.pt'
my_kpGraph = DeepGraphKPModel(**gpu_setting)
load_kpGraph = my_kpGraph.load_model(graph_model)

camera_mode = False
if camera_mode:
    input_video = "0"
else:
    input_video = '../Demo_0707_ori.avi'

my_video = Video(camera_mode, input_video, video_width=1280, video_height=720)

record_path = '../out/test_result.avi'
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 15
video_writer = cv2.VideoWriter(filename=record_path, fourcc=fourcc, fps=fps, frameSize=(1280, 720))

colours = np.random.rand(32, 3) * 255
ret = True
frame_idx = -1

bbox_1 = []
bbox_2 = []

while ret:
    ret, frame = my_video.capOneFrame()
    frame_idx += 1

    try:
        # -------------detection only----------------
        # output_det = my_det.get_output(frame)
        # if len(output_det) != 0 and len(output_det[1]) >= 2:
        #     bbox_1 = output_det[1][0]['body_box'].astype(np.int32)
        #     bbox_2 = output_det[1][1]['body_box'].astype(np.int32)
        #     kp_1 = my_kp.get_output(frame, bbox_1[None, :])
        #     kp_2 = my_kp.get_output(frame, bbox_2[None, :])

        output_track = my_trace.get_smooth_output(my_trackModule, det_setting, frame)
        kp_now = my_kp_trace.get_smooth_landmark(my_kp, output_track, frame)

        if kp_now.shape[0] >= 2:
            kp_1 = kp_now[0]
            kp_2 = kp_now[1]
            bbox_1 = output_track[0]['body_box']
            bbox_2 = output_track[1]['body_box']

            feature_1 = my_kpGraph.get_feature(kp_1, bbox_1)
            feature_2 = my_kpGraph.get_feature(kp_2, bbox_2)

            # Euclidian distance
            diff = feature_1 - feature_2
            dist_sq = torch.sum(pow(diff, 2), 1)
            match_score = torch.sqrt(dist_sq)
            match_score_ = round(float(match_score), 3)
            print(match_score_)

            frame_ori = copy.deepcopy(frame)
            color_1 = (int(colours[1 % 32, 0]), int(colours[1 % 32, 1]), int(colours[1 % 32, 2]))
            frame = draw_hkp_simple(frame, kp_1)
            cv2.rectangle(frame, (int(bbox_1[0]), int(bbox_1[1])), (int(bbox_1[2]), int(bbox_1[3])), [255, 255, 0], 2)

            color_2 = (int(colours[2 % 32, 0]), int(colours[2 % 32, 1]), int(colours[2 % 32, 2]))
            frame = draw_hkp_simple(frame, kp_2)
            cv2.rectangle(frame, (int(bbox_2[0]), int(bbox_2[1])), (int(bbox_2[2]), int(bbox_2[3])), [0, 255, 255], 2)

            cv2.putText(frame,
                        'matching score = {}, frame_idx={}'.format(match_score_, frame_idx),
                        (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 3)
            cv2.imshow("frame", frame)
            video_writer.write(frame)
            cv2.waitKey(1)

    except:
        print('Error!')