# @Time : 2021/7/13 11:20
# @Author : Leia.Ma
# @File : demo.py
# @Software: PyCharm

import copy
import json
import time
import cv2
import numpy as np
from src import Video
from src import TrackModule
from src import DeepSiamDRKPModel as DeepKPModel
from src import TrackFilter, LandmarkFilter
from src.utils import draw_hkp_simple
from src import actionState, stateNote

time_now = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())[5:16]

# -----------------------Load config-------------------------------------------
with open("../user/lib/setting_trail.json") as f:
    config = json.load(f)
gpu_setting = config["gpu_setting"]

# -----------------------Dection & Track-------------------------------------------
my_trackModule = TrackModule(**config["deepModel"]["deepDetection"]["structure"],
                             **gpu_setting,
                             **{"single_mode": True})
det_load_ret = my_trackModule.load_model(config["deepModel"]["deepDetection"]["model"])
det_setting = config["deepModel"]["deepDetection"]["parameters"]

# ---------------------------Keypoint-------------------------------------------
my_kp = DeepKPModel(**config["deepModel"]["deepKeypoint"]["structure"],
                    **gpu_setting)
kp_load_ret = my_kp.load_model(config["deepModel"]["deepKeypoint"]["model"])

# ---------------------------Filter-------------------------------------------
my_trace = TrackFilter()
my_kp_trace = LandmarkFilter()

# -----------------------Check load model ------------------------------------
if (not det_load_ret) or (not kp_load_ret):
    print("Load Model Error!")
    exit()


# ##############################Start########################################
input_video = '../HZJ_ori_08-06_11_08.avi'
camera_mode = True

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 24

record_path = '../user/output/poseMatch_{}.avi'.format(time_now)
video_writer = cv2.VideoWriter(filename=record_path, fourcc=fourcc, fps=fps, frameSize=(1280, 720))

if camera_mode:
    input_video = '0'
    ori_record_path = '../user/output/poseMatch_ori_{}.avi'.format(time_now)
    ori_video_writer = cv2.VideoWriter(filename=ori_record_path, fourcc=fourcc, fps=fps, frameSize=(1280, 720))

my_video = Video(camera_mode, input_video, video_width=1280, video_height=720)

colours = np.random.rand(32, 3) * 255
ret = True
frame_idx = -1

angle_thre = 30
# point_list = [[11, 9, 9, 7], [7, 6, 11, 9]]
point_list = [[7, 6, 11, 9]]

observation_time = 20
angles_diff_buffer = [0] * observation_time
last_angle = 0
stateBuffer = stateNote()
lmFilter = LandmarkFilter()


while ret:
    stateNow = actionState(stateBuffer)
    ret, frame = my_video.capOneFrame()
    frame_idx += 1

    if camera_mode:
        frame_ori = copy.deepcopy(frame)
        ori_video_writer.write(frame_ori)

    output_track = my_trace.get_smooth_output(my_trackModule, det_setting, frame)

    if len(output_track) != 0:
        # bbox = output_track[:, :4].astype(np.int32)
        # fbox = output_track[:, 4: 8].astype(np.int32)
        # flm = output_track[:, 8:18].astype(np.int32)
        # id = output_track[:, -1].astype(np.uint8)

        kp_now = my_kp_trace.get_smooth_landmark(my_kp, output_track, frame)

    #     out_cur = my_det.get_output(frame, **detection_setting)
    #     kp_now = lmFilter.get_smooth_landmark(my_kp, out_cur[1][0]['body_box'][None, :], frame)  # Smooth keypoint!

        actionNow, angleChange = stateNow.updateState(kp_now)

        frame = draw_hkp_simple(frame, kp_now[0], kpt_score_thr=0.05)

        cv2.putText(frame, str(stateBuffer.screenDirection), (600, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 3)
        cv2.putText(frame,'active = {}, action1={}, action2={}'.format(stateBuffer.active, stateBuffer.flagAction1, stateBuffer.flagAction2),
                    (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 3)

    cv2.imshow("frame", frame)
    video_writer.write(frame)
    cv2.waitKey(1)

    # except:
    #     print('Error!')
