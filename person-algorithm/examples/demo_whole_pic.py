# @Time : 2021/3/17 14:57
# @Author : Altair.Huazj
# @File : demo.py
# @Software: PyCharm
import json
import cv2
import numpy as np
from src import DeepDetModel, DeepDetModelWhole, Video
# from src import DeepKPModel
from src import TrackWholeModule
from src import DeepSiamDRKPModel as DeepKPModel
from src import TrackFilter, LandmarkFilter
from src.utils import draw_hkp, draw_hkp_simple
import glob

colours = np.random.rand(32, 3)*255

# load config file##########################################################################
with open("../user/lib/setting_whole_trail.json") as f:
    config = json.load(f)
# gpu setting
gpu_setting = config["gpu_setting"]
############################################################################################

# det&track algorithm#######################################################################
my_trackModule = DeepDetModelWhole(**config["deepModel"]["deepDetection"]["structure"],
                             **gpu_setting)
det_setting = config["deepModel"]["deepDetection"]["parameters"]
det_load_ret = my_trackModule.load_model(config["deepModel"]["deepDetection"]["model"])
# my_detModule = DeepDetModelWhole(**config["deepModel"]["deepDetection"]["structure"],
#                              **gpu_setting,
#                              **{"single_mode": True})
# det_setting = config["deepModel"]["deepDetection"]["parameters"]
# det_load_ret = my_detModule.load_model(config["deepModel"]["deepDetection"]["model"])
############################################################################################

# keypoint algorithm########################################################################
my_kp = DeepKPModel(**config["deepModel"]["deepKeypoint"]["structure"],
                    **gpu_setting)
kp_load_ret = my_kp.load_model(config["deepModel"]["deepKeypoint"]["model"])
############################################################################################

# filter trace##############################################################################
my_trace = TrackFilter()
my_kp_trace = LandmarkFilter()
############################################################################################

# create video fourcc#######################################################################
input_video = "../2.mp4"
camera_mode = False
video_name = "../pose_to_hand_raw_det.avi"
fourcc = cv2.VideoWriter_fourcc(*'mjpg')
fps = 25
size = (1280, 720)
# raw_video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
# my_video = Video(camera_mode, input_video)
# my_video = Video(True, 0, video_width=1280, video_height=720)
############################################################################################

# load model check##########################################################################
if (not det_load_ret) or (not kp_load_ret):
    print("Load Model Error!")
    exit()
############################################################################################
color = (np.random.random((18, 3)) * 255).astype(np.uint8)
ret = True
# follow_id = 0
dir = r"D:\project\person_detection\return_data_20210805\2_CameraRS"
pics = glob.glob(dir+"\*.jpg")
for pic in pics[0:]:
    input_image = cv2.imread(pic)
    shape, output = my_trackModule.get_output(input_image, **det_setting)
    for idx, out in enumerate(output):
        cv2.rectangle(input_image, (int(out['body_box'][0]), int(out['body_box'][1])),
                      (int(out['body_box'][2]), int(out['body_box'][3])), colours[idx], 2)
        if out['face_conf'] is not None:
            cv2.rectangle(input_image, (int(out['face_box'][0]), int(out['face_box'][1])),
                          (int(out['face_box'][2]), int(out['face_box'][3])), colours[idx], 2)
            cv2.circle(input_image, (int(out['face_lm'][0]), int(out['face_lm'][1])), 1, (0, 0, 255), 4)
            cv2.circle(input_image, (int(out['face_lm'][2]), int(out['face_lm'][3])), 1, (0, 255, 255), 4)
            cv2.circle(input_image, (int(out['face_lm'][4]), int(out['face_lm'][5])), 1, (255, 0, 255), 4)
            cv2.circle(input_image, (int(out['face_lm'][6]), int(out['face_lm'][7])), 1, (0, 255, 0), 4)
            cv2.circle(input_image, (int(out['face_lm'][8]), int(out['face_lm'][9])), 1, (255, 0, 0), 4)
        if out['lh_box'] is not None:
            cv2.rectangle(input_image, (int(out['lh_box'][0]), int(out['lh_box'][1])),
                          (int(out['lh_box'][2]), int(out['lh_box'][3])), colours[idx], 2)
            cv2.circle(input_image, (int(out['lh_lm'][0]), int(out['lh_lm'][1])), 1, (0, 0, 255), 4)
            cv2.circle(input_image, (int(out['lh_lm'][2]), int(out['lh_lm'][3])), 1, (0, 255, 255), 4)
            cv2.circle(input_image, (int(out['lh_lm'][4]), int(out['lh_lm'][5])), 1, (255, 0, 255), 4)
            cv2.circle(input_image, (int(out['lh_lm'][6]), int(out['lh_lm'][7])), 1, (0, 255, 0), 4)
            cv2.putText(input_image, "left", (int(out['lh_box'][0]), int(out['lh_box'][1])),
                        cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 0], 1)
        if out['rh_box'] is not None:
            cv2.rectangle(input_image, (int(out['rh_box'][0]), int(out['rh_box'][1])),
                          (int(out['rh_box'][2]), int(out['rh_box'][3])), colours[idx], 2)
            cv2.circle(input_image, (int(out['rh_lm'][0]), int(out['rh_lm'][1])), 1, (0, 0, 255), 4)
            cv2.circle(input_image, (int(out['rh_lm'][2]), int(out['rh_lm'][3])), 1, (0, 255, 255), 4)
            cv2.circle(input_image, (int(out['rh_lm'][4]), int(out['rh_lm'][5])), 1, (255, 0, 255), 4)
            cv2.circle(input_image, (int(out['rh_lm'][6]), int(out['rh_lm'][7])), 1, (0, 255, 0), 4)
            cv2.putText(input_image, "right", (int(out['rh_box'][0]), int(out['rh_box'][1])),
                        cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1)
    cv2.imshow("res", input_image)
    cv2.waitKey(0)
# while ret:
#     ret, input_image = my_video.capOneFrame()
#     frame_id = my_video.nowFrame - 1
#     # output = my_trackModule.get_result(input_image)
#     output = my_trace.get_smooth_output(my_trackModule, det_setting, input_image)
#     for idx, out in enumerate(output):
#         # print(out)
#         # human_kp_n = my_kp.get_output(input_image, out['body_box'][None, :])
#         # input_image = draw_hkp_simple(input_image, human_kp_n[0])
#         c = color[out["id"]%18].tolist()
#         cv2.rectangle(input_image, (int(out['body_box'][0]), int(out['body_box'][1])),
#                       (int(out['body_box'][2]), int(out['body_box'][3])), c, 2)
#         if out['face_conf'] is not None:
#             cv2.rectangle(input_image, (int(out['face_box'][0]), int(out['face_box'][1])),
#                           (int(out['face_box'][2]), int(out['face_box'][3])), c, 2)
#             cv2.circle(input_image, (int(out['face_lm'][0]), int(out['face_lm'][1])), 1, (0, 0, 255), 4)
#             cv2.circle(input_image, (int(out['face_lm'][2]), int(out['face_lm'][3])), 1, (0, 255, 255), 4)
#             cv2.circle(input_image, (int(out['face_lm'][4]), int(out['face_lm'][5])), 1, (255, 0, 255), 4)
#             cv2.circle(input_image, (int(out['face_lm'][6]), int(out['face_lm'][7])), 1, (0, 255, 0), 4)
#             cv2.circle(input_image, (int(out['face_lm'][8]), int(out['face_lm'][9])), 1, (255, 0, 0), 4)
#         if out['lh_box'] is not None:
#             cv2.rectangle(input_image, (int(out['lh_box'][0]), int(out['lh_box'][1])),
#                           (int(out['lh_box'][2]), int(out['lh_box'][3])), c, 2)
#             cv2.circle(input_image, (int(out['lh_lm'][0]), int(out['lh_lm'][1])), 1, (0, 0, 255), 4)
#             cv2.circle(input_image, (int(out['lh_lm'][2]), int(out['lh_lm'][3])), 1, (0, 255, 255), 4)
#             cv2.circle(input_image, (int(out['lh_lm'][4]), int(out['lh_lm'][5])), 1, (255, 0, 255), 4)
#             cv2.circle(input_image, (int(out['lh_lm'][6]), int(out['lh_lm'][7])), 1, (0, 255, 0), 4)
#             cv2.putText(input_image, "left", (int(out['lh_box'][0]), int(out['lh_box'][1])),
#                         cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 0], 1)
#         if out['rh_box'] is not None:
#             cv2.rectangle(input_image, (int(out['rh_box'][0]), int(out['rh_box'][1])),
#                           (int(out['rh_box'][2]), int(out['rh_box'][3])), c, 2)
#             cv2.circle(input_image, (int(out['rh_lm'][0]), int(out['rh_lm'][1])), 1, (0, 0, 255), 4)
#             cv2.circle(input_image, (int(out['rh_lm'][2]), int(out['rh_lm'][3])), 1, (0, 255, 255), 4)
#             cv2.circle(input_image, (int(out['rh_lm'][4]), int(out['rh_lm'][5])), 1, (255, 0, 255), 4)
#             cv2.circle(input_image, (int(out['rh_lm'][6]), int(out['rh_lm'][7])), 1, (0, 255, 0), 4)
#             cv2.putText(input_image, "right", (int(out['rh_box'][0]), int(out['rh_box'][1])),
#                         cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1)
#     cv2.imshow("res", input_image)
#     raw_video_writer.write(input_image)
#     cv2.waitKey(1)
    # print(1)