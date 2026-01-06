import json
import cv2
# import pandas as pd
import numpy as np
from src import Video
from src import TrackModule, MatchModule

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

# match algorithm###########################################################################
my_matchModule = MatchModule(**gpu_setting)
match_load_ret = my_matchModule.load_model(reid_model=config["deepModel"]["deepReidRec"]["model"],
                                           face_model=config["deepModel"]["deepFaceRec"]["model"])
############################################################################################

# create video fourcc#######################################################################
# input_video = "../2.mp4"
camera_mode = True
# video_name = "../match.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15
size = (1280, 720)
# video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
my_video = Video(camera_mode, 0)
############################################################################################

# load model check##########################################################################
if (not det_load_ret) or (not match_load_ret):
    print("Load Model Error!")
    exit()
############################################################################################

ret = True
while ret:
    ret, frame = my_video.capOneFrame()
    now_frame = my_video.nowFrame - 1
    det_result = my_trackModule.get_result(frame, **det_setting)
    # det_result = np.array(det_result)
    # det_result[:, 0:2] = det_result[:, 0:2] - det_result[:, 2:4]/2
    # det_result[:, 2:4] = det_result[:, 0:2] + det_result[:, 2:4]
    # # if (det_result[:, 4:8] != -1).all():
    # det_result[:, 4:6] = det_result[:, 4:6] - det_result[:, 6:8] / 2
    # det_result[:, 6:8] = det_result[:, 4:6] + det_result[:, 6:8]
    # det_result = det_result.astype(np.int)
    cv2.putText(frame, str(now_frame), (0, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 3)
    if len(det_result) != 0:
        my_matchModule.match(frame, det_result)
        persons = my_matchModule.get_confirmed_person()
        for person in persons:
            name = person.person_name
            bbox = person.body.bbox
            fbox = person.face.bbox
            flm = person.face.landmark
            track_id = person.confirmedTrackID
            color = (person.person_color[2],person.person_color[1],person.person_color[0])
            t_size = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[0]) + t_size[0] + 3, int(bbox[1]) - t_size[1] - 4), color, -1)
            cv2.putText(frame, str(name), (int(bbox[0]), int(bbox[1]) - 2),
                        cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            if (fbox > 0).all():
                cv2.rectangle(frame, (int(fbox[0]), int(fbox[1])), (int(fbox[2]), int(fbox[3])), color, 2)
                cv2.circle(frame, (int(flm[0]), int(flm[1])), 1, (0, 0, 255), 4)
                cv2.circle(frame, (int(flm[2]), int(flm[3])), 1, (0, 255, 255), 4)
                cv2.circle(frame, (int(flm[4]), int(flm[5])), 1, (255, 0, 255), 4)
                cv2.circle(frame, (int(flm[6]), int(flm[7])), 1, (0, 255, 0), 4)
                cv2.circle(frame, (int(flm[8]), int(flm[9])), 1, (255, 0, 0), 4)
    cv2.imshow("result", frame)
    cv2.waitKey(1)
