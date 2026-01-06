# @Time : 2021/3/17 14:57 
# @Author : Altair.Huazj
# @File : demo.py 
# @Software: PyCharm
import json
import cv2
import numpy as np
from src import DeepDetModel, Video
# from src import DeepKPModel
from src import TrackModule
# from src import DeepSiamDRKPModel as DeepKPModel
from src import DeepKPModel as DeepKPModel
from src import TrackFilter, LandmarkFilter
from src.utils import draw_hkp, draw_hkp_simple

colours = np.random.rand(32,3)*255

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
camera_mode = True
video_name = "../pose_to_hand_raw_det.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15
size = (1280, 720)
raw_video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
# my_video = Video(camera_mode, 0, video_width=1280, video_height=720)
my_video = Video(False, input_video)
############################################################################################

# load model check##########################################################################
if (not det_load_ret) or (not kp_load_ret):
    print("Load Model Error!")
    exit()
############################################################################################

ret = True
# follow_id = 0
while ret:
    ret, frame = my_video.capOneFrame()
    frame_id = my_video.nowFrame - 1
    # frame = cv2.imread("../debug_rotate.jpg")
    # cv2.putText(frame, str(frame_id), (0, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 3)
    # first_time = True
    # if frame_id > 143:
    # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # frame = cv2.flip(frame, 1)
    # my_trackModule.clear()
    #     if first_time:
    #         my_trace.track_box = None
    #         my_trace.previous_image = None
    #         my_kp_trace.previous_landmarks_set = None
    #         my_kp_trace.landmark_dict = {}
    #         my_trackModule.clear()
    #         first_time = False

    # frame_n = frame.copy()
    # frame = cv2.imread("/home/tmq/Documents/handtracking_data/img/Screenshot from pose_to_hand_02.avi.png")
    # raw_video_writer.write(frame)
    output_track = my_trace.get_smooth_output(my_trackModule, det_setting, frame)
    # output_track_n = my_trackModule.get_result(frame, **detection_setting)
    # if len(output_track)!=0 and len(output_track_n)!=0:
    if len(output_track) != 0:
        # bbox = output_track[:, :4].astype(np.int32)
        # fbox = output_track[:, 4: 8].astype(np.int32)
        # flm = output_track[:, 8:18].astype(np.int32)
        # id = output_track[:, -1].astype(np.uint8)
        # bbox_n = output_track_n[:, :4].astype(np.int32)
        # fbox_n = output_track_n[:, 4: 8].astype(np.int32)
        # flm_n = output_track_n[:, 8:18].astype(np.int32)
        # id_n = output_track_n[:, -1].astype(np.uint8)
    #     # if follow_id not in id:
    #     #     for track in my_trackModule.tracker.tracks:
    #     #         if track.track_id == follow_id and track.is_deleted():
    #     #             conf = conf
    #     #             max_conf = np.argmax(conf)
    #     #             idx_ = max_conf
    #     #             follow_id = id[idx_]
    #     #             break
    #     #     if follow_id==0:
    #     #         conf = conf
    #     #         max_conf = np.argmax(conf)
    #     #         idx_ = max_conf
    #     #         follow_id = id[idx_]
    #     # else:
    #     #     if follow_id not in id:
    #     #         cv2.imshow("test", frame)
    #     #         cv2.waitKey(1)
    #     #         continue
    #     #     idx_ = np.argwhere(id == follow_id)[0, 0]
    #     if follow_id not in id:
    #         conf = conf
    #         max_conf = np.argmax(conf)
    #         idx_ = max_conf
    #         follow_id = id[idx_]
    #     else:
    #         idx_ = np.argwhere(id == follow_id)[0, 0]
        human_kp = my_kp_trace.get_smooth_landmark(my_kp, output_track, frame)
    #     human_kp = my_kp.get_output(frame, bbox[0, :][None, :])
    #     human_kp_n = my_kp.get_output(frame, bbox[0, :][None, :])
    #     color = (int(colours[id[0] % 32, 0]), int(colours[id[0] % 32, 1]), int(colours[id[0] % 32, 2]))
        color = (int(colours[output_track[0]["id"] % 32, 0]),
                 int(colours[output_track[0]["id"] % 32, 1]),
                 int(colours[output_track[0]["id"] % 32, 2]))
        right_elbow = human_kp[0][8]
        right_wrist = human_kp[0][10]
        # right_elbow_n = human_kp_n[0][8]
        # right_wrist_n = human_kp_n[0][10]

        if right_elbow[1] > right_wrist[1]:
            if output_track[0]["face_box"] is not None:
                side = max(output_track[0]["face_box"][3] - output_track[0]["face_box"][1],
                           output_track[0]["face_box"][2] - output_track[0]["face_box"][0])
            else:
                side = -1
            side = side * 2.5
            # side_n = max(fbox_n[0][3] - fbox_n[0][1], fbox_n[0][2] - fbox_n[0][0])
            # side_n = side_n * 2.5

            x = right_wrist[0]
            y = right_wrist[1] - side / 3
            w = side
            h = side

            # x_n = right_wrist_n[0]
            # y_n = right_wrist_n[1] - side_n / 3
            # w_n = side_n
            # h_n = side_n

            # print(right_wrist)
            # print((x, y))
            # print(side)
            # input()

            hbox = np.array([x, y, w, h, 0])[None, :]
            # hbox_n = np.array([x_n, y_n, w_n, h_n, 0])[None, :]

            if side < 50:
                cv2.imshow("test", frame)
                cv2.waitKey(1)
                continue
            else:
                # print(x)
                # print(y)
                # print(w)
                # print(h)
                xmin = max(int(x - w / 2), 0)
                xmax = min(int(x + w / 2), frame.shape[1])
                ymin = max(int(y - h / 2), 0)
                ymax = min(int(y + h / 2), frame.shape[0])
                # print(xmin)
                # print(ymin)
                # print(xmax)
                # print(ymax)

                img = frame[ymin:ymax, xmin:xmax]
                # cv2.imshow("debug", img)

                norm_hbox = hbox[0]
                # norm_hbox_n = hbox_n[0]
                # norm_hbox[0] = norm_hbox[0] / frame.shape[1]
                # norm_hbox[1] = norm_hbox[1] / frame.shape[0]
                # norm_hbox[2] = norm_hbox[2] / frame.shape[1]
                # norm_hbox[3] = norm_hbox[3] / frame.shape[0]
                # print(norm_hbox)
                cv2.rectangle(frame, (int(norm_hbox[0]-norm_hbox[2]/2), int(norm_hbox[1]-norm_hbox[2]/2)),
                              (int(norm_hbox[0]+norm_hbox[2]/2), int(norm_hbox[1]+norm_hbox[3]/2)),
                              color, 2)
                # cv2.rectangle(frame_n, (int(norm_hbox_n[0]-norm_hbox_n[2]/2), int(norm_hbox_n[1]-norm_hbox_n[2]/2)),
                #               (int(norm_hbox_n[0]+norm_hbox_n[2]/2), int(norm_hbox_n[1]+norm_hbox_n[3]/2)),
                #               color, 2)
                # frame = draw_rotate_rect(frame, norm_hbox)
                #
                #
                # norm_landmarks = hand_tracking(img)
                # if norm_landmarks is not None:
                #     norm_landmarks = landmarks_projection(norm_hbox, norm_landmarks)
                #     frame = draw_norm_landmarks_and_roi(frame, norm_landmarks, norm_hbox)

                # cv2.rectangle(frame, (int(x), int(y)),
                #               (int(x + w), int(y + h)),
                #               color, 2)
            # points = []
            # points.append((int(right_elbow[0]), int(right_elbow[1])))
            # points.append((int(right_wrist[0]), int(right_wrist[1])))
            # frame = draw_points(frame, points)

        cv2.rectangle(frame, (int(output_track[0]["body_box"][0]), int(output_track[0]["body_box"][1])),
                      (int(output_track[0]["body_box"][2]), int(output_track[0]["body_box"][3])),
                                            color, 2)
        # try:
        #     cv2.rectangle(frame, (int(output_track[1]["body_box"][0]), int(output_track[1]["body_box"][1])),
        #                   (int(output_track[1]["body_box"][2]), int(output_track[1]["body_box"][3])),
        #                                         color, 2)
        # except:
        #     pass
        # cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])),
        #               (int(bbox[0][2]), int(bbox[0][3])),
        #                                     color, 2)
        frame = draw_hkp_simple(frame, human_kp[0], kpt_score_thr=0.05)
        if output_track[0]["face_box"] is not None:
            cv2.rectangle(frame, (int(output_track[0]["face_box"][0]), int(output_track[0]["face_box"][1])),
                          (int(output_track[0]["face_box"][2]), int(output_track[0]["face_box"][3])),
                                                color, 2)
        # cv2.rectangle(frame, (int(fbox[0][0]), int(fbox[0][1])),
        #               (int(fbox[0][2]), int(fbox[0][3])),
        #                                     color, 2)

        # cv2.rectangle(frame_n, (int(bbox_n[0][0]), int(bbox_n[0][1])),
        #               (int(bbox_n[0][2]), int(bbox_n[0][3])),
        #                                     color, 2)
        # cv2.rectangle(frame_n, (int(bbox_n[0][0]), int(bbox_n[0][1])),
        #               (int(bbox_n[0][2]), int(bbox_n[0][3])),
        #                                     color, 2)
        # frame_n = draw_hkp_simple(frame_n, human_kp_n[0])
        # cv2.rectangle(frame_n, (int(fbox_n[0][0]), int(fbox_n[0][1])),
        #               (int(fbox_n[0][2]), int(fbox_n[0][3])),
        #                                     color, 2)
    # cat_images = np.hstack((frame, frame_n))
    # video_writer.write(frame)
    cv2.imshow("test", frame)
    raw_video_writer.write(frame)
    cv2.waitKey(1)

