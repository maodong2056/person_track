import cv2
import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from src import TrackWholeModule
from src import DeepSiamDRKPModel as DeepKPModel
from src import TrackFilter, LandmarkFilter
from src import Deep3DKPModel, Deep3DKPLiteModel
from src.utils.draw_3dkps import fig2data
from src.utils import draw_hkp, draw_hkp_simple, get_3d_depth_by_single_box, get_3d_depth_bybox, vis_keypoints

color = np.random.rand(32, 3)*255

# load config file##########################################################################
with open("../user/lib/setting_whole_trail.json") as f:
    config = json.load(f)
# gpu setting
gpu_setting = config["gpu_setting"]
############################################################################################

# det&track algorithm#######################################################################
my_trackModule = TrackWholeModule(**config["deepModel"]["deepDetection"]["structure"],
                             **gpu_setting,
                             **{"single_mode": True})
det_setting = config["deepModel"]["deepDetection"]["parameters"]
det_load_ret = my_trackModule.load_model(config["deepModel"]["deepDetection"]["model"])
############################################################################################

# 3Dkeypoint algorithm######################################################################
my_3dkp_model = Deep3DKPModel(**config["deepModel"]["deep3DKeypoint"]["structure"],
                    **gpu_setting)
tdkp_load_ret = my_3dkp_model.load_model(config["deepModel"]["deep3DKeypoint"]["model"])
############################################################################################

# filter trace##############################################################################
my_trace = TrackFilter(diff_thres=3.5)
############################################################################################
i = datetime.datetime.now()
video_name = "../user/output/hand_pull_push_compare_{}_{}_{}_{}.avi".format(i.year, i.month, i.day, i.hour)
size = (1920, 480)
fourcc = cv2.VideoWriter_fourcc(*'mjpg')
fps = 15
raw_video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
# load images###############################################################################
root_dir = r"D:\project\arch\data"
txt_list = os.path.join(root_dir, "list.txt")
with open(txt_list, "r") as f:
    images_ids = f.readlines()
############################################################################################
area1 = []
area2 = []
area3 = []
area4 = []
area5 = []
depth_scale = 0.0010000000474974513
for ids in images_ids:
    frame = cv2.imread(os.path.join(root_dir, ids.strip()+".jpg"))
    depth_image = np.load(os.path.join(root_dir, ids.strip()+".npy"))
    output = my_trace.get_smooth_output(my_trackModule, det_setting, frame)
    plt_figure = None
    root = False
    if len(output) != 0:
        bbox = np.array([out["body_box"] for out in output]).astype(np.int32)
        try:
            root_depth_list = get_3d_depth_bybox(depth_image, bbox, depth_scale, percent=20)
            kp_2d, kp_3d = my_3dkp_model.get_output(frame, bbox[0, :][None, :], root_depth_list)
            root = True
        except:
            root = False
        frame = vis_keypoints(frame, kp_2d[0], kp_thresh=0.02)
    for idx, out in enumerate(output):
        # print(out)
        # human_kp_n = my_kp.get_output(input_image, out['body_box'][None, :])
        # input_image = draw_hkp_simple(input_image, human_kp_n[0])
        c = color[out["id"]%18].tolist()
        cv2.rectangle(frame, (int(out['body_box'][0]), int(out['body_box'][1])),
                      (int(out['body_box'][2]), int(out['body_box'][3])), c, 2)
        if out['face_conf'] is not None:
            cv2.rectangle(frame, (int(out['face_box'][0]), int(out['face_box'][1])),
                          (int(out['face_box'][2]), int(out['face_box'][3])), c, 2)
            cv2.circle(frame, (int(out['face_lm'][0]), int(out['face_lm'][1])), 1, (0, 0, 255), 4)
            cv2.circle(frame, (int(out['face_lm'][2]), int(out['face_lm'][3])), 1, (0, 255, 255), 4)
            cv2.circle(frame, (int(out['face_lm'][4]), int(out['face_lm'][5])), 1, (255, 0, 255), 4)
            cv2.circle(frame, (int(out['face_lm'][6]), int(out['face_lm'][7])), 1, (0, 255, 0), 4)
            cv2.circle(frame, (int(out['face_lm'][8]), int(out['face_lm'][9])), 1, (255, 0, 0), 4)
        if out['lh_box'] is not None:
            cv2.rectangle(frame, (int(out['lh_box'][0]), int(out['lh_box'][1])),
                          (int(out['lh_box'][2]), int(out['lh_box'][3])), c, 2)
            cv2.circle(frame, (int(out['lh_lm'][0]), int(out['lh_lm'][1])), 1, (0, 0, 255), 4)
            cv2.circle(frame, (int(out['lh_lm'][2]), int(out['lh_lm'][3])), 1, (0, 255, 255), 4)
            cv2.circle(frame, (int(out['lh_lm'][4]), int(out['lh_lm'][5])), 1, (255, 0, 255), 4)
            cv2.circle(frame, (int(out['lh_lm'][6]), int(out['lh_lm'][7])), 1, (0, 255, 0), 4)
            cv2.putText(frame, "left", (int(out['lh_box'][0]), int(out['lh_box'][1])),
                        cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 0], 1)
        if out['rh_box'] is not None:
            cv2.rectangle(frame, (int(out['rh_box'][0]), int(out['rh_box'][1])),
                          (int(out['rh_box'][2]), int(out['rh_box'][3])), c, 2)
            cv2.circle(frame, (int(out['rh_lm'][0]), int(out['rh_lm'][1])), 1, (0, 0, 255), 4)
            cv2.circle(frame, (int(out['rh_lm'][2]), int(out['rh_lm'][3])), 1, (0, 255, 255), 4)
            cv2.circle(frame, (int(out['rh_lm'][4]), int(out['rh_lm'][5])), 1, (255, 0, 255), 4)
            cv2.circle(frame, (int(out['rh_lm'][6]), int(out['rh_lm'][7])), 1, (0, 255, 0), 4)
            cv2.putText(frame, "right", (int(out['rh_box'][0]), int(out['rh_box'][1])),
                        cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 255], 1)
            rb_center = (out['rh_box'][0]+(out['rh_box'][2]-out['rh_box'][0])/2.,
                         out['rh_box'][1]+(out['rh_box'][3]-out['rh_box'][1])/2.)
            # root_depth = get_3d_depth_by_single_box(depth_image, out['rh_box'], my_video_cap.depth_scale, percent=80)
            depth = depth_image[int(rb_center[1]), int(rb_center[0])]
            if depth != 0:
                area1.append(depth)
            root_depth30 = get_3d_depth_by_single_box(depth_image, out['rh_box'], depth_scale, percent=30)
            # area2.append(root_depth30)
            root_depth50 = get_3d_depth_by_single_box(depth_image, out['rh_box'], depth_scale, percent=50)
            if root:
                area3.append(root_depth50)
                plt_figure = depth_image[int(out['rh_box'][1]):int(out['rh_box'][3]),
                                int(out['rh_box'][0]):int(out['rh_box'][2])]
                area4.append(kp_3d[0, 16, 2])
                kp = kp_2d[0, 16].astype(int)
                depth_six = depth_image[kp[1], kp[0]]
                if depth_six != 0:
                    area5.append(depth_six)
    area1 = area1[-500:]
    # area2 = area2[-500:]
    area3 = area3[-500:]
    area4 = area4[-500:]
    area5 = area5[-500:]
    fig = plt.figure()
    plt.plot(area1, label='center')
    # plt.plot(area2, label='percent 30')
    plt.plot(area3, label='percent 50')
    plt.plot(area4, label='kps3d')
    plt.plot(area5, label='kps2d')
    plt.legend(loc='upper left')
    image = fig2data(fig)
    plt.close(fig)
    image_fig = np.zeros((480, 640, 3), dtype=np.uint8)
    if plt_figure is not None:
        fig = plt.figure()
        plt.imshow(plt_figure)
        image_fig = fig2data(fig)
        plt.close(fig)
    #     cv2.imshow("depth_image", image_fig)
    # cv2.imshow("win", frame)
    # cv2.imshow("area", image)
    cat_image = np.hstack((frame, image_fig, image))
    cv2.imshow("cat_image", cat_image)
    raw_video_writer.write(cat_image)
    cv2.waitKey(1)
    # print(1)