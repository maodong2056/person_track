from src import RealSenceVideo
import os
import cv2
import numpy as np


# load images###############################################################################
fps = 15
my_video_cap = RealSenceVideo(video_width=1280, video_height=720, video_fps=fps)
ret = True
save_dir = r"D:\project\arch\realsense_data\data_cat2"
save_txt = os.path.join(save_dir, "list.txt")
# cv2.namedWindow('win', 0)
frame_id = 0
txt_list = []
############################################################################################
while ret:
    depth_image, frame = my_video_cap.capOneFrame()
    list_name = "{:0>6d}".format(frame_id)
    cv2.imwrite(os.path.join(save_dir, list_name+".jpg"), frame)
    np.save(os.path.join(save_dir, list_name+".npy"), depth_image)
    txt_list.append(list_name)
    cv2.imshow("win", frame)
    cv2.waitKey(1)
    with open(save_txt, "a+") as f:
        f.write(list_name + "\n")
    frame_id += 1

