
import cv2
import numpy as np
from loguru import logger
import time
import  os
from glob import glob
from src import Video
from src.algorithm.api import PersonDetection

from src.utils import draw_person_bbox, draw_keypoints, draw_keypoints_3d

person_det = PersonDetection("user/settings/model/detection/body_detection/yolox.json")
#input_video = r"D:\share\dataset\test_dataset\跟随视频\gj3.mp4"
input_video = r"D:\person-algorithm-package\\2022-1-18/video/1.MOV"
camera_mode = False
fps = 25
size = (1280, 720)
my_video = Video(camera_mode, input_video)
############################################################################################


ret = True
while ret:
    ret, input_image = my_video.capOneFrame()
    if not ret:
        break

    output = person_det.get_output(input_image)

    input_image = draw_person_bbox(input_image, output)

    input_image = cv2.resize(input_image, (1280, 720))
    cv2.imshow("win", input_image)
    cv2.waitKey(1)



