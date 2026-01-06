import cv2
from src import Video
from src.utils import PartName, KeyPointType

from src.algorithm.api import PersonDetection
from src.algorithm.api import PersonFaceRecognition
from src.utils import draw_person_bbox

person_det = PersonDetection("user/settings/model/detection/body_detection/centernet_lite_8down_ratio.json")
# person_recog = PersonFaceRecognition('user/settings/model/recognition/face_recogniton/arcface_mobilefacenet.json')
person_recog = PersonFaceRecognition('user/settings/model/recognition/face_recogniton/arcface_mobilefacenet.json')
# create video fourcc#######################################################################
# input_video = "/home/xhzh/mystorage/share_with_vm/数据集/face/部门自己采集/camera_save_video_zhxh/zhxh_2.0m2022_03_01_16_49_25.mp4"
# camera_mode = False
input_video = r"D:\share\dataset\test_dataset\跟随视频\gj3.mp4"
# input_video = r"WIN_20220331_16_22_58_Pro.mp4"
camera_mode = True
fps = 25
size = (1280, 720)
# my_video = Video(camera_mode, input_video)
my_video = Video(camera_mode, 0, 1280, 720)
############################################################################################

ret = True
idx = 0
while ret:
    ret, input_image = my_video.capOneFrame()
    if not ret:
        break
    output = person_det.get_output(input_image)
    output = person_recog.get_output(input_image, output)
    print(output[0].get_feature(PartName.face_part))

    # features = [item.get_feature(PartName.face_part) for item in output]
    image = draw_person_bbox(input_image, output, draw_conf=True)
    cv2.imshow("im", image)
    cv2.waitKey(1)

    print(1)

