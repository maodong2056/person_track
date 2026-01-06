import cv2
from src import Video
from src.utils import HandPartName, KeyPointType
from src.algorithm.api import HandDetection, HandKps2D

from src.utils import draw_hand

hand_det = HandDetection('user/settings/model/detection/hand_detection/blazepalm_det.json')
hand_kps2d = HandKps2D('user/settings/model/keypoint/hand_kps2d/blazepalm_kps.json')
# create video fourcc#######################################################################
# input_video = "/home/xhzh/mystorage/share_with_vm/数据集/face/部门自己采集/camera_save_video_zhxh/zhxh_2.0m2022_03_01_16_49_25.mp4"
# input_video = r"D:\share\dataset\2.0_test_data\2.0_video\eg1.mp4"
input_video = 0
camera_mode = True
fps = 25
size = (1280, 720)
my_video = Video(camera_mode, input_video)
############################################################################################
TRACKING = False
output = []
ret = True
idx = 0
while ret:
    ret, input_image = my_video.capOneFrame()
    if not ret:
        break
    if TRACKING==False:
        output = hand_det.get_output(input_image)
    output = hand_kps2d.get_output(input_image, output)

    img, tracking = draw_hand(input_image,
                        output,
                        # draw_hand_box = False,
                        draw_hand_box = True,
                        draw_hand_kps = True,
                        draw_hand_roi = True,
                        is_norm = True,
                        tracking = TRACKING,
                        )
    #手部跟踪启用，只有在画面中某个手关键点低于阈值时才会重启手掌检测。
    # 所以在跟踪状态下，画面中新出现的手不会被检测到
    # if tracking:
    #     TRACKING = True
    # else:
    #     TRACKING = False


    cv2.imshow("im", img)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    # print(1)

