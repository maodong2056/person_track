import cv2
from src import Video
from src.utils import PartName, KeyPointType
from src.algorithm.api import PersonDetection
from src.algorithm.api import PersonMutiTrack
from src.algorithm.api import PersonFollow, PersonTrackerState
from src.algorithm.api.person_follow.person_item_track import PersonFollowItem
from src.utils import draw_person_bbox, draw_keypoints, draw_keypoints_3d
from glob import glob
import os
from mmcv.cnn import get_model_complexity_info
# //mobilev2_reid_qinbao_2563_20220708.pth.tar
# //reid_model_merge_last_2430
person_det = PersonDetection("user/settings/model/detection/body_detection/centernet_lite_8down_ratio.json")
person_track = PersonMutiTrack("user/settings/model/track/deep_sort.json")
person_follow = PersonFollow("user/settings/model/recognition/body_recognition/pcb_reid_mobilev2.json",
                             person_item=PersonFollowItem)
# create video fourcc#######################################################################




def get_img(img_path = 't10img/',img_type ='*.jp*' ):
    images = glob(os.path.join(img_path,img_type))
    # images.sort()

    # images.sort(key=lambda x: int(x.split('/')[-1].split('\\')[-1].split('.')[0]))
    # images.sort(key=lambda x: int(x.split('\\')[-1].split('_')[-1].split('.')[0])) # haitao
    # images.sort(key=lambda x: int(x.split('\\')[-1].split('_')[-1].split('.')[0]))  # haitao
    images.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))  # haitao
    for img in images:
        yield img
#input_video = r"D:\share\dataset\test_dataset\跟随视频\gj3.mp4"
# input_video = r"D:\person-algorithm-package\\2.0_video/20201210160251-1-1-1.avi"
# camera_mode = False
input_video = r"D:\person-algorithm-package\2.0_video\qinbao\5.mp4"
# input_video = r"2022-1-18/video/3.MOV"as
camera_mode = False
fps = 25
size = (1024, 700)
my_video = Video(camera_mode, input_video)
############################################################################################
video_name = "0720_result.avi"
fourcc = cv2.VideoWriter_fourcc(*'mjpg')

raw_video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
tracking_state = PersonTrackerState.Uninit

ret = True
count = 1
# D:\person-algorithm-package\2022-1-18\20_video_0707\pic_0615
for file in get_img(r'D:\person-algorithm-package\2022-1-18\20_video_0707\pic_0907'):
# for file in get_img(r'D:\person-algorithm-package\2022-1-18\20220401\eg.1'):
# for file in get_img(r'D:\person-algorithm-package\2022-1-18\gj12/'):
# for file in get_img(r'D:\person-algorithm-package\2022-1-18\tracking_error0609_qinbao\33'):
# for file in get_img(r'D:\person-algorithm-package\2022-1-18\20220816wushibie\pic_5'):
#
# while ret:
#
#     ret, input_image = my_video.capOneFrame()
# # #     print(count)
    tracking_target = None
    print(count)
    count +=1
    input_image = cv2.imread(file)#
    # input_image = cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR)
# while ret:
#     ret, input_image = my_video.capOneFrame()
#     if not ret:
#         break
    output = person_det.get_output(input_image)
    output = person_track.get_output(input_image, output)

    if tracking_state == PersonTrackerState.Uninit:
        if len(output)!=0:
            track_ret = person_follow.init_track(input_image, person_items=output)
            # track_ret = person_follow.init_track_setinit_box(input_image, person_item=output[0])
            person_follow.Set_Templete_update_Frequence(20)
            if track_ret:
                # person_follow.Set_update_Frame(20)
                tracking_state = person_follow.get_state()
                tracking_target = person_follow.get_tracking_person()
                input_image = draw_person_bbox(input_image, [tracking_target], draw_tracking=True)
                input_image = draw_person_bbox(input_image, output)
    else:
        tracking_state = person_follow.update_track(input_image, output)
        tracking_target = person_follow.get_tracking_person()
        input_image = draw_person_bbox(input_image, [tracking_target], draw_tracking=True)
        input_image = draw_person_bbox(input_image, output)



    cv2.imshow("im", input_image)
    cv2.waitKey(1)
    # print(1)

