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
# person_det = PersonDetection("user/settings/model/detection/body_detection/centernet_lite_8down_ratio.json")
person_det = PersonDetection("user/settings/model/detection/body_detection/centernet_lite_8down_ratio_old.json")
person_track = PersonMutiTrack("user/settings/model/track/deep_sort.json")
person_follow = PersonFollow("user/settings/model/recognition/body_recognition/pcb_reid_mobilev2.json",
                             person_item=PersonFollowItem)
# create video fourcc#######################################################################




def get_img(img_path = 't10img/',img_type ='*.jp*' ):
    images = glob(os.path.join(img_path,img_type))
    # images.sort()

    # images.sort(key=lambda x: int(x.split('/')[-1].split('\\')[-1].split('.')[0]))
    images.sort(key=lambda x: int(x.split('\\')[-1].split('_')[-1].split('.')[0])) # haitao
    # images.sort(key=lambda x: int(x.split('\\')[-1].split('_')[-1].split('.')[0]))  # haitao
    # images.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))  # haitao
    for img in images:
        yield img
#input_video = r"D:\share\dataset\test_dataset\跟随视频\gj3.mp4"
# input_video = r"D:\person-algorithm-package\\2.0_video/20201210160251-1-1-1.avi"
# camera_mode = False
input_video = r"D:\person-algorithm-package\2.0_video\qinbao\init.mp4"
# input_video = r"D:\person-algorithm-package\2.0_video\4.mp4"
# input_video = r"2022-1-18/video/3.MOV"as
camera_mode = False
fps = 15
size = (1280, 960)
my_video = Video(camera_mode,input_video,video_width=1280, video_height=960)
############################################################################################
video_name = "test/1.avi"
fourcc = cv2.VideoWriter_fourcc(*'mjpg')

raw_video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
tracking_state = PersonTrackerState.Uninit

ret = True
count = 1
# D:\person-algorithm-package\2022-1-18\20_video_0707\pic_0615
# for file in get_img(r'C:\Users\yangkang\Desktop\guang'):
# for file in get_img(r'D:\person-algorithm-package\2022-1-18\20220401\eg.1'):
# for file in get_img(r'D:\person-algorithm-package\2022-1-18\gj12'):
# for file in get_img(r'D:\person-algorithm-package\2022-1-18\20_video_0707\new_camera_img\2023_02_10_15_01_52_619234'):
# for file in get_img(r'D:\person-algorithm-package\2022-1-18\qinbao_1280_jpg'):
#
while ret:
#
    ret, input_image = my_video.capOneFrame()


    # input_image = input_image[160:, 160:]


# # # #     print(count)
#     input_image = cv2.resize(input_image, (1280, 720))

    # raw_video_writer.write(input_image)
    tracking_target = None
    print(count)
    count +=1
    # input_image = cv2.imread(file)#
    print(input_image.shape)
    # input_image = cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR)

    # cv2.imwrite(r'D:\person-algorithm-package\2022-1-18\20_video_0707\jiao/'+str(count)+'.jpg',input_image)
# while ret:
#     ret, input_image = my_video.capOneFrame()
#     if not ret:
#         break
    output = person_det.get_output(input_image)
    input_image = draw_person_bbox(input_image, output)
    output = person_track.get_output(input_image, output)


    # if count ==45 or count ==83:
    #     tracking_state = person_follow.reset_Tracker()

    if tracking_state == PersonTrackerState.Uninit:
        if len(output)!=0:
            track_ret = person_follow.init_track(input_image, person_items=output)
            # track_ret = person_follow.init_track_setinit_box(input_image, person_item=output[1])
            # person_follow.Set_Templete_update_Frequence(15)
            # person_follow.Set_reid_global_gate_thresh(0.182)
            if track_ret:
                # person_follow.Set_update_Frame(20)
                tracking_state = person_follow.get_state()
                tracking_target = person_follow.get_tracking_person()
                input_image = draw_person_bbox(input_image, [tracking_target], draw_tracking=True, draw_conf=True)
                input_image = draw_person_bbox(input_image, output, draw_conf=True)
    else:
        tracking_state = person_follow.update_track(input_image, output)
        tracking_target = person_follow.get_tracking_person()
        input_image = draw_person_bbox(input_image, [tracking_target], draw_tracking=True, draw_conf=True)
        input_image = draw_person_bbox(input_image, output, draw_conf=True)


    txt_name = '0720_txt/'
    new_file = txt_name + '0720_txt' + '.txt'
    f = open(new_file ,'a+')
    if tracking_target is not  None:
         res = tracking_target.get_box(PartName.body_part).tolist()
         for i in res:
            f.write(str(i) + ' ')
         f.write('\n')
    else:
        f.write('-1')
        f.write('\n')
    # f.close()
    cv2.putText(input_image,str(count), (20,20), cv2.FONT_HERSHEY_SIMPLEX,
    0.7,(255,255,255), 1, cv2.LINE_AA)
    input_image = cv2.resize(input_image, (1280, 960))
    raw_video_writer.write(input_image)
    cv2.imshow("im", input_image)
    cv2.waitKey(1)
    # print(1)
# f.close()



