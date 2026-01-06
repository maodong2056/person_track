import cv2
import datetime
from src.video import Video, Picture
from src.utils import PartName, KeyPointType
from src.algorithm.api import PersonKps2D
from src.algorithm.api import PersonKps3D
from src.algorithm.api import PersonDetection
from src.algorithm.api import PersonMutiTrack
from src.algorithm.api import PersonFollow
from src.utils import draw_person_bbox, draw_keypoints, draw_keypoints_3d


person_det = PersonDetection("user/settings/model/detection/body_detection/centernet_lite_8down_ratio.json")
person_kps2d = PersonKps2D("user/settings/model/keypoint/kps2d/siamdr_hm36m.json")
person_track = PersonMutiTrack("user/settings/model/track/deep_sort.json")
person_kps3d = PersonKps3D("user/settings/model/keypoint/kps3d/posenet.json")
person_kps3d_skele = PersonKps3D("user/settings/model/keypoint/kps3d/graformer_kps17.json")
person_follow = PersonFollow("user/settings/model/recognition/body_recognition/pcb_reid_mobilev2.json")
# create video fourcc#######################################################################
# input_video = "WIN_20220331_16_22_58_Pro.mp4"
# input_video = "2.mp4"
input_video = r"D:\share\dataset\2.0_test_data\2.0_video\eg1.mp4"
camera_mode = False
fps = 25
size = (1280, 720)
my_video = Video(camera_mode, input_video)
my_pic = Picture(True, r"D:\share\dataset\2.0_test_data\2.0_video\pics")
############################################################################################

###############################video writer##########################################################
# i = datetime.datetime.now()
# tag = "model_last"
# video_name = "user/output/2dpose_video_2.0cam_{}_{}_{}_{}_{}.avi".format(tag, i.year, i.month, i.day, i.hour)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 15
# size = (1280, 720)
# raw_video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
############################################################################################
ret = True
idx = 0
while ret:
    ret, input_image = my_video.capOneFrame()
    if not ret:
        break
# for _, input_image in my_pic:
#     print(idx)
    output = person_det.get_output(input_image)
    output = person_track.get_output(input_image, output)
    # output = person_track.get_output(input_image, output)
    output = person_kps2d.get_output(input_image, output)
    # output = person_kps3d.get_output(input_image, output)
    # output = person_kps3d_skele.get_output(input_image, output)
    image = draw_person_bbox(input_image, output, draw_conf=True)
    idx += 1
    image = draw_keypoints(image,
                           output,
                           PartName.body_part,
                           person_kps2d.keypoint_limb,
                           person_kps2d.limb_color,
                           person_kps2d.keypoint_color,
                           person_kps2d.kps_vis_thres)
    # depth_image = draw_keypoints_3d([output[0]],
    #                                 PartName.body_part,
    #                                 person_kps3d_skele.keypoint_limb,
    #                                 person_kps3d_skele.kps_vis_thres)
    cv2.imshow("im", image)
    # raw_video_writer.write(image)
    # cv2.imshow("im_depth", depth_image)
    cv2.waitKey(1)
    # print(1)
# raw_video_writer.release()
