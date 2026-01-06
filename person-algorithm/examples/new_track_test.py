import cv2
from src import Video
import numpy as np
import datetime
from src.video import Picture
from src.utils import PartName, KeyPointType, crop_by_keypoints, transform_preds
from src.algorithm.api import PersonDetection
from src.algorithm.api import PersonMutiTrack
from src.algorithm.api import PersonKps3D
from src.algorithm.api import PersonFollow, PersonTrackerState
from src.algorithm.api.person_follow.person_item_track import PersonFollowItem
from src.utils import draw_person_bbox, draw_keypoints, draw_keypoints_3d
from src.utils import pixcel_camera_transfer


person_det = PersonDetection("user/settings/model/detection/body_detection/centernet_lite_8down_ratio.json")
person_kps3d = PersonKps3D("user/settings/model/keypoint/kps3d/posenet.json")
person_kps3d_skele = PersonKps3D("user/settings/model/keypoint/kps3d/graformer_kps17.json")
person_track = PersonMutiTrack("user/settings/model/track/deep_sort.json")
person_follow = PersonFollow("user/settings/model/recognition/body_recognition/pcb_reid_mobilev2.json",
                             person_item=PersonFollowItem)
# create video fourcc#######################################################################

#input_video = r"D:\share\dataset\test_dataset\跟随视频\gj3.mp4"
# input_video = r"D:\share\dataset\3d_kps_dataset\collections\temp2hzj\WIN_20210820_16_37_06_Pro.mp4"
input_video = r"WIN_20220331_16_22_58_Pro.mp4"
camera_mode = False
fps = 25
size = (1280, 720)
my_video = Video(camera_mode, input_video)
# my_video = Video(camera_mode, 0, video_width=1280, video_height=720)
############################################################################################

###############################video writer##########################################################
i = datetime.datetime.now()
tag = "cam2"
video_name = "user/output/3dpose_video_deebotcam_{}_{}_{}_{}_{}.avi".format(tag, i.year, i.month, i.day, i.hour)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15
size = (640, 480)
raw_video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
############################################################################################
input_image = r"D:\share\dataset\test_dataset\point2wash\realsense_data\data_point"
my_pic = Picture(True, input_image)


tracking_state = PersonTrackerState.Uninit
ret = True
kps = []



while ret:
    ret, input_image = my_video.capOneFrame()
    if not ret:
        break
# for _, input_image in my_pic:
    output = person_det.get_output(input_image)
    output = person_track.get_output(input_image, output)
    if tracking_state == PersonTrackerState.Uninit:
        if len(output)!=0:
            track_ret = person_follow.init_track(input_image, person_items=output)
            if track_ret:
                tracking_state = person_follow.get_state()
                tracking_target = person_follow.get_tracking_person()
                input_image = draw_person_bbox(input_image, [tracking_target], draw_tracking=True)
                input_image = draw_person_bbox(input_image, output)
    else:
        tracking_state = person_follow.update_track(input_image, output)
        tracking_target = person_follow.get_tracking_person()
        tracking_target_box = tracking_target.get_box(PartName.body_part)
        tracking_target_conf = tracking_target.get_conf(PartName.body_part)
        tracking_target_box[0] -= 100
        tracking_target_box[2] += 100
        tracking_target.update_box(PartName.body_part, tracking_target_box, tracking_target_conf)
        tracking_target = person_kps3d.get_output(input_image, [tracking_target])
        tracking_target = person_kps3d_skele.get_output(input_image, tracking_target)
        input_image = draw_person_bbox(input_image, tracking_target, draw_tracking=True)
        input_image = draw_person_bbox(input_image, output)
        if tracking_state == PersonTrackerState.Tracking:
            input_image = draw_keypoints(input_image,
                                   tracking_target,
                                   PartName.body_part,
                                   person_kps3d.keypoint_limb,
                                   person_kps3d.limb_color,
                                   person_kps3d.keypoint_color,
                                   person_kps3d.kps_vis_thres)
            depth_image = draw_keypoints_3d([tracking_target[0]],
                                            PartName.body_part,
                                            person_kps3d_skele.keypoint_limb,
                                            person_kps3d_skele.kps_vis_thres)
            keypoint = tracking_target[0].get_keypoint(PartName.body_part, KeyPointType.Kps2D)
            # keypoint[:, :2] = pixcel_camera_transfer(keypoint, f1, c1, f2, c2)
            # keypoint[:, :2] = trans_kps(keypoint[:, :2], (800, 800))
            # center = keypoint[0]
            # shift = np.array([500, 575, 1]) - center
            # keypoint[:, :2] = keypoint[:, :2] + shift[:2]
            kps.append(keypoint)
            cv2.imshow("de_im", depth_image)
            raw_video_writer.write(depth_image)


    input_image = cv2.resize(input_image, (1280, 720))
    cv2.imshow("im", input_image)
    cv2.waitKey(1)
    # print(1)

raw_video_writer.release()
np.save("test_kps.npy", np.array(kps)[:, :17, :])