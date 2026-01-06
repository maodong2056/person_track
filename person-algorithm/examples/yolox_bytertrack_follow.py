
import cv2

from src import Video
from src.algorithm.api import PersonDetection
from src.algorithm.api import PersonFollow, PersonTrackerState
from src.algorithm.api import PersonMutiTrack
from src.algorithm.api.person_follow.person_item_track import PersonFollowItem

import  numpy as np

from src.utils import draw_person_bbox, draw_keypoints, draw_keypoints_3d

person_det = PersonDetection("user/settings/model/detection/body_detection/yolox.json")
#
person_follow = PersonFollow("user/settings/model/recognition/body_recognition/pcb_reid_mobilev2.json",
                             person_item=PersonFollowItem)
# byte_tracker = BYTETracker( frame_rate=15)
person_track = PersonMutiTrack("user/settings/model/track/btye_track.json")
# create video fourcc#######################################################################

#input_video = r"D:\share\dataset\test_dataset\跟随视频\gj3.mp4"
input_video = r"D:\person-algorithm-package\2.0_video\dibao\1.mp4"
camera_mode = False
fps = 25
size = (1024, 700)
my_video = Video(camera_mode,input_video,video_width=1280, video_height=960)
# my_video = Video(camera_mode, 0,video_width=1920, video_height=960)
############################################################################################
video_name = "test/1.avi"
fourcc = cv2.VideoWriter_fourcc(*'mjpg')

raw_video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
tracking_state = PersonTrackerState.Uninit

K = np.zeros((3, 3))

K[0, 0] = 324.579
K[1, 1] = 324.082
K[0, 2] = 661.791
K[1, 2] = 479.482
K[2, 2] = 1


# K = np.array((324.579, 0, 661.791,0, 324.082,479.482 )).reshape(2,3)

distCoeffs = np.float32([0.50028, 0.580403, -0.000276571,
                                            8.50996e-05, 0.0205898,0.539282,
                                                                0.621697, 0.0760686])

ret = True
while ret:
    ret, input_image = my_video.capOneFrame()

    img_undistored = cv2.undistort(input_image, K, distCoeffs)
    cv2.imwrite('11.png', img_undistored)
    # input_image = input_image[160:160+700, 160:160+1024]


    # if not ret:
    #     break
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
        input_image = draw_person_bbox(input_image, [tracking_target], draw_tracking=True)
        input_image = draw_person_bbox(input_image, output)
    # input_image = draw_person_bbox(input_image, output)
    input_image = cv2.resize(input_image, (1024, 700))
    raw_video_writer.write(input_image)
    cv2.imshow("im", input_image)
    cv2.waitKey(1)
    # print(1)

