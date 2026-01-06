import cv2
import datetime
import numpy as np
from src.video import Video, Picture, RealsensePicture, VideoWriter
from src.utils import PartName, KeyPointType
from src.algorithm.api import PersonKps2D
from src.algorithm.api import PersonKps3D
from src.algorithm.api import PersonPointWash
from src.algorithm.api import PersonDetection
from src.algorithm.api import PersonMutiTrack
from src.algorithm.api import PersonFollow
from src.utils import draw_person_bbox, draw_keypoints, draw_keypoints_3d
from src.algorithm.deep_model.filter import FrameFilter

depth_scale = 0.0010000000474974513

person_det = PersonDetection("user/settings/model/detection/body_detection/centernet.json")
person_kps2d = PersonKps2D("user/settings/model/keypoint/kps2d/siamdr_hm36m.json")
# person_kps3d = PersonPointWash("user/settings/model/keypoint/kps3d/posenet.json", depth_scale=depth_scale)
person_kps3d_skele = PersonPointWash("user/settings/model/keypoint/kps3d/graformer_kps17_realsense.json", depth_scale=depth_scale)
my_pics = RealsensePicture(True, r"D:\share\dataset\realsense_dataset\point2wash\realsense_4010_2m_ml")
my_frame_filter = FrameFilter(filter_3dkeypoint=False, use_id_match=False)
result = []
# video_writer = VideoWriter("20220412_point2wash_animal_test", "filter_hm36m_mlunity+pointv2_cat2", size=(1280, 720))
# video_writer_d = VideoWriter("20220412_point2wash_animal_test", "filter_hm36m_mlunity+pointv2_cat2_depth", size=(640, 480))
for _, _, input_image, depth_image in my_pics:
    output = person_det.get_output(input_image)
    output = person_kps2d.get_output(input_image, output)
    # output = person_kps3d.get_output(input_image, output, depth_image)
    output = my_frame_filter.get_output(input_image, output)
    output = person_kps3d_skele.get_output(input_image, output, depth_image=None)

    image = draw_person_bbox(input_image, output, draw_conf=True)
    image = draw_keypoints(image,
                           output,
                           PartName.body_part,
                           person_kps3d_skele.keypoint_limb,
                           person_kps3d_skele.limb_color,
                           person_kps3d_skele.keypoint_color,
                           person_kps3d_skele.kps_vis_thres)
    depth_image = draw_keypoints_3d([output[0]],
                                    PartName.body_part,
                                    person_kps3d_skele.keypoint_limb,
                                    person_kps3d_skele.kps_vis_thres,
                                    z_depth=3000)

    cv2.imshow("im", image)
    cv2.imshow("depth", depth_image)
    radius_, moving_dist, moving_direction = output[0].calculate()
    print(radius_, moving_dist, moving_direction)
    result.append([radius_, moving_dist, moving_direction])
    # video_writer.write_frame(image)
    # video_writer_d.write_frame(depth_image)
    cv2.waitKey(1)
# video_writer.end_write()
# video_writer_d.end_write()
result = np.array(result)
print(result[2:120].mean(axis=0))
print(result[145:225].mean(axis=0))
print(result[249:338].mean(axis=0))
print(result[365:455].mean(axis=0))
print(result[477:557].mean(axis=0))
print(result[575:693].mean(axis=0))

