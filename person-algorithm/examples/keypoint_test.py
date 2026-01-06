import cv2
from src.video import Picture
from src.utils import PartName, KeyPointType
from src.algorithm.api import PersonKps2D
from src.algorithm.api import PersonDetection
from src.utils import draw_person_bbox, draw_keypoints, draw_keypoints_3d

pics_mode = True
my_pics = Picture(True, r"D:\share\dataset\test_dataset\keypoints_test_20220322\pics")
person_det = PersonDetection("user/settings/model/detection/body_detection/centernet_lite_8down_ratio.json")
person_kps2d = PersonKps2D("user/settings/model/keypoint/kps2d/siamdr_hm36m.json")

for idx, (img_name, pic) in enumerate(my_pics):
    output = person_det.get_output(pic)
    output = person_kps2d.get_output(pic, output)
    image = draw_person_bbox(pic, output, draw_face=False, draw_conf=True)
    image = draw_keypoints(image,
                           output,
                           PartName.body_part,
                           person_kps2d.keypoint_limb,
                           person_kps2d.limb_color,
                           person_kps2d.keypoint_color,
                           person_kps2d.kps_vis_thres)
    cv2.imshow("im", image)
    print(idx)
    # cv2.imwrite(img_name.replace("pics", "results"), image)
    cv2.waitKey(0)