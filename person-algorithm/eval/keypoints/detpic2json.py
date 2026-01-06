# from src import DeepSiamDRKPModel as DeepKPModel
# from src import DeepDetModel
import json
import numpy as np
import os
import cv2
# from pycocotools.coco import COCO
#-------------------------------- dirs -----------------------------------------------
# collect_dir = "/raid/hzj/data/keypoint_project/Collection/20210615"
# json_dir = os.path.join(collect_dir, "dataset.json")
# images_dir = os.path.join(collect_dir, "images")
# save_dir = os.path.join(collect_dir, "my_result.json")
# # labels = os.listdir(images_dir)
# coco = COCO(json_dir)
# image_list = coco.loadImgs(coco.getImgIds())
# dics = []
#-------------------------------------------------------------------------------------

#-------------------------------- model ----------------------------------------------
# with open("../user/lib/setting_trail.json") as f:
#     config = json.load(f)
# # gpu setting
# gpu_setting = config["gpu_setting"]
#
# my_detModule = DeepDetModel(**config["deepModel"]["deepDetection"]["structure"], **gpu_setting)
# det_setting = config["deepModel"]["deepDetection"]["parameters"]
# det_load_ret = my_detModule.load_model(config["deepModel"]["deepDetection"]["model"])
# print(det_load_ret)
# my_kpModule = DeepKPModel(**config["deepModel"]["deepKeypoint"]["structure"], **gpu_setting)
# kp_setting = config["deepModel"]["deepKeypoint"]["parameters"]
# kp_load_ret = my_kpModule.load_model(config["deepModel"]["deepKeypoint"]["model"])
# print(kp_load_ret)
#-------------------------------------------------------------------------------------


def detection_to_json(images_dir, my_detModule, det_setting, my_kpModule, kp_setting, image_list:list, save_dir):
    image_num = -1
    in_vis_thre = 0.01
    num_keypoints = 17
    dics = []

    for img_info in image_list:  # 20201126160137_000135.jpg
        image_path = os.path.join(images_dir, img_info["file_name"])
        img = cv2.imread(image_path)
        if img is None:
            print(image_path)
            continue

        image_num += 1
        if image_num % 100 == 0:
            print(image_num)

        image_shape, output = my_detModule.get_output(img, **det_setting)

        bboxes = []
        bboxes_scores = []
        for res in output:
            bboxes.append(res["body_box"])
            bboxes_scores.append(res["body_conf"])

        if bboxes != []:
            kps = my_kpModule.get_output(img, np.array(bboxes), **kp_setting)

            for i in range(len(bboxes)):
                ann_dic = {}
                ann_dic["category_id"] = 1
                ann_dic["image_id"] = img_info["id"]
                ann_dic["image_name"] = img_info["file_name"]

                bbox = bboxes[i]
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[2]
                ymax = bbox[3]
                w = xmax - xmin
                h = ymax - ymin
                ann_dic['bbox'] = [float(xmin), float(ymin), float(w), float(h)]

                box_score = bboxes_scores[i]
                kp = kps[i]
                kpt_score = 0
                valid_num = 0
                for n_jt in range(num_keypoints):
                    t_s = kp[n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                # new_score = kpt_score * box_score
                new_score = kpt_score

                human_kps = []
                for _kp in kp:
                    human_kps.append(float(_kp[0]))
                    human_kps.append(float(_kp[1]))
                    human_kps.append(float(_kp[2]))

                ann_dic['keypoints'] = human_kps
                ann_dic['score'] = float(new_score)
                dics.append(ann_dic)

    out_file = open(save_dir, "w")
    json.dump(dics, out_file, indent=4)
    out_file.close()