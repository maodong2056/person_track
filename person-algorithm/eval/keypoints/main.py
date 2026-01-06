import os
import cv2
import json
import numpy as np
# from pycocotools.coco import COCO
from eval.keypoints.xml2json import get_xml_to_json
from eval.keypoints.detpic2json import detection_to_json
from src import DeepDetModel, DeepSiamDRKPModel
from eval.keypoints.eval_visable import MyKpsEval
from eval.keypoints.eval_visable_mpii import MyKpsEval as MyKpsEval_MPII

dataset_dir = "/raid/hzj/data/keypoint_project/Collection/20211025_rs"
save_dir = os.path.join(dataset_dir, "dataset.json")
save_result_dir = os.path.join(dataset_dir, "my_result.json")
xml_dir = os.path.join(dataset_dir, "4000xml")
images_dir = os.path.join(dataset_dir, "images")
vis_dir = os.path.join(dataset_dir, "vis_res")
if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)

#-------------------------------- model -------------------------------------------#
# with open("../user/lib/setting_trail.json") as f:
#     config = json.load(f)
# # gpu setting
# gpu_setting = config["gpu_setting"]
#
# my_detModule = DeepDetModel(**config["deepModel"]["deepDetection"]["structure"], **gpu_setting)
# det_setting = config["deepModel"]["deepDetection"]["parameters"]
# det_load_ret = my_detModule.load_model(config["deepModel"]["deepDetection"]["model"])
# print(det_load_ret)
# my_kpModule = DeepSiamDRKPModel(**config["deepModel"]["deepKeypoint"]["structure"], **gpu_setting)
# kp_setting = config["deepModel"]["deepKeypoint"]["parameters"]
# kp_load_ret = my_kpModule.load_model(config["deepModel"]["deepKeypoint"]["model"])
# print(kp_load_ret)
#----------------------------------------------------------------------------------#

#--------------------------- convert xml files to json-----------------------------#
# print("Start Convert xml to json!")
# try:
#     dic = get_xml_to_json(xml_dir, images_dir)
#     out_file = open(save_dir, "w")
#     json.dump(dic, out_file, indent=4)
#     out_file.close()
# except Exception as e:
#     print("Convert json fail!, {}".format(e))
#     exit()
# print("Convert json success!")
#----------------------------------------------------------------------------------#

#----------------------------- get all detection result----------------------------#
# print("Start Get All Detection Result!")
# coco = COCO(save_dir)
# image_list = coco.loadImgs(coco.getImgIds())
# try:
#     detection_to_json(images_dir, my_detModule, det_setting, my_kpModule, kp_setting, image_list, save_result_dir)
# except Exception as e:
#     print("Get All Detection Result fail!, {}".format(e))
#     exit()
# print("Get All Detection Result success!")
#----------------------------------------------------------------------------------#

#----------------------------- get coco result-------------------------------------#
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# coco = COCO(save_dir)
# coco_dt = coco.loadRes(save_result_dir)
# print("--------------------------CoCo Bbox!----------------------------")
# coco_eval = COCOeval(coco, coco_dt, 'bbox')
# coco_eval.params.useSegm = None
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()
# print("--------------------------CoCo Kps!-----------------------------")
# coco_eval = COCOeval(coco, coco_dt, 'keypoints')
# coco_eval.params.useSegm = None
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()
# print("---------------------CoCo Result Finish!------------------------")
#----------------------------------------------------------------------------------#

#------------------------ use self type eval && save images------------------------#
# my_eval = MyKpsEval(images_dir, vis_dir, save_dir, save_result_dir, write_image=False)
# box_ap, box_ar, kps_ap, kps_ar = my_eval.eval()
# print("-"*20, "box_ap:", box_ap, "-"*20)
# print("-"*20, "box_ar:", box_ar, "-"*20)
# print("-"*20, "kps_ap:", kps_ap, "-"*20)
# print("-"*20, "kps_ar:", kps_ar, "-"*20)
#----------------------------------------------------------------------------------#

#------------------------ use self type eval && save images------------------------#
my_eval = MyKpsEval_MPII(images_dir, vis_dir, save_dir, save_result_dir, write_image=True)
box_ap, box_ar, kps_ap, kps_ar = my_eval.eval()
print("-"*20, "box_ap:", box_ap, "-"*20)
print("-"*20, "box_ar:", box_ar, "-"*20)
print("-"*20, "kps_ap:", kps_ap, "-"*20)
print("-"*20, "kps_ar:", kps_ar, "-"*20)
#----------------------------------------------------------------------------------#