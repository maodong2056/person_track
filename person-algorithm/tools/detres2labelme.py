# @Time : 2021/3/17 14:57
# @Author : Altair.Huazj
# @File : demo.py
# @Software: PyCharm
import json
import cv2
import os
import shutil
import numpy as np
from src import DeepDetModel, DeepDetModelWhole, Video
# from src import DeepKPModel
from src import TrackWholeModule
from src import TrackFilter, LandmarkFilter
from src.utils import draw_hkp, draw_hkp_simple
from src import detection_to_labelme

# load config file##########################################################################
with open("../user/lib/setting_trail.json") as f:
    config = json.load(f)
# gpu setting
gpu_setting = config["gpu_setting"]
############################################################################################

# det&track algorithm#######################################################################
my_detModel = DeepDetModel(**config["deepModel"]["deepDetection"]["structure"],
                           **gpu_setting)
det_setting = config["deepModel"]["deepDetection"]["parameters"]
det_load_ret = my_detModel.load_model(config["deepModel"]["deepDetection"]["model"])
############################################################################################


def get_det_res(image):
    _, output = my_detModel.get_output(image, **det_setting)
    return output

def res2labelme(filepath, result):
    labelme_res = detection_to_labelme(filepath, result)
    return labelme_res

def get_labelme_result(filepath):
    image = cv2.imread(filepath)
    output = get_det_res(image)
    labelme_output = res2labelme(filepath, output)
    return labelme_output

if __name__ == '__main__':
    base_path = r"C:\Users\Altair\Desktop\renti"
    txt_file = os.path.join(base_path, "select.txt")
    with open(txt_file, "r") as f:
        files = f.readlines()
    for idx, file in enumerate(files):
        file_abs = os.path.join(base_path, file).strip()
        (filepath_root, tempfilename) = os.path.split(file_abs)
        new_file_root = filepath_root.replace("img", "img_select")
        if not os.path.exists(new_file_root):
            os.mkdir(new_file_root)
        new_file_path = os.path.join(new_file_root, tempfilename)
        shutil.copy(file_abs, new_file_path)
        json_file = new_file_path.replace(".jpg", ".json")
        res = get_labelme_result(file_abs)
        with open(json_file, "w") as f:
            json.dump(res, f)
        print("\rProcess {}/{}".format(idx+1, len(files)), end="")


