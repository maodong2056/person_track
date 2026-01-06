import cv2
from src import Video
from src.utils import PartName, KeyPointType

from src.utils import draw_person_bbox, draw_keypoints, draw_keypoints_3d
from glob import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import  time
def get_img(img_path = 't10img/',img_type ='*.jp*' ):
    images = glob(os.path.join(img_path,img_type))
    # images.sort()

    # images.sort(key=lambda x: int(x.split('/')[-1].split('\\')[-1].split('.')[0]))
    # images.sort(key=lambda x: int(x.split('\\')[-1].split('_')[-1].split('.')[0])) # haitao
    # images.sort(key=lambda x: int(x.split('\\')[-1].split('_')[-1].split('.')[0]))  # haitao
    # images.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))  # haitao
    for img in images:
        yield img



def judge_occ_level(input_image):

    occ0 = int(960 / 960 *384)
    occ1 = int(938 /960 *384)
    occ2 = int(390 / 960*384)
    occ3 = int(190 /960*384)
    occ4 = 0

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    edges = gray
    # print(gray.mean())
    # print(gray.var())
    # print(gray.std())
    gray_height =gray.shape[0]
    gray_weight = gray.shape[1]

    # edges = cv2.Canny(gray, 10, 90, apertureSize=3)
    # ret, edges = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    occlude_level_percent = 0


    occ1_edge_down = edges[occ1:gray_height, :]
    occ1_down_level = np.sum(occ1_edge_down < 20) / ((occ0 - occ1) * gray_weight)
    print('occ1_down_level',occ1_down_level)
    cv2.imshow("im", occ1_edge_down)
    cv2.waitKey(1)
    # occ_level1
    occ1_edge_top = edges[0:occ1,:]
    occ1_top_level = np.sum(occ1_edge_top < 20) / (occ1 * gray_weight)
    print('occ1_top_level', occ1_top_level)

    if occ1_down_level <= 0.92 or occ1_top_level <=0.1 :
        occlude_level_percent = 0
        print('无遮挡情况',occlude_level_percent)
        return occlude_level_percent

    else:
        print('===有遮挡====')

        # occ2_edge_top = edges[0:occ2, :]
        # occ2_top_contours, _ = cv2.findContours(occ2_edge_top, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # occ2_gray_top = gray[0:occ2, :]
        # occ2_top_mean = occ2_gray_top.mean()
        #
        # # print('top2 contors', len(occ2_top_contours), occ2_top_mean)
        #
        # occ2_edge_down = edges[occ2:gray_height, :]
        # occ2_down_contours, _ = cv2.findContours(occ2_edge_down, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # occ2_gray_down = gray[occ2:gray_height, :]
        # occ2_down_mean = occ2_gray_down.mean()
        #
        # # print('down2 contors', len(occ2_down_contours), occ2_down_mean)
        # percent_occ2 = len(occ2_down_contours)/(len(occ2_down_contours+occ2_top_contours)+1e-8)
        # # print('percent occ2',percent_occ2)
        #
        # # occ_level2
        # occ3_edge_top = edges[0:occ3, :]
        # occ3_top_contours, _ = cv2.findContours(occ3_edge_top, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # occ3_gray_top = gray[0:occ3, :]
        # occ3_top_mean = occ3_gray_top.mean()
        #
        # # print('top3 contors', len(occ3_top_contours), occ3_top_mean)
        #
        # occ3_edge_down = edges[occ3:gray_height, :]
        # occ3_down_contours, _ = cv2.findContours(occ3_edge_down, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # occ3_gray_down = gray[occ3:gray_height, :]
        # occ3_down_mean = occ3_gray_down.mean()
        #
        # # print('down3 contors', len(occ3_down_contours), occ3_down_mean)
        # percent_occ3 = len(occ3_down_contours)/(len(occ3_down_contours+occ3_top_contours)+1e-8)
        # # print('percent occ3',percent_occ3)
        #
        #
        #
        # # ret, binary = cv2.threshold(input_image,100,255, cv2.THRESH_BINARY_INV)
        # contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # gray_mean = gray.mean()
        # print("Total contour :", len(contours),gray.mean())

        #
        #
        # if occ1_down_mean<=5 or len(occ1_down_contours)<=3 or (percent_occ1 <=0.03 and occ1_down_mean <=15):
        #     occlude_level_percent = 1 - occ1 / gray_height
        #
        #     #第二种遮挡一半，会漏光，出现错误边缘，比率放宽
        #     if (percent_occ2 <=0.05 and occ2_down_mean <=25) or len(occ2_down_contours) <=5 or occ2_down_mean<=10:
        #         occlude_level_percent = 1 - occ2 / gray_height
        #
        #         #第三种情况，也会漏光，去左边交界最值
        #         if (percent_occ3<=0.05 and occ3_down_mean <=50) or len(occ3_down_contours)<=3 or  occ3_down_mean<=10:
        #             occlude_level_percent = 1 - occ3 / gray_height
        #
        #             if len(contours)<=8 or gray_mean <=10:
        #                 occlude_level_percent =  1- occ4/gray_height


    return occlude_level_percent

for file in get_img(r'D:\person-algorithm-package\2022-1-18\zhedangshuju\ch1/'):
# for file in get_img(r'D:\person-algorithm-package\2022-1-18\20_video_0707\new_camera_img\2'):
#

#
    print(file)
    input_image = cv2.imread(file)#
    input_image = cv2.resize(input_image, (512, 384))
    t1 = time.time()
    occ_level = judge_occ_level(input_image)
    t2 = time.time()
    print('time ',t2-t1)
    print('==============occlevel=========',occ_level)
    # cv2.imshow("im", input_image)
    # cv2.waitKey(0)
    # input_image = cv2.resize(input_image, (512, 384))



