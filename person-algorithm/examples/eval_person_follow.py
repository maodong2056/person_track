import cv2
from src import Video
from src.utils import PartName, KeyPointType
from src.algorithm.api import PersonDetection
from src.algorithm.api import PersonMutiTrack
from src.algorithm.api import PersonFollow, PersonTrackerState
from src.algorithm.api.person_follow.person_item_track import PersonFollowItem
from src.utils import draw_person_bbox, draw_keypoints, draw_keypoints_3d
from  src.utils.iou_cal import iou_cal_xyxy
from glob import glob
import os
import pandas as pd
import numpy as np
# create video fourcc#######################################################################
choose_init_id_number = 1   # 选择跟踪第几个人

class trackState:
    Lost = 0
    error = 1
    Tracking = 2

root_dir = r"D:\reid_data\reid_project_video/"
family_name_path  =root_dir + 'family_001/'
img_per_path=None
gt_path = os.path.join(family_name_path, "label_my.csv")
label_data = pd.read_csv(gt_path)
frame_index = 0
test_all_video_path  = label_data.groupby(label_data['label_file_path'])

file = open('eval_person_follow.txt','w')
video_name = "eg1.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 29
# size = (640, 480)
size = (1280, 720)

out = cv2.VideoWriter(video_name, fourcc, fps, size)


def conform_track_id_img(image, person_all_box,person_all_gt_id,target_id):
    image_w = image.shape[1]
    image_h = image.shape[0]

    center_dist = []
    conform_init_index = -1


    for box_index, result in enumerate(person_all_box):
        bbox = result  # x1 y1 x2 y2
        # 归一化0-1范围计算距离中心点距离
        box_center_w = ((bbox[0] + bbox[2]) / 2) / image_w
        box_center_h = ((bbox[1] + bbox[3]) / 2) / image_h

        box_center_dist = abs(0.5 - box_center_h) + abs(0.5 - box_center_w)
        center_dist.append(box_center_dist)

    # 去最小值索引

    min_dist_index = center_dist.index(min(center_dist))

    best_init_box = person_all_box[min_dist_index]

    # 选择初始化ID特征，选择中间位置，不选择靠近边上位置

    center_x = (best_init_box[0] + best_init_box[2]) / 2.0
    ratio_w = center_x / image_w

    center_y = (best_init_box[1] + best_init_box[3]) / 2.0
    ratio_h = center_y / image_h

    box_h = best_init_box[3] - best_init_box[1]
    ratio_box_h_img = box_h / image_h
    current_target_id = person_all_gt_id[min_dist_index]
    iou_state = iou_cal_xyxy(best_init_box.reshape(-1, 4), person_all_box).reshape(-1).tolist()
    no_occ = True
    for iou in iou_state:
        if iou >0.2 and iou<1 :
            no_occ = False


    # 判断目标处于中间位置，作为初始化目标并提取特征
    if ratio_w > 0.2 and ratio_w < 0.8 and ratio_h > 0.01 and ratio_h < 0.9 and ratio_box_h_img >= 0.001 and current_target_id==target_id and no_occ :
        conform_init_index = min_dist_index
    return conform_init_index

frame_id = 1

frame_id_list = []
gt_track_id_list = []
tracking_state_list = []
precision_width_list = []
tracking_state_pre_csv_list=[]



for per_video in test_all_video_path:

    person_det = PersonDetection("user/settings/model/detection/body_detection/centernet_lite_8down_ratio.json")
    person_track = PersonMutiTrack("user/settings/model/track/deep_sort.json")
    person_follow = PersonFollow("user/settings/model/recognition/body_recognition/pcb_reid_mobilev2.json",
                                 person_item=PersonFollowItem)
    img_all_path = per_video[1].groupby(per_video[1]['frame_id'])
    # frame_index = 0
    all_object_person_num = 1
    track_gt_person_id = -1
    successfuly_track_person_id = 0

    per_tracklet_successful_count = 0
    per_tracklet_gt_count =0
    successful_rate = 0

    pre_frame_object = 0
    Tracklet_number_count = 0
    pre_tracklet = -1

    all_tracklet_success_rate = []

    precision_frame_count = 0
    precion_frame_tp_count = 0
    precisin_all_frame = 0

    pre_object_track_state = -1
    all_success_iou = []

    per_tracklet_correct_count = 0
    per_tracklet_precison_all_count = 0
    all_tracklet_count_precison = []

    conforme_state = -1
    # frame_id = 1

    precision_width = 0

    tracking_state = PersonTrackerState.Uninit
    for i,img_path in enumerate(img_all_path):
        # print(img_path)
        person_all_box = []
        person_all_gt_id = []
        current_frame_object = 0
        current_object_track_state = 0
        # if img_path[1]['type'][-1] !='video':
        #     continue
        for k,img in enumerate( img_path[1]['frame_id']):
            img_per_path = img
            if img_per_path.split('/')[1] == 'Init':
                continue
            person_bxc =img_path[1]['bxc'][frame_index]
            person_byc = img_path[1]['byc'][frame_index]
            person_bw = img_path[1]['bw'][frame_index]
            person_bh = img_path[1]['bh'][frame_index]
            person_gt_id = img_path[1]['person_num'][frame_index]
            #convert xyxy
            person_x1 = person_bxc - person_bw / 2.0
            person_y1 = person_byc - person_bh / 2.0
            person_x2 = person_bxc + person_bw / 2.0
            person_y2 = person_byc + person_bh / 2.0
            person_all_box.append([person_x1,person_y1,person_x2,person_y2])
            person_all_gt_id.append(person_gt_id)


            # img_per_path = img['frame_id'][int(frame_index)]
            frame_index +=1
        if img_per_path.split('/')[1] =='Init':
            continue
        person_all_box = np.array(person_all_box).reshape(-1, 4)

        img_path_frame = root_dir + img_per_path
        input_image = cv2.imread(img_path_frame)  #

        # conforme_state = conform_track_id_img(input_image, person_all_box, person_all_gt_id, choose_init_id_number)
        if conforme_state ==-1:
            conforme_state = conform_track_id_img(input_image, person_all_box, person_all_gt_id, choose_init_id_number)
            person_all_box=[]
            person_all_gt_id=[]
            continue



        if track_gt_person_id not in person_all_gt_id:
            pre_frame_object = 0

            gt_track_id = -1
        else:
            all_object_person_num +=1
            gt_track_id = track_gt_person_id
            # pre_frame_object = 1
            current_frame_object = 1
        if track_gt_person_id in person_all_gt_id:
            pre_tracklet = Tracklet_number_count
            if pre_frame_object ==0 and current_frame_object ==1:
                pre_frame_object =1
                Tracklet_number_count+=1
                # print('==========',pre_frame_object,current_frame_object)

            if pre_tracklet == Tracklet_number_count:
                per_tracklet_gt_count += 1

        # print('per_tracklet_gt_count',per_tracklet_gt_count)
        tracking_state_pre = -1  #未给出预测结果 -1
        output = person_det.get_output(input_image)
        output = person_track.get_output(input_image, output)
        if tracking_state == PersonTrackerState.Uninit:
            if len(output)!=0:
                track_ret = person_follow.init_track(input_image, person_items=output)
                if track_ret:
                    tracking_state = person_follow.get_state()
                    tracking_target = person_follow.get_tracking_person()
                    tracking_box = tracking_target.get_box(PartName.body_part)
                    #计算跟踪框与gt的iou距离，初始化id
                    track_gt_person_iou = iou_cal_xyxy(tracking_box.reshape(-1,4),person_all_box).reshape(-1).tolist()
                    # print(track_gt_person_iou)
                    max_iou = max(track_gt_person_iou)
                    max_iou_index = track_gt_person_iou.index(max(track_gt_person_iou))
                    track_gt_person_id = person_all_gt_id[max_iou_index]
                    input_image = draw_person_bbox(input_image, [tracking_target], draw_tracking=True)
                    input_image = draw_person_bbox(input_image, output)
                    tracking_state_pre = trackState.Tracking
        else:
            tracking_state = person_follow.update_track(input_image, output)
            tracking_target = person_follow.get_tracking_person()
            input_image = draw_person_bbox(input_image, [tracking_target], draw_tracking=True)
            input_image = draw_person_bbox(input_image, output)
        precision_width = 0
        if tracking_state == PersonTrackerState.Tracking:
            tracking_box = tracking_target.get_box(PartName.body_part)
            # print(tracking_box)
            track_person_iou = iou_cal_xyxy(tracking_box.reshape(-1, 4), person_all_box).reshape(-1).tolist()
            # print(track_gt_person_iou)
            max_iou_track = max(track_person_iou)
            max_iou_index_track = track_person_iou.index(max(track_person_iou))
            track_person_id = person_all_gt_id[max_iou_index_track]

            #


            #计算整个视频的准确率
            precision_frame_count +=1
            if max_iou_track >= 0.45 and track_person_id == track_gt_person_id :
                gt_object_width = person_all_box[max_iou_index_track][2] - person_all_box[max_iou_index_track][0]
                track_box_width = tracking_box[2]-tracking_box[0]
                precision_width =  1.0 - abs(gt_object_width-track_box_width) / gt_object_width
                # if precision_width < 0.9 :
                #     print('precision_width==',precision_width)
                #     print('===============')
                all_success_iou.append(max_iou_track)
                precion_frame_tp_count +=1
                precisin_all_frame =precion_frame_tp_count/precision_frame_count


                tracking_state_pre =  trackState.Tracking
            if max_iou_track >= 0.45 and track_person_id != track_gt_person_id:
                tracking_state_pre = trackState.error
                max_iou_track = 0
                all_success_iou.append(max_iou_track)

            if max_iou_track <0.45:
                 max_iou_track = 0
                 all_success_iou.append(max_iou_track)
                 tracking_state_pre = trackState.error

                #计算整个视频成功率
            if track_person_id == track_gt_person_id and max_iou_track>=0.5:
                # print('=====successfully track per tracklet====', track_person_id)
                successfuly_track_person_id +=1
                successful_rate = successfuly_track_person_id / all_object_person_num

                # print('all frame successful rate ',successful_rate)
                #计算每段轨迹成功率
                if  pre_tracklet == Tracklet_number_count:

                    if track_person_id == track_gt_person_id and max_iou_track >= 0.5:
                        per_tracklet_successful_count += 1
        if tracking_state == PersonTrackerState.Lost:
            max_iou_track = -1
            all_success_iou.append(max_iou_track)
            tracking_state_pre = trackState.Lost



        if pre_tracklet != Tracklet_number_count and pre_tracklet > 0:
            per_tracklet_successful_rate = per_tracklet_successful_count / (per_tracklet_gt_count + 1e-9)
            # print('=======per tracklet  successful rate========== ', per_tracklet_successful_rate)
            all_tracklet_success_rate.append(per_tracklet_successful_rate)
            per_tracklet_gt_count = 0
            per_tracklet_successful_count = 0

        tracking_state_pre_csv = tracking_state_pre
        # print('frame_id',frame_id)
        # print('gt_track_id',gt_track_id)
        # print('tracking_state_pre_csv',tracking_state_pre_csv)
        # print('precision_width',precision_width)

        frame_id_list.append(frame_id)
        gt_track_id_list.append(gt_track_id)
        tracking_state_pre_csv_list.append(tracking_state_pre_csv)
        precision_width_list.append(precision_width)

        frame_id+=1
        input_image = cv2.resize(input_image, (1280, 720))
        out.write(input_image)
        cv2.imshow("im", input_image)
        cv2.waitKey(1)
    if img_per_path.split('/')[1] == 'Init':
        continue
    success_tracklet_05 = 0
    success_tracklet_06 =0
    success_tracklet_07 = 0
    success_tracklet_08 =0
    success_tracklet_09 = 0
    if len(all_tracklet_success_rate) == 0:
        per_tracklet_successful_rate = successful_rate
        all_tracklet_success_rate.append(per_tracklet_successful_rate)

    print('all_tracklet_success_rate',all_tracklet_success_rate)
    for iou in all_tracklet_success_rate:
        if iou > 0.5:
            success_tracklet_05 +=1
        if iou > 0.6:
            success_tracklet_06 +=1
        if iou > 0.7:
            success_tracklet_07 +=1
        if iou > 0.8:
            success_tracklet_08 +=1
        if iou > 0.9:
            success_tracklet_09 +=1
    success_05_rate = success_tracklet_05 / (len(all_tracklet_success_rate) + 1e-9)
    success_06_rate = success_tracklet_06 / (len(all_tracklet_success_rate) + 1e-9)
    success_07_rate = success_tracklet_07 / (len(all_tracklet_success_rate) + 1e-9)
    success_08_rate = success_tracklet_08 / (len(all_tracklet_success_rate) + 1e-9)
    success_09_rate = success_tracklet_09 / (len(all_tracklet_success_rate) + 1e-9)

    print('----success_05_rate-----',success_05_rate)
    print('----success_06_rate-----', success_06_rate)
    print('----success_07_rate-----', success_07_rate)
    print('----success_08_rate-----', success_08_rate)
    print('----success_09_rate-----', success_09_rate)
    file.write('----success_05_rate-----'+str(success_05_rate))
    file.write('\n')
    file.write('----success_06_rate-----'+str(success_06_rate))
    file.write('\n')
    file.write('----success_07_rate-----'+str(success_07_rate))
    file.write('\n')
    file.write('----success_08_rate-----'+str(success_08_rate))
    file.write('\n')
    file.write('----success_09_rate-----'+str(success_09_rate))
    file.write('\n')


    print('=====all frame successful rate===== ', successful_rate)
    file.write('=====all frame successful rate====='+str(successful_rate))
    file.write('\n')
    print('====precisin_all_frame======', precisin_all_frame)
    file.write('=====precisin_all_frame====='+str(precisin_all_frame))
    file.write('\n')


    #计算每个轨迹的precision
    all_success_iou.append(-1)#[0.6,0.55,0,-1,-1,-1,0.54,0,-1]
    for prec_iou in all_success_iou:

        if prec_iou != -1:
            if prec_iou >= 0.5:
                per_tracklet_correct_count += 1
            per_tracklet_precison_all_count += 1
        else:

            if pre_object_track_state >-1 :
                if per_tracklet_precison_all_count >=4:#过滤一帧或者两帧的轨迹
                    per_tracklet_precesion = per_tracklet_correct_count / (per_tracklet_precison_all_count + 1e-9)

                    per_tracklet_correct_count = 0
                    per_tracklet_precison_all_count = 0

                    all_tracklet_count_precison.append(per_tracklet_precesion)
        pre_object_track_state = prec_iou

    mean_05_precision = np.mean(all_tracklet_count_precison)
    print('=====mean_05_presion=====',mean_05_precision)
    file.write('=====mean_05_presion=====' + str(mean_05_precision))
    file.write('\n')

dataframe = pd.DataFrame({'frame_id':frame_id_list,'gt_track_id_list':gt_track_id_list,'tracking_state_pre_csv_list':tracking_state_pre_csv_list,'precision_width_list':precision_width_list})
csv_name = 'pre_csv/' + family_name_path.split('/')[1]+'_'+str(choose_init_id_number)+ '.csv'

dataframe.to_csv(csv_name,index = False,sep=',')






