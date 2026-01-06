import numpy as np
from  numpy import  *
import  torch
from src.algorithm.deepModel.deepReidRec import ReidRecMobileModel
from src.algorithm.baseAlgorithm import iou_cal_xyxy

class PersonTrackerState:

    Uninit = 1
    Tracking = 2
    Lost = 3


class ObjectTracker(ReidRecMobileModel):
    """
    gate_threshold:目标丢失匹配阈值
    init_feature_budget：特征bank的长度，正常为2,即第一帧模板特征和最近前n帧的特征
    max_history_id_lenth:历史轨迹id的最大长度

    """

    def __init__(
            self,
            gate_threshold=0.23,
            init_feature_budget=3,
            sum_det_count=20,
            max_history_id_lenth=10,
            *args, **kwargs):

        self.gate_threshold = gate_threshold
        self.init_feature_budget = init_feature_budget
        self.max_lenth_track_id_history = max_history_id_lenth

        #list储存目标特征模板bank用于匹配
        self.init_feature_bank = []
        #初始化计数器，用于多少帧需要保存目标模板，提取特征保存在bank中
        self.count_num_det = 0
        #每20帧就提取一次目标模板保存在bank中
        self.sum_det_count = sum_det_count
        #初始化需要跟踪的id
        self.tracking_id = None
        #储存历史出现过的目标轨迹id
        self.track_id_history = []
        #储存历史出现的init_id,消失再出现的id也要匹配
        self.init_id_history = []
        #统计丢失的时间或者帧数
        self.lost_frame_number = 0
        #丢失多少帧以后调用全局搜索
        self.lost_max_number = 40
        self.state = PersonTrackerState.Uninit
        #多帧确定初始化
        self.same_init_num_count = 0
        self.first_init_track_id = -1
        self.conform_int_max_number = 3
        #初始化reid特征网络
        super(ObjectTracker, self).__init__(*args, **kwargs)


    def _cosin_metric(self, x1, x2):

        dot_result = (x1.unsqueeze(-1) * x2.transpose(1, 0).unsqueeze(0)).sum(dim=1)
        normal_dot = torch.norm(x1, 2, 1, True) * torch.norm(x2, 2, 1, True).transpose(1, 0)
        return 1. - dot_result / normal_dot

    def get_body_embedding(self, image, det_result):

        embs = self.get_output(image, det_result)
        return embs

    def compute_cosine_score_index(self, image, det_result, feat_bank):
        current_det_feat = self.get_body_embedding(image, det_result)

        det_feat_similarities = self._cosin_metric(current_det_feat, feat_bank)
        det_feat_similarities = det_feat_similarities.min(axis=1).values.view(-1, 1)

        det_dist_cost_metric = det_feat_similarities.cpu()

        best_det_dist_score_index = det_dist_cost_metric.min(axis=0)

        det_dist_score, det_box_index = best_det_dist_score_index.values, best_det_dist_score_index.indices

        return det_dist_score,det_box_index

    def init_tracker(self,image,bboxes,init_track_id):
        if init_track_id == self.first_init_track_id:
            self.same_init_num_count += 1
        else:
            self.same_init_num_count = 0

        self.first_init_track_id = init_track_id

        if self.same_init_num_count >= self.conform_int_max_number:
            self.init_feature_bank = self.get_body_embedding(image, [bboxes])
            self.tracking_id = init_track_id
            self.state = PersonTrackerState.Tracking
            self.current_track_box = bboxes


    def reset_tracker(self):
        # 重置跟踪器，删除所有存储信息，然后重新初始化跟踪框
        self.init_feature_bank = []
        self.track_id_history = []


        self.init_id_history = []

        self.state = PersonTrackerState.Uninit

    def update_track(self, image, bboxes, tracker_id):

        detections = bboxes#Tracker.preprocess_input(bboxes, class_ids, detection_scores)

        if (self.tracking_id in tracker_id) ==True:

            index = np.where(tracker_id == self.tracking_id)[0][0]
            self.current_track_box = detections[index]
            current_iou_state = iou_cal_xyxy(detections, self.current_track_box[None, :])

            if not np.logical_and(current_iou_state >= 0.005, current_iou_state <= 0.9).any():
                self.count_num_det += 1
                self.state = PersonTrackerState.Tracking

                if self.count_num_det == self.sum_det_count:
                    self.count_num_det = 0
                    current_body_emb = self.get_body_embedding(image, self.current_track_box[None, :])

                    if len(self.init_feature_bank) < self.init_feature_budget:

                        self.init_feature_bank = torch.cat([self.init_feature_bank, current_body_emb])
                    else:
                        if self.current_track_box[0] / image.shape[1] >= 0.03 and self.current_track_box[2] /image.shape[1] <= 0.97:

                            self.init_feature_bank[-1] = current_body_emb.view(-1)

                        if self.current_track_box[0]/image.shape[1] >=0.2 and self.current_track_box[2]/image.shape[1] <=0.8:
                            self.init_feature_bank[-2] = current_body_emb.view(-1)

                            self.init_feature_bank[-1] = current_body_emb.view(-1)


            #else:
            if np.logical_and(current_iou_state >= 0.15, current_iou_state <= 0.9).any():

                cos_score_occ, cos_index_occ = self.compute_cosine_score_index(image, [self.current_track_box],
                                                                               self.init_feature_bank)

                print('current_occ  cosine score ', cos_score_occ)

                if cos_score_occ > 0.2:  # 表示当前帧与模板bank中相似度不够，不是同一个人，需要重新匹配周围的框，计算与当前帧重叠的框
                    # 选择重叠的候选框作为候选目标框，iou大于0.1小于0.9
                    candidate_iou_state = np.where(np.logical_and(current_iou_state >= 0.15, current_iou_state <= 0.9))[0]

                    occ_detections = detections[candidate_iou_state]
                    occ_track_id = tracker_id[candidate_iou_state]
                    occ_dist_score, occ_box_index = self.compute_cosine_score_index(image,
                                                                                    occ_detections,
                                                                                    self.init_feature_bank)
                    if occ_dist_score <= 0.2:
                        self.tracking_id = occ_track_id[occ_box_index]
                        self.current_track_box = occ_detections[occ_box_index]
                        self.state = PersonTrackerState.Tracking
                    else:
                        print('person_tracker lost')
                        self.state = PersonTrackerState.Lost
                else:
                    #iou>0.3 丢到 hist track
                    #print('hist track')
                    self.state = PersonTrackerState.Tracking


            if self.state==PersonTrackerState.Tracking:
                self.lost_frame_number = 0
                if self.tracking_id not in self.init_id_history:
                    self.init_id_history.append(self.tracking_id)

                    # 如果历史轨迹超过十个，则保存最近10个的轨迹，之前的忘记
                    if len(self.init_id_history) >= self.max_lenth_track_id_history:
                        self.init_id_history = self.init_id_history[-self.max_lenth_track_id_history:]

                for current_id in tracker_id:
                    if current_id!=self.tracking_id and current_id not in self.track_id_history:
                        self.track_id_history.append(current_id)
                    if len(self.track_id_history) >= self.max_lenth_track_id_history:
                        self.track_id_history = self.track_id_history[-self.max_lenth_track_id_history:]

        #用于寻找丢失目标
        if (self.tracking_id in tracker_id) == False  or self.state ==PersonTrackerState.Lost:

            #print('lost')
            self.lost_frame_number +=1
            self.state = PersonTrackerState.Lost

            #候选匹配框与候选ID
            candidate_box = []
            candidata_track_id = []

            #将初始化id历史框排除再其他历史id轨迹,
            self.track_id_history = [ i for i in self.track_id_history if i not in self.init_id_history]

            for index,current_id in enumerate(tracker_id):

                if current_id not in self.track_id_history:
                    candidate_box.append(detections[index])
                    candidata_track_id.append(current_id)

            if len(candidate_box) > 0:
                candidate_box = np.array(candidate_box)
                candidata_track_id = np.array(candidata_track_id)

                dist_score, box_index = self.compute_cosine_score_index(image, candidate_box, self.init_feature_bank)

                #print('lost dist score ',dist_score)

                if dist_score < self.gate_threshold:

                    self.lost_frame_number = 0
                    self.tracking_id = candidata_track_id[box_index]
                    self.count_num_det = 0
                    self.current_track_box = candidate_box[box_index]
                    self.state = PersonTrackerState.Tracking
                else:
                    self.state = PersonTrackerState.Lost
                    #过于不相似，可以直接不匹配了？
                    # if dist_score > 0.5:
                    #     print('lost dist score ', dist_score)
                    #     new_appear_id  = candidata_track_id[box_index]
                    #     if  candidata_track_id[box_index] not in self.track_id_history:
                    #         self.track_id_history.append(new_appear_id)


                #
            #全局搜索,条件严格一点.一轮搜索以后,还是没有匹配到则进行全局搜索
            if self.lost_frame_number > self.lost_max_number:

                #self.init_feature_bank = self.init_feature_bank[0:1, :]
                global_dist_score, global_box_index = self.compute_cosine_score_index(image, detections, self.init_feature_bank)

                print('global_dist_score//////////',global_dist_score)

                if global_dist_score < 0.18:
                    self.lost_frame_number = 0
                    self.tracking_id = tracker_id[global_box_index]

                    self.count_num_det = 0

                    self.current_track_box = detections[global_box_index]
                    self.state = PersonTrackerState.Tracking

        return self.state



    def get_result(self):

        return  self.current_track_box#,self.tracking_id


