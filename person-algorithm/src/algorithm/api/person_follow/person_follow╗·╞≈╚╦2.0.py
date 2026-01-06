import numpy as np
from src.utils.iou_cal import iou_cal_xyxy
from .person_item_track import PersonFollowItem, PersonReidIdx
from src.utils import PersonItem, PartName, TrackingState
from src.utils import load_model_setting
from src.algorithm.deep_model import get_model, MODEL_TASK
from src.algorithm.api.basic_task import BasicTask

class PersonTrackerState:
    Uninit = 1
    Tracking = 2
    Lost = 3

class PersonFollow(BasicTask):
    def __init__(self, model_setting, person_item=PersonFollowItem):
        super(PersonFollow, self).__init__()
        self.model_name, \
        self.model_dir, \
        self.structure, \
        self.para, self.gpu_set \
            = load_model_setting(model_setting)
        self.person_item = person_item
        self.model = get_model(MODEL_TASK.TASK_REID, self.model_name,
                               self.model_dir,
                               **self.structure, **self.gpu_set)
        self.same_init_num_count = 0
        self.last_init_id = -1
        self.conform_int_max_number = 3#self.para["conform_int_max_number"]
        self.max_feature_bank = self.para["max_feature_bank"]
        self.reid_dim = self.para["reid_dim"]
        self.occlude_iou_range = self.para["occlude_iou_range"]
        self.tracking_person = None
        self.track_count_num = 0
        self.max_save_frame = self.para["max_save_frame"]
        self.max_save_id = self.para["max_save_id"]
        self.state = PersonTrackerState.Uninit
        self.occ_feat_score = self.para["occ_feat_score"]
        self.glob_feat_score = self.para["glob_feat_score"]
        self.gate_threshold = 0.24
        self.target_id = []
        self.history_id = []
        self.lost_frame_count = 0
        self.max_lost_frame = self.para["max_lost_frame"]
        self.save_feat_score = self.para["save_feat_score"]
        self.detindex = -1
        self.Init_featuer_active = False


    # def get_output(self, image, outputs, *args, **kwargs):
    #     print(1)

    def conform_init_box(self, image, person_items):
        image_w = image.shape[1]
        image_h = image.shape[0]

        center_dist = []
        conform_init_index = -1

        for box_index, result in enumerate(person_items):
            bbox = result.get_box(PartName.body_part) # x1 y1 x2 y2
            # 归一化0-1范围计算距离中心点距离
            box_center_w = ((bbox[0] + bbox[2]) / 2) / image_w
            box_center_h = ((bbox[1] + bbox[3]) / 2) / image_h

            box_center_dist = abs(0.5 - box_center_h) + abs(0.5 - box_center_w)
            center_dist.append(box_center_dist)

        # 去最小值索引

        min_dist_index = center_dist.index(min(center_dist))

        best_init_box = person_items[min_dist_index].get_box(PartName.body_part)

        # 选择初始化ID特征，选择中间位置，不选择靠近边上位置

        center_x = (best_init_box[0] + best_init_box[2]) / 2.0
        ratio_w = center_x / image_w

        center_y = (best_init_box[1] + best_init_box[3]) / 2.0
        ratio_h = center_y / image_h

        box_h = best_init_box[3] - best_init_box[1]
        ratio_box_h_img = box_h / image_h

        # 判断目标处于中间位置，作为初始化目标并提取特征
        if ratio_w > 0.2 and ratio_w < 0.8 and ratio_h > 0.01 and ratio_h < 0.9 and ratio_box_h_img >= 0.001 :
            conform_init_index = min_dist_index
        return conform_init_index

    def init_track(self, image, person_items):

        conforme_init_box_init = self.conform_init_box(image, person_items)
        if conforme_init_box_init > -1:
            person_item = person_items[conforme_init_box_init]
            if person_item.get_track_id() == self.last_init_id:
                self.same_init_num_count += 1
            else:
                self.same_init_num_count = 0
            self.last_init_id = person_item.get_track_id()
            if self.same_init_num_count > self.conform_int_max_number:
                self.state = PersonTrackerState.Tracking
                # self.tracking_person = self.person_item(person_item, self.max_feature_bank, self.reid_dim)
                self.__update_tracking_person(person_item)
                init_box = self.tracking_person.get_box(PartName.body_part)
                #沁宝最近0.9236 正常0.909  # 2.0待定，save_feat_score待定
                body_emb = self.__get_feaure(image, [self.tracking_person])
                self.__update_feaure(self.tracking_person, body_emb, PersonReidIdx.init_feature)
                # if init_box[3] / image.shape[0] < 0.925: # 沁宝跟随
                if init_box[3] / image.shape[0] < 2: # 2.0跟随
                    self.Init_featuer_active = True
                    # body_emb = self.__get_feaure(image, [self.tracking_person])
                    # self.__update_feaure(self.tracking_person, body_emb, PersonReidIdx.init_feature)

                return True
        return False
    def init_track_setinit_box(self, image, person_item):

        # conforme_init_box_init = self.conform_init_box(image, person_items)
        # if conforme_init_box_init > -1:
        person_item = person_item



        self.state = PersonTrackerState.Tracking
        # self.tracking_person = self.person_item(person_item, self.max_feature_bank, self.reid_dim)
        self.__update_tracking_person(person_item)
        init_box = self.tracking_person.get_box(PartName.body_part)
        #沁宝最近0.9236 正常0.909  # 2.0待定，save_feat_score待定
        body_emb = self.__get_feaure(image, [self.tracking_person])
        self.__update_feaure(self.tracking_person, body_emb, PersonReidIdx.init_feature)
        # if init_box[3] / image.shape[0] < 0.925:
        # if init_box[3] / image.shape[0] < 2:
        self.Init_featuer_active = True
            # body_emb = self.__get_feaure(image, [self.tracking_person])
            # self.__update_feaure(self.tracking_person, body_emb, PersonReidIdx.init_feature)

        return True

    def Set_Templete_update_Frequence(self,feature_update_frame = 20):
        #设置更新模板频率
        self.max_save_frame = feature_update_frame
        return  True

    def Set_reid_global_gate_thresh(self,gate_thres = 0.18):
        #设置更新模板频率
        self.glob_feat_score = gate_thres
        self.gate_threshold = self.glob_feat_score+0.02


        return  True

    def reset_Tracker(self):
        self.state = PersonTrackerState.Uninit
        if(self.tracking_person != None):
            self.tracking_person = None
        self.target_id = []
        self.history_id = []
        self.same_init_num_count = 0
        return  self.state



    def _cosin_metric(self, x1, x2):

        dot_result = (x1[..., None] * x2.transpose(1, 0)[None, ...]).sum(axis=1)
        normal_dot = np.linalg.norm(x1, 2, 1, True) * np.linalg.norm(x2, 2, 1, True).transpose(1, 0)
        return 1. - dot_result / normal_dot

    def update_track(self, image, person_items):
        assert self.tracking_person is not None
        target_person = None
        marked_idx = -1
        lost_state = True # 丢失态
        occl_state = False # 遮挡态
        occl_idx = []
        confirm_state = False # 确认跟踪态
        embedding_p = []
        person_items = [p for p in person_items if p.get_track_id() is not None]

        # 丢失态判定
        person_ids = [p.get_track_id() for p in person_items]

        if self.tracking_person.get_track_id() in person_ids:
            lost_state = False
            marked_idx = person_ids.index(self.tracking_person.get_track_id())
            self.detindex= marked_idx
        else:
            lost_state = True
        # 遮挡态判定
        if not lost_state:
            if len(person_items) == 1:# 只有一个目标，则确认跟踪
                occl_state = False
            else: #如果有多个目标，则需要判断iou重合度
                marked_box = [person_items[marked_idx].get_box(PartName.body_part)]
                inputs_box = [p.get_box(PartName.body_part) for idx, p in enumerate(person_items)]
                ious = iou_cal_xyxy(np.array(marked_box), np.array(inputs_box))
                iou_mask = np.logical_and(ious >= self.occlude_iou_range[0], ious <= self.occlude_iou_range[1])
                if iou_mask.any():
                    occl_state = True
                    occl_idx += [idx for idx, m in enumerate(iou_mask[0]) if m == True  and idx != marked_idx]
                else:
                    occl_state = False

        # 丢失态匹配逻辑
        if lost_state:
            # 判定是否要进行全局搜索
            if self.lost_frame_count > self.max_lost_frame:
                embedding_p += [p for idx, p in enumerate(person_items)]
                #self.det_indexes += [idx for idx, p in enumerate(person_items)]
                score = self.glob_feat_score
            else:
                embedding_p += [p for idx, p in enumerate(person_items) if p.get_track_id() not in self.history_id]
                #self.det_indexes += [idx for idx, p in enumerate(person_items)]
                score =  self.gate_threshold

            print('==============match score======',score)
            det_dist_score, det_box_index, det_feat = self.__match(image, embedding_p)
            if det_dist_score <= score and det_box_index != -1:
                target_person = embedding_p[det_box_index]
                confirm_state = True
                self.detindex = det_box_index
                self.__mark_track(target_person)
                self.history_id += [p.get_track_id() for idx, p in enumerate(embedding_p) if idx!=det_box_index
                                    and p.get_track_id() not in self.history_id]
            else:
                confirm_state = False
                self.__mark_lost()
        else:
            if occl_state:
                det_dist_score, det_box_index, det_feat = self.__match(image, [person_items[marked_idx]])
                conform_dist = det_dist_score
                if det_dist_score <= self.occ_feat_score and det_box_index != -1:
                    target_person = person_items[marked_idx]
                    confirm_state = True
                    self.detindex = marked_idx
                    self.__mark_track(target_person)
                    self.history_id += [p.get_track_id() for idx, p in enumerate(person_items) if idx != marked_idx
                                        and p.get_track_id() not in self.history_id]
                else:
                    embedding_p += [person_items[idx] for idx in occl_idx]

                    det_dist_score, det_box_index, det_feat = self.__match(image, embedding_p)
                    if det_dist_score <= self.occ_feat_score and det_box_index != -1:
                        target_person_idx = occl_idx[det_box_index]
                        target_person = person_items[target_person_idx]
                        confirm_state = True
                        self.detindex = target_person_idx
                        self.__mark_track(target_person)
                        self.history_id += [p.get_track_id() for idx, p in enumerate(person_items) if idx != target_person_idx
                                            and p.get_track_id() not in self.history_id]
                    else:
                        # if det_dist_score > 0.3://多目标对了，单目标没有匹配上需要优化  使用save_feat_score标记丢失
                        # if det_dist_score > self.save_feat_score:
                        #保存模板条件严格； 判断遮挡后的丢失后的iou放宽 occlude_iou_range 严格  mark丢失宽松
                        confirm_state = False
                        self.__mark_lost()
                    # if confirm_state == False:
                    #     print(occl_idx)
                    #     confirm_state = True
                    #     target_person = person_items[marked_idx]
                    #     self.detindex = marked_idx
                    #     self.__mark_track(target_person)
            else:
                confirm_state = True
                target_person = person_items[marked_idx]
                self.detindex = marked_idx
                self.__mark_track(target_person)

        if confirm_state:
            if target_person.get_track_id() not in self.target_id:
                self.target_id.append(target_person.get_track_id())
            if not occl_state and (self.track_count_num % self.max_save_frame)==0:
                # 存入时判断下是否相似度足够高
                bbox = target_person.get_box(PartName.body_part)
                if self.Init_featuer_active == False:
                    if bbox[3] / image.shape[0] < 2:#沁宝最近距离50cm露出脚开始保持第一帧模板
                    # if bbox[3] / image.shape[0] < 2:  # 2.0 不设置
                        body_emb = self.__get_feaure(image, [target_person])
                        self.__update_feaure(self.tracking_person, body_emb, PersonReidIdx.init_feature)
                        self.Init_featuer_active =True
                else:
                # emb = self.__get_feaure(image, [target_person])
                    det_dist_score, det_box_index, det_feat = self.__match(image, [target_person])
                    if det_dist_score <= self.save_feat_score and det_box_index != -1:
                        ratio_h = (bbox[3]-bbox[1]) /image.shape[0]
                        # print('=====ratio_h=======', ratio_h)
                        if bbox[0] / image.shape[1] >= 0.2 and bbox[2] / image.shape[1] <= 0.8 and  \
                                bbox[1] / image.shape[0] > 0.03 and bbox[3] / image.shape[0] <0.97 and ratio_h >=0.3:
                        # if bbox[0] / image.shape[1] >= 0.2 and bbox[2] / image.shape[1] <= 0.8 :
                        #     self.__update_feaure(self.tracking_person, det_feat, PersonReidIdx.middle_feature)
                            if target_person._face_box is not None:
                                self.__update_feaure(self.tracking_person, det_feat, PersonReidIdx.middle_feature)
                            else:
                                self.__update_feaure(self.tracking_person, det_feat, self.max_feature_bank -1)

                        else:
                            if bbox[0] / image.shape[1] >= 0.07 and bbox[2] / image.shape[1] <= 0.93:
                                self.__update_feaure(self.tracking_person, det_feat, PersonReidIdx.edge_feature)

                        if bbox[1] / image.shape[0] >= 0.02 and bbox[3] / image.shape[0] >= 0.98:#增加近距离两个模板
                            print ('===jindist=====',bbox[3] / image.shape[0])
                            if target_person._face_box is not None:
                                self.__update_feaure(self.tracking_person, det_feat, 3)
                            else:
                                self.__update_feaure(self.tracking_person, det_feat, 4)
                        # if bbox[0] / image.shape[1] >= 0.2 and bbox[2] / image.shape[1] <= 0.8 and  bbox[1] / image.shape[0] > 0.08 and bbox[3] / image.shape[0] <0.92:
                        # # if bbox[0] / image.shape[1] >= 0.2 and bbox[2] / image.shape[1] <= 0.8 :
                        #     self.__update_feaure(self.tracking_person, det_feat, PersonReidIdx.middle_feature)
                            # self.__update_feaure(self.tracking_person, det_feat, PersonReidIdx.edge_feature)
                    else:
                        # 如果相似度不够高，则为丢失
                        confirm_state = False
                        self.__mark_lost()
                        self.tracking_person.update_trackid(-1)
            if confirm_state:
                for item in person_items:
                    item_id = item.get_track_id()
                    if item_id != target_person.get_track_id() and item_id not in self.history_id:
                        self.history_id.append(item_id)

        self.target_id = self.target_id[-self.max_save_id:]
        self.history_id = self.history_id[-self.max_save_id:]
        # if self.state ==2:
        #     a=person_items[self.detindex].get_box(PartName.body_part)
        #     b=self.tracking_person.get_box(PartName.body_part)
        #     print('1111111111',a)
        #     print('2222222222',b)
        #     print('==========detindex',self.detindex)
        #     if a[0]!=b[0]:
        #         print('==========detindex',self.detindex)
        return self.state

    def __mark_track(self, person_item):
        self.state = PersonTrackerState.Tracking
        self.__update_tracking_person(person_item)
        if person_item.get_track_id() not in self.target_id:
            self.target_id.append(person_item.get_track_id())
        self.track_count_num += 1
        self.lost_frame_count = 0
        self.tracking_person.update_trackState(TrackingState.Tracking)


    def __mark_lost(self):
        self.state = PersonTrackerState.Lost
        self.track_count_num = 0
        self.lost_frame_count += 1
        self.tracking_person.update_trackState(TrackingState.UnTracking)
        # self.tracking_person.update_trackid(-1)

    # def __

    def __update_tracking_person(self, person_item):
        if self.tracking_person is None:
            self.tracking_person = self.person_item(person_item, self.max_feature_bank, self.reid_dim)
        else:
            self.tracking_person.update(person_item)

    def get_state(self):
        return self.state

    def get_tracking_person(self):
        return self.tracking_person

    def __match(self, image, inputs):
        if self.state == PersonTrackerState.Uninit:
            return False
        else:
            target_feature_bank = self.tracking_person.get_reid_feature()
        boxes = [inp.get_box(PartName.body_part) for inp in inputs]
        if len(boxes) != 0:
            inputs_feature_bank = self.model.get_output(image, boxes)
            feat_similarities = self._cosin_metric(target_feature_bank, inputs_feature_bank)
            feat_similarities = feat_similarities.min(axis=0)
            det_dist_score, det_box_index = feat_similarities.min(axis=0), \
                                            feat_similarities.argmin(axis=0)
            ret_feat = inputs_feature_bank[det_box_index]
        else:
            # inputs_feature_bank = np.zeros((0, self.reid_dim))
            det_dist_score, det_box_index = -1, -1
            ret_feat = np.zeros((1, self.reid_dim)) - 1

        print('=====feat_consin_score=======',det_dist_score)
        return det_dist_score, det_box_index, ret_feat


    def __get_feaure(self, image, inputs):
        bbox = [inp.get_box(PartName.body_part) for inp in inputs]
        features = self.model.get_output(image, bbox)
        return features

    def __update_feaure(self, input, feature, idx):
        input.update_reid_feature(feature, idx)

