import numpy as np
from src.utils.iou_cal import iou_cal_xyxy
from .person_item_track import PersonFollowItem, PersonReidIdx
from src.utils import PersonItem, PartName, TrackingState
from src.utils import load_model_setting
from src.algorithm.deep_model import get_model, MODEL_TASK
from src.algorithm.api.basic_task import BasicTask
from .RRClassifier import RRClassifier,RRClassifierWithStrategy
from sklearn import  manifold
import matplotlib.pyplot as plt
def draw_tsne(x_feature,y_label):
    # x_feature shape = [n,2048]
    # y_label = [n]

    tsne = manifold.TSNE(n_components=2,init='pca',random_state=501)
    x_tsne = tsne.fit_transform(x_feature)
    print('x_feature',x_feature.shape)
    print('x_tsne',x_tsne.shape)
    #print('y_label',y_label.shape)
    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    X_norm = (x_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1],  str(y_label[i]),color=plt.cm.Set1(y_label[i]),
                 fontdict={'weight': 'bold', 'size': 18})
    plt.xticks([])
    plt.yticks([])
    plt.show()

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
        self.glob_feat_score = 0.35 #self.para["glob_feat_score"]

        self.id_switch_score = 0.45
        self.id_match_score = 0.6
        self.conform_id_num = 2
        self.conform_id_count = 0
        self.same_person_id = -1


        self.gate_threshold = 0.24
        self.target_id = []
        self.history_id = []
        self.lost_frame_count = 0
        self.max_lost_frame = self.para["max_lost_frame"]
        self.save_feat_score = self.para["save_feat_score"]
        self.detindex = -1
        self.Init_featuer_active = False

        alpha = 1.0
        samples = 64
        threshold = 0.02
        self.classifier = RRClassifierWithStrategy(alpha, samples, threshold)

        # self.classifier = RRClassifier(alpha=1, sampleNumbers=64)


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

            confirm_state = True
            target_person = person_items[marked_idx]
            self.detindex = marked_idx
            self.__mark_track(target_person)
        else:
            lost_state = True
            self.__mark_lost()
        # 遮挡态判定
        #丢失态预测逐帧
        if self.classifier.train_finished == True:
            if confirm_state:
                lost_state = False
                target_box = person_items[marked_idx].get_box(PartName.body_part)

                target_feat  = self.model.get_output(image, [target_box])
                target_feat = target_feat[0]

                target_score = self.classifier.predict(target_feat)
                print('===target===score ', target_score)

                if target_score <= self.id_switch_score:
                    confirm_state = False
                    lost_state = True
                    self.__mark_lost()
            if lost_state == True and confirm_state == False:
            #丢失状态全局搜索 正常跟踪状态只判断自身得分
                confirm_state = False
                print('start predict')
                predict_confidences = []
                predict_boxes  = [p.get_box(PartName.body_part) for p in person_items]
                predict_features = self.model.get_output(image, predict_boxes)
                for pre_feat in predict_features:
                    predict_score = self.classifier.predict(pre_feat)
                    print('all_box_predict_conf:',predict_score)
                    predict_confidences.append(predict_score)
                #求最大值和索引
                if len(predict_confidences)>0:
                    self.detindex = predict_confidences.index(max(predict_confidences))
                    max_predict_confidence = max(predict_confidences)
                    print('==predict_score==',max_predict_confidence)

                    if max_predict_confidence >= self.id_match_score:

                        #多帧确认 判断同一个id
                        candidate_person_id = person_items[self.detindex].get_track_id()
                        if candidate_person_id == self.same_person_id:
                            self.conform_id_count +=1
                        else:
                            self.conform_id_count =1
                        self.same_person_id = candidate_person_id

                        if self.conform_id_count >= self.conform_id_num:
                            self.conform_id_count = 0
                            confirm_state = True
                            lost_state = False
                            target_person = person_items[self.detindex]
                            self.__mark_track(target_person)

        if confirm_state:
            all_box = [p.get_box(PartName.body_part) for p in person_items]
            # print(all_box)
            features = []
            feature_labels = []
            features_all = self.model.get_output(image, all_box)

            for i in range(0,len(all_box)):
                features.append([features_all[i],np.array(3),all_box[i].reshape(-1,4)])
            for i in range(0,len(person_items)):
                if i == marked_idx:
                    feature_labels.append(1)
                else:
                    feature_labels.append(0)
            # self.classifier.update_cache(features_all, feature_labels)
            self.classifier.update_cache(features,feature_labels)

            if len(self.classifier.posTrainSet)>=10 and len(self.classifier.negLabels)>=1:
                self.classifier.update_classifier()


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
        # self.tracking_person.get_track_id()
        self.tracking_person.update_trackid(-1)

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

