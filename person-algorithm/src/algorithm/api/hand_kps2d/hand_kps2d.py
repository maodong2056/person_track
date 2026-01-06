import numpy as np
from src.utils import load_model_setting
from src.utils import HandItem, HandPartName, KeyPointType
from src.algorithm.deep_model import get_model, MODEL_TASK
from src.algorithm.api.basic_task import BasicTask


class HandKps2D(BasicTask):
    def __init__(self, model_setting, hand_item=HandItem):
        super(HandKps2D, self).__init__()
        self.model_name, \
        self.model_dir, \
        self.structure, \
        self.para, self.gpu_set \
            = load_model_setting(model_setting)
        self.hand_item = hand_item
        self.model = get_model(MODEL_TASK.TASK_HAND2DKPS, self.model_name,
                               self.model_dir,
                               **self.structure, **self.gpu_set)
        self.data_type = KeyPointType.Kps2D
        # self.joints_name = self.para["joints_name"]
        # self.keypoint_limb = self.para["keypoint_limb"]
        # self.keypoint_color_map = np.array(self.para["keypoint_color_map"])
        # assert len(self.para["limb_color_index"]) == len(self.keypoint_limb)
        # assert len(self.para["keypoint_color_index"]) == len(self.joints_name)
        # self.limb_color = self.keypoint_color_map[self.para["limb_color_index"]]
        # self.keypoint_color = self.keypoint_color_map[self.para["keypoint_color_index"]]
        # self.kps_vis_thres = self.para["keypoint_vis_threshold"]


    def get_output(self, image, inputs, *args, **kwargs):
        model_input = self.__transfer_input(inputs)
        if len(inputs) != 0:
            outputs = self.model.get_output(image, model_input, **self.para)
        else:
            outputs = []
        result = self.__transfer_output(inputs, outputs)
        return result

    def __transfer_input(self, inputs, *args, **kwargs):
        model_input = []
        #此格式不同主要是由于该关键点模型不止需要ｂｏｘ还需要手掌的关键点来计算旋转角度. 正常只需要get_box #todo
        #检测和关键点模型的输入都是归一化值
        # for input in inputs:
            # detection_ = []
            # detection_.extend(input.get_box(HandPartName.hand_part))
            # detection_.extend(input.get_lm(HandPartName.hand_part).reshape(-1))
            # detection_.extend([input.get_conf(HandPartName.hand_part)])
            #
            # model_input.append(detection_)

        #新方案，直接改为统一调用roi作为输入
        for input in inputs:
            model_input.append(input.get_roi(HandPartName.hand_part))
        return np.array(model_input)

    def __transfer_output(self, hand_items, outputs, *args, **kwargs): #这个地方根据手掌关键点模型get_output来改正
        for hand_item, out in zip(hand_items, outputs):
            keypoint = out['hand_keypoint']
            hand_item.update_keypoint(HandPartName.hand_part, keypoint, self.data_type)
            hand_item.update_roi(HandPartName.hand_part, out['hand_roi']) #todo 对于跟踪，用关键点形成的框
            hand_item.update_trackState(out['hand_tracking'])
            # print(f'3hand_tracking1:{hand_item._hand_tracking_state}')
        return hand_items
