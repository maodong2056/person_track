import numpy as np
from src.utils import load_model_setting
from src.utils import PersonItem, PartName, KeyPointType
from src.algorithm.deep_model import get_model, MODEL_TASK
from src.algorithm.api.basic_task import BasicTask


class PersonKps2D(BasicTask):
    def __init__(self, model_setting, person_item=PersonItem):
        super(PersonKps2D, self).__init__()
        self.model_name, \
        self.model_dir, \
        self.structure, \
        self.para, self.gpu_set \
            = load_model_setting(model_setting)
        self.person_item = person_item
        self.model = get_model(MODEL_TASK.TASK_2DKPS, self.model_name,
                               self.model_dir,
                               **self.structure, **self.gpu_set)
        self.data_type = KeyPointType.Kps2D
        self.joints_name = self.para["joints_name"]
        self.keypoint_limb = self.para["keypoint_limb"]
        self.keypoint_color_map = np.array(self.para["keypoint_color_map"])
        assert len(self.para["limb_color_index"]) == len(self.keypoint_limb)
        assert len(self.para["keypoint_color_index"]) == len(self.joints_name)
        self.limb_color = self.keypoint_color_map[self.para["limb_color_index"]]
        self.keypoint_color = self.keypoint_color_map[self.para["keypoint_color_index"]]
        self.kps_vis_thres = self.para["keypoint_vis_threshold"]


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
        for input in inputs:
            model_input.append(input.get_box(PartName.body_part))
        return np.array(model_input)

    def __transfer_output(self, person_items, outputs, *args, **kwargs):
        for person_item, out in zip(person_items, outputs):
            person_item.update_keypoint(PartName.body_part, out, self.data_type)
        return person_items
