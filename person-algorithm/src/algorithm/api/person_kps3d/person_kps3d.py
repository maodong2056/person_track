import numpy as np
from src.utils import cam2pixel
from src.utils import load_model_setting
from src.utils import PersonItem, PartName, KeyPointType
from src.algorithm.deep_model import get_model, MODEL_TASK
from src.algorithm.baseAlgorithm import get_3d_depth_bybox, eval_rgb_depth_by_boxes
from src.algorithm.api.basic_task import BasicTask


class PersonKps3D(BasicTask):
    def __init__(self, model_setting, person_item=PersonItem, depth_scale=None):
        super(PersonKps3D, self).__init__()
        self.model_name, \
        self.model_dir, \
        self.structure, \
        self.para, self.gpu_set \
            = load_model_setting(model_setting)
        self.person_item = person_item
        self.model = get_model(MODEL_TASK.TASK_3DKPS, self.model_name,
                               self.model_dir,
                               **self.structure, **self.gpu_set)
        self.data_type = KeyPointType.Kps3D
        # self.focal = self.structure["focal"]
        # self.princpt = self.structure["princpt"]
        self.depth_scale = depth_scale
        self.input_type = self.structure["type"]
        self.percent = self.structure["percent"]
        self.joints_name = self.para["joints_name"]
        self.keypoint_limb = self.para["keypoint_limb"]
        self.keypoint_color_map = np.array(self.para["keypoint_color_map"])
        self.focal = self.structure["focal_cam1"] if "focal_cam1" in self.structure.keys() else self.structure["focal"]
        self.princpt = self.structure["princpt_cam1"] if "princpt_cam1" in self.structure.keys() else self.structure["princpt"]
        assert len(self.para["limb_color_index"]) == len(self.keypoint_limb)
        assert len(self.para["keypoint_color_index"]) == len(self.joints_name)
        self.limb_color = self.keypoint_color_map[self.para["limb_color_index"]]
        self.keypoint_color = self.keypoint_color_map[self.para["keypoint_color_index"]]
        self.kps_vis_thres = self.para["keypoint_vis_threshold"]

    def get_output(self, image, inputs, depth_image=None, *args, **kwargs):
        model_input, root_depth = self.__transfer_input(image, inputs, depth_image)
        if len(model_input) != 0:
            outputs = self.model.get_output(image, model_input, root_depth, **self.para)
        else:
            outputs = []
        result = self.__transfer_output(inputs, outputs)
        return result

    def __transfer_input(self, image, inputs, depth_image, *args, **kwargs):
        model_input = []
        for input in inputs:
            model_input.append(input.get_box(PartName.body_part))
        model_input = np.array(model_input)
        if depth_image is not None:
            root_depth = get_3d_depth_bybox(depth_image,
                                            model_input,
                                            depth_scale=self.depth_scale,
                                            percent=self.percent)
        else:
            root_depth = eval_rgb_depth_by_boxes(image,
                                                 model_input,
                                                 f=self.focal)
            print(root_depth)
        if self.input_type == "skeleton":
            model_input = []
            for input in inputs:
                model_input.append(input.get_keypoint(PartName.body_part, keypoint_type=KeyPointType.Kps2D))
        return np.array(model_input), root_depth

    def __transfer_output(self, person_items, outputs, *args, **kwargs):
        if self.input_type == "image":
            for person_item, kp2d, kp3d in zip(person_items, outputs[0], outputs[1]):
                person_item.update_keypoint(PartName.body_part, kp3d, self.data_type)
                person_item.update_keypoint(PartName.body_part, kp2d, KeyPointType.Kps2D, update_type=False)
        else:
            for person_item, kp3d in zip(person_items, outputs):
                person_item.update_keypoint(PartName.body_part, kp3d, self.data_type)
        return person_items

