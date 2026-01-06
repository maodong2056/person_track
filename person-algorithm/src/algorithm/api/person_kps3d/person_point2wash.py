import numpy as np
from src.utils import cam2pixel
from src.utils import load_model_setting
from src.utils import PersonItem, PartName, KeyPointType
from src.algorithm.deep_model import get_model, MODEL_TASK
from src.algorithm.api.basic_task import BasicTask
from src.algorithm.baseAlgorithm import get_3d_depth_bybox, eval_rgb_depth_by_personItem
from src.algorithm.api.person_kps3d.person_kps3d import PersonKps3D
from src.algorithm.api.person_kps3d.person_item_point import PersonPointItem


class PersonPointWash(PersonKps3D):
    def __init__(self, model_setting, person_item=PersonPointItem, depth_scale=None):
        super(PersonPointWash, self).__init__ \
            (model_setting, person_item=person_item, depth_scale=depth_scale)

    def get_output(self, image, inputs, depth_image=None, *args, **kwargs):
        if len(inputs) != 0:
            model_input, root_depth = self.__transfer_input(image, inputs, depth_image)
            outputs = self.model.get_output(image, model_input, root_depth, **self.para)
        else:
            outputs = []
        result = self.__transfer_output(inputs, outputs)
        return result

    def __transfer_input(self, image, inputs, depth_image, *args, **kwargs):
        model_input = np.array([item.get_box(PartName.body_part) for item in inputs])
        if depth_image is not None:
            root_depth = get_3d_depth_bybox(depth_image,
                                            model_input,
                                            depth_scale=self.depth_scale,
                                            percent=self.percent)
        else:
            root_depth = eval_rgb_depth_by_personItem(image, inputs,
                                                      self.joints_name,
                                                      self.kps_vis_thres,
                                                      self.focal,
                                                      self.input_type)
        print(root_depth)
        if self.input_type == "skeleton":
            model_input = []
            for input in inputs:
                model_input.append(input.get_keypoint(PartName.body_part, keypoint_type=KeyPointType.Kps2D))
        return np.array(model_input), root_depth

    def __transfer_output(self, person_items, outputs, *args, **kwargs):
        person_items_result = []
        if self.input_type == "image":
            for person_item, kp2d, kp3d in zip(person_items, outputs[0], outputs[1]):
                person_item.update_keypoint(PartName.body_part, kp3d, self.data_type)
                person_item.update_keypoint(PartName.body_part, kp2d, KeyPointType.Kps2D, update_type=False)
                person_items_result.append(self.person_item(person_item, self.joints_name))
        else:
            for person_item, kp3d in zip(person_items, outputs):
                person_item.update_keypoint(PartName.body_part, kp3d, self.data_type)
                person_items_result.append(self.person_item(person_item, self.joints_name))
        return person_items_result

