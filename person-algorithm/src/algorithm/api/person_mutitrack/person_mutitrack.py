import numpy as np
import random
from copy import deepcopy
from src.utils import load_model_setting
from src.utils import PersonItem, PartName
from src.algorithm.api.basic_task import BasicTask
from src.algorithm.deep_model import get_model, MODEL_TASK


class PersonMutiTrack(BasicTask):
    def __init__(self, model_setting, person_item=PersonItem):
        super(PersonMutiTrack, self).__init__()
        self.model_name, \
        self.model_dir, \
        self.structure, \
        self.para, self.gpu_set \
            = load_model_setting(model_setting)
        self.model = get_model(MODEL_TASK.TASK_TRACK,
                               self.model_name,
                               **self.structure,
                               **self.para)
        self.person_item = person_item

    def get_output(self, image, inputs, *args, **kwargs):
        model_input = deepcopy(inputs)
        model_input, str_dict = self.__transfer_input(model_input)
        outputs = self.model.get_output(model_input, img_size=image.shape)
        result = self.__transfer_output(inputs, outputs, str_dict)
        return result

    def __transfer_input(self, inputs, *args, **kwargs):
        model_input = []
        str_dict = {}
        for idx, input in enumerate(inputs):
            btag, ftag = random.randint(1, 1000000), random.randint(1, 1000000)
            model_input.append({
                "body_box": input.get_box(PartName.body_part),
                "body_conf": input.get_conf(PartName.body_part),
                "body_tag": btag,
                "reid": input.get_feature(PartName.body_part),
                "face_box": input.get_box(PartName.face_part) if input.get_part_state(PartName.face_part) else None,
                "face_conf": input.get_conf(PartName.face_part) if input.get_part_state(PartName.face_part) else None,
                "face_lm": input.get_lm(PartName.face_part) if input.get_part_state(PartName.face_part) else None,
                "face_tag": ftag,
            })
            str_dict[(btag, ftag)] = idx
        return model_input, str_dict

    def __transfer_output(self, person_items, outputs, str_dict, *args, **kwargs):
        for out in outputs:
            btag, ftag = out["body_tag"], out["face_tag"]
            if not (btag, ftag) in str_dict.keys():
                continue
            idx = str_dict[(btag, ftag)]
            if idx in range(len(person_items)):
                person_items[idx].update_trackid(out["id"])
        return person_items