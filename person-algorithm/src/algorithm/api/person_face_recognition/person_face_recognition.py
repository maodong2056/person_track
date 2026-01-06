import numpy as np
import random
from copy import deepcopy
from src.utils import load_model_setting
from src.utils import PersonItem, PartName
from src.algorithm.api.basic_task import BasicTask
from src.algorithm.deep_model import get_model, MODEL_TASK


class PersonFaceRecognition(BasicTask):
    def __init__(self, model_setting, person_item=PersonItem):
        super(PersonFaceRecognition, self).__init__()
        self.model_name, \
        self.model_dir, \
        self.structure, \
        self.para, self.gpu_set \
            = load_model_setting(model_setting)
        self.model = get_model(MODEL_TASK.TASK_FACEREC,
                               self.model_name, self.model_dir,
                               **self.structure, **self.gpu_set)
        self.person_item = person_item

    def get_output(self, image, inputs, *args, **kwargs):
        model_input = deepcopy(inputs)
        model_input, face_mask = self.__transfer_input(model_input)
        if len(model_input) != 0:
            outputs = self.model.get_output(image, model_input, **self.para)
        else:
            outputs = []
        result = self.__transfer_output(inputs, outputs, face_mask)
        return result

    def __transfer_input(self, inputs, *args, **kwargs):
        model_input = []
        face_mask = []
        for idx, inp in enumerate(inputs):
            if inp.get_part_state(PartName.face_part):
                model_input.append({
                    "face_box": inp.get_box(PartName.face_part),
                    "face_conf": inp.get_conf(PartName.face_part),
                    "face_lm": inp.get_lm(PartName.face_part),
                })
                face_mask.append(True)
            else:
                face_mask.append(False)
        return model_input, face_mask

    def __transfer_output(self, person_items, outputs, mask, *args, **kwargs):
        for idx, (person_item, m) in enumerate(zip(person_items, mask)):
            if m:
                face_embedding = outputs[idx]
                person_item.update_feature(PartName.face_part, face_embedding)
        return person_items