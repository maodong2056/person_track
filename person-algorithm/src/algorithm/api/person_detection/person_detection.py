from src.utils import PersonItem, PartName
from src.utils import load_model_setting
from src.algorithm.deep_model import get_model, MODEL_TASK
from src.algorithm.api.basic_task import BasicTask

class PersonDetection(BasicTask):
    def __init__(self, model_setting, person_item=PersonItem):
        super(PersonDetection, self).__init__()
        self.model_name, \
        self.model_dir, \
        self.structure, \
        self.para, self.gpu_set \
            = load_model_setting(model_setting)
        self.person_item = person_item
        self.model = get_model(MODEL_TASK.TASK_BODY, self.model_name,
                               self.model_dir,
                               **self.structure, **self.gpu_set)
    
    def get_output(self, image, *args, **kwargs):
        _, outputs = self.model.get_output(image, **self.para)
        result = self.__transfer_output(outputs)
        return result

    def __transfer_output(self, outputs, *args, **kwargs):
        result_list = []
        for out in outputs:
            person = self.person_item(out["body_box"], out["body_conf"])
            if out["reid"] is not None:
                person.update_feature(PartName.body_part, out["reid"])
            if out["face_conf"] is not None:
                person.update_box(PartName.face_part, out["face_box"], out["face_conf"])
                person.update_lm(PartName.face_part, out["face_lm"])
            if out["lh_conf"] is not None:
                person.update_box(PartName.lefthand_part, out["lh_box"], out["lh_conf"])
                person.update_lm(PartName.lefthand_part, out["lh_lm"])
            if out["rh_conf"] is not None:
                person.update_box(PartName.righthand_part, out["rh_box"], out["rh_conf"])
                person.update_lm(PartName.righthand_part, out["rh_lm"])
            result_list.append(person)
        return result_list
