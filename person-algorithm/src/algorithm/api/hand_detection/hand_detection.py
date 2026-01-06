# from src.utils import PersonItem, HandPartName
from src.utils import HandItem, HandPartName
from src.utils import load_model_setting
from src.algorithm.deep_model import get_model, MODEL_TASK
from src.algorithm.api.basic_task import BasicTask


class HandDetection(BasicTask):
    def __init__(self, model_setting, hand_item=HandItem):
        super(HandDetection, self).__init__()
        self.model_name, \
        self.model_dir, \
        self.structure, \
        self.para, self.gpu_set \
            = load_model_setting(model_setting)
        self.hand_item = hand_item
        self.model = get_model(MODEL_TASK.TASK_HANDDET, self.model_name,
                               self.model_dir,
                               **self.structure, **self.gpu_set)

    def get_output(self, image, *args, **kwargs):
        outputs = self.model.get_output(image, **self.para)
        result = self.__transfer_output(outputs)
        return result

    def __transfer_output(self, outputs, *args, **kwargs):
        result_list = []
        for out in outputs:
            hand = self.hand_item(out["hand_box"], out["hand_conf"])
            hand.update_lm(HandPartName.hand_part, out["hand_lm"])
            hand.update_roi(HandPartName.hand_part, out['hand_roi'])
            result_list.append(hand)
        return result_list
