import numpy as np
from src.utils import PersonItem, PartName
from src.utils.person_item import person_wrapper

class PersonReidFlag:
    changed = 1
    unchanged = 0

class PersonReidIdx:
    init_feature = 0
    middle_feature = 1
    edge_feature = 2



@person_wrapper
class PersonFollowItem(PersonItem):
    def __init__(self, person_item,
                 max_feature_bank: int,
                 reid_dim: int):
        # super(PersonFollowItem, self).__init__(person_item.get_box(PartName.body_part),
        #                                        person_item.get_conf(PartName.body_part))
        # self.update(person_item)
        self.__reid_feature_bank = np.zeros((max_feature_bank, reid_dim)) - 1

    def update_reid_feature(self, reid: np.array, idx: int):
        self.__reid_feature_bank[idx] = reid

    def get_reid_feature(self):
        return self.__reid_feature_bank


    # def update(self, person):


