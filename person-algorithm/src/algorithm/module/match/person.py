import logging
from src.algorithm.module.match.face import _Face
from src.algorithm.module.match.body import _Body


logger = logging.getLogger(__name__)
class PersonState:
    Inactivate = 0
    Tentative = 1
    Confirmed = 2
    Reconfirm = 3

class _Person(object):
    def __init__(self, person_name, person_sex, person_color, faceEmb, max_match_step=10, max_age=70, device=None, *args, **kwargs):
        self.person_name = person_name
        self.person_sex = person_sex
        self.person_color = person_color
        self.device = device
        self.face = _Face(faceEmb)
        self.body = _Body(device=self.device)
        self.trackStepMap = {}
        self.trackAgeMap = {}
        self.confirmedTrackID = None
        self.state = PersonState.Inactivate
        self.max_age = max_age
        self.max_match_step = max_match_step
        self.time_since_update = 0
        # self.bbox = None

    def step(self):
        self.time_since_update += 1
        self.face.step()
        if self.time_since_update > self.max_age:
            self.state = PersonState.Inactivate
            self.time_since_update = 1
        for track_id in self.trackAgeMap:
            self.trackAgeMap[track_id] += 1

    def update(self, track_id, faceEmb, fdis, fbox, bodyEmb, bdis, bbox, iou):
        try:
            # self.bbox = bbox
            self.time_since_update = 0
            self.face.update(fbox, emb=faceEmb, dis=fdis)
            self.body.update(bbox, emb=bodyEmb, face_dis=fdis, body_dis=bdis, iou=iou)
            # 检查字典中是否存在该track id， 如若存在则进行计数标记
            if track_id not in self.trackStepMap.keys():
                self.trackStepMap[track_id] = 1
                self.trackAgeMap[track_id] = 0
            else:
                self.trackStepMap[track_id] += 1
                self.trackAgeMap[track_id] = 0
            # 如果原来有id
            if self.confirmedTrackID is not None:
                # 如果track id 不为当前已经确认的id，且状态为Confirm，则标记为待激活状态，计数归1，需要进行重新确认
                if self.confirmedTrackID != track_id and self.get_confirm():
                    self.state = PersonState.Tentative
                    self.trackStepMap[track_id] = 1
                # 如果track id 当前已经确认的id，且状态为待激活状态，则变为激活状态，以前丢失这帧又找回
                if self.confirmedTrackID == track_id and (self.state == PersonState.Tentative or self.state == PersonState.Reconfirm):
                    self.state = PersonState.Confirmed
            # 如果该track id命中超过阈值
            if self.confirmedTrackID != track_id and self.trackStepMap[track_id] >= self.max_match_step:
                # 如果原来的确认id是None（没有进行过确认），则将该ID标记为确认ID
                self.confirmedTrackID = track_id
                self.state = PersonState.Confirmed
                # 确认一个新的id后 将别的id计数清0
                for id in self.trackStepMap.keys():
                    if id != track_id:
                        self.trackStepMap[id] = 0
            if self.trackStepMap[track_id]%50 == 0 and self.state == PersonState.Confirmed:
                self.state = PersonState.Reconfirm
        except Exception as e:
            logger.error(e)


    def mark_miss(self):
        """
        每次结尾使用，用于标记已经丢失的目标，
        如果丢帧没有update的话，则标记为消失，下次update需要重新确认
        :return:
        """
        if self.confirmedTrackID is not None:
            if self.trackAgeMap[self.confirmedTrackID]!=0:
                self.state = PersonState.Tentative
        del_id = []
        for id in self.trackAgeMap.keys():
            if self.trackAgeMap[id] > self.max_age:
                del_id.append(id)
        for id in del_id:
            del self.trackAgeMap[id]
            del self.trackStepMap[id]
            if self.confirmedTrackID == id:
                self.confirmedTrackID = None

    def get_confirm(self):
        if self.state==PersonState.Confirmed or self.state==PersonState.Reconfirm:
            return True
        else:
            return False

    def get_nore_confirm(self):
        if self.state==PersonState.Confirmed:
            return True
        else:
            return False

    def get_embedding(self):
        face_emb = self.face.get_face_embedding()
        body_emb = self.body.get_body_embedding()
        return face_emb, body_emb






