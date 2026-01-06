import json
import numpy, torch, logging, os
from src.algorithm.module.match.person import _Person

logger = logging.getLogger(__name__)

class _People(object):
    def __init__(self, personDictFile, personNameFile, personEmbFile, device=None, max_match_step=10, max_age=70):
        self.personDictFile = personDictFile
        self.personNameFile = personNameFile
        self.personEmbFile = personEmbFile
        self.max_match_step = max_match_step
        self.max_age = max_age
        # personBank 与 trackid一一对应
        self.personBank = []
        # None表示inactive -1表示teactivate 其他表示track id
        self.confirmed_trackid = []
        self.device = device
        self.update_bank()


    def update_bank(self):
        if os.path.exists(self.personDictFile):
            with open(self.personDictFile, "r") as f:
                persons = json.load(f)
            self.personBank = []
            self.confirmed_trackid = []
            person_names = numpy.load(self.personNameFile)
            person_emb = torch.load(self.personEmbFile)
            for idx, per in enumerate(persons):
                name = per["person_name"]
                if person_names[idx + 1] == name:
                    emb = person_emb[idx]
                    self.personBank.append(_Person(**per, faceEmb=emb,
                                                  max_match_step=self.max_match_step,
                                                  max_age=self.max_age,
                                                  device=self.device))
                    self.confirmed_trackid.append(None)
                else:
                    logging.error("Person bank error!")
        else:
            logging.error("不存在用户文件，启动后请录入用户。")

    def step(self):
        for per in self.personBank:
            per.step()

    def get_embedding(self, results):
        # track_ids = results[:, -1]
        track_ids = numpy.array([res["id"] for res in results])
        confirm_idx = []
        confirm_det_idx = []
        # unconfirm_idx = []
        for idx, id in enumerate(track_ids):
            if id in self.confirmed_trackid:
                confirm_idx.append(self.confirmed_trackid.index(id))
                confirm_det_idx.append(idx)
        unconfirm_idx = [idx for idx in range(len(self.confirmed_trackid)) if idx not in confirm_idx]
        unconfirm_det_idx = [idx for idx in range(len(track_ids)) if idx not in confirm_det_idx]
        f_emb = []
        b_emb = []
        for uc_idx in unconfirm_idx:
            face, body = self.personBank[uc_idx].get_embedding()
            f_emb.append(face)
            b_emb.append(body)
        if len(f_emb) != 0:
            f_emb = torch.stack(f_emb)
            b_emb = torch.stack(b_emb)
        return confirm_idx, unconfirm_idx, confirm_det_idx, unconfirm_det_idx, f_emb, b_emb

    def update(self, det, emb_dict, confirm_person_idx, confirm_det_idx,
               matched_idx, unmatched_person_idx, unmatched_det_idx, iou):
        try:
            for i, idx in enumerate(confirm_person_idx):
                # track_id = det[confirm_det_idx[i]][-1]
                # body_box = det[confirm_det_idx[i]][:4]
                # face_box = det[confirm_det_idx[i]][4:18]
                # face_emb = None
                # body_emb = None
                track_id = det[confirm_det_idx[i]]["id"]
                body_box = det[confirm_det_idx[i]]["body_box"]
                face_box = numpy.concatenate([det[confirm_det_idx[i]]["face_box"], det[confirm_det_idx[i]]["face_lm"]]) if det[confirm_det_idx[i]]["face_box"] is not None else numpy.ones(14, ) * -1
                face_emb = None
                body_emb = None
                self.personBank[idx].update(track_id, face_emb, None, face_box, body_emb, None, body_box, None)

            for person_idx, det_idx, f_dist, b_dist, iou in matched_idx:
                # track_id = det[det_idx][-1]
                # body_box = det[det_idx][:4]
                # face_box = det[det_idx][4:18]
                # face_emb = emb_dict[det_idx][0]
                # body_emb = emb_dict[det_idx][1]
                track_id = det[det_idx]["id"]
                body_box = det[det_idx]["body_box"]
                face_box = numpy.concatenate([det[det_idx]["face_box"], det[det_idx]["face_lm"]]) if det[det_idx]["face_box"] is not None else numpy.ones(14, ) * -1
                face_emb = emb_dict[det_idx][0]
                body_emb = emb_dict[det_idx][1]
                self.personBank[person_idx].update(track_id, face_emb, f_dist, face_box, body_emb, b_dist, body_box, iou)

            for idx, per in enumerate(self.personBank):
                per.mark_miss()
                if per.get_nore_confirm():
                    self.confirmed_trackid[idx] = per.confirmedTrackID
                else:
                    self.confirmed_trackid[idx] = None
        except Exception as e:
            logger.error(e)

    def get_confirmed_person(self):
        per_list = []
        for per in self.personBank:
            if per.get_confirm():
                per_list.append(per)
        return per_list

    def clear(self):
        self.update_bank()
    #
    # def update(self, track_id, faceEmb, fdis, fbox, bodyEmb, bbox, bdis):
    #     self.face.update(fbox, emb=faceEmb, dis=fdis)
    #     # 检查字典中是否存在该track id， 如若存在则进行计数标记
    #     if track_id not in self.trackStepMap.keys():
    #         self.trackStepMap[track_id] = 1
    #         self.trackAgeMap[track_id] = 0
    #     else:
    #         self.trackStepMap[track_id] += 1
    #         self.trackAgeMap[track_id] = 0
    #     # 如果该track id命中超过阈值
    #     if self.trackStepMap[track_id] >= self.max_match_step:
    #         # 如果原来的确认id是None（没有进行过确认），则将该ID标记为确认ID
    #         if self.confirmedTrackID is None:
    #             self.confirmedTrackID = track_id
    #             self.state = PersonState.Confirmed
    #         # 如果原来有id
    #         else:
    #             # 如果track id 不为当前已经确认的id，且状态为Confirm，则标记为待激活状态，需要进行重新确认
    #             if self.confirmedTrackID != track_id and self.state==PersonState.Confirmed:
    #                 self.state = PersonState.Tentative
    #             # 如果track id 当前已经确认的id，且状态为待激活状态，则变为激活状态，以前丢失这帧又找回
    #             if self.confirmedTrackID == track_id and self.state==PersonState.Tentative:
    #                 self.state = PersonState.Confirmed
    #
    #
    #
    # def mark_miss(self):
    #     """
    #     每次结尾使用，用于标记已经丢失的目标，
    #     如果丢帧没有update的话，则标记为消失，下次update需要重新确认
    #     :return:
    #     """
    #     if self.trackAgeMap[self.confirmedTrackID]!=0:
    #         self.state = PersonState.Tentative
    #     for id in self.trackAgeMap.keys():
    #         if self.trackAgeMap[id] > self.max_age:
    #             del self.trackAgeMap[id]
    #             del self.trackStepMap[id]
    #             if self.confirmedTrackID == id:
    #                 self.confirmedTrackID = None
    #
    # def get_confirm(self, track_id):
    #     if track_id == self.confirmedTrackID and self.state==PersonState.Confirmed:
    #         return True
    #     else:
    #         return False
    #
    # def get_embedding(self):
    #     face_emb = self.face.get_face_embedding()
    #     return face_emb



if __name__ == '__main__':
    my_people = _People("../../../user/lib/persons.json",
                       "../../../user/lib/facebank/names.npy",
                       "../../../user/lib/facebank/facebank.pth")


