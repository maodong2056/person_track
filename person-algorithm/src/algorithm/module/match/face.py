import time
import torch

class FaceState:
    Inactivate = 0
    Tentative = 1
    Confirmed = 2


class _Face():
    def __init__(self, embedding, max_update_step=50, max_match_time=60, max_match_step=10, update_dis=0.5):
        self.init_time = time.time()
        self.embedding = embedding
        self.last_update_time = self.init_time
        self.update_embedding = []
        self.updated = False
        self.max_update_step = max_update_step
        self.max_match_time = max_match_time
        self.max_match_step = max_match_step
        self.hit = 0
        self.update_dis = update_dis
        self.state = FaceState.Inactivate
        self.bbox = None
        self.landmark = None

    def step(self):
        if (time.time() - self.last_update_time) > self.max_match_time:
            self.state = FaceState.Inactivate
            self.hit = 0

    def update(self, bbox, emb=None, dis=None):
        #  如果未进行过初始化更新则记录更新
        if not self.updated and dis is not None:
            # TODO: Add Update Condition
            if dis < self.update_dis:
                self.update_embedding.append(emb)
            if len(self.update_embedding) == self.max_update_step:
                # TODO: Add How to Concat The embedding
                self.update_embedding.append(self.embedding)
                up_emb = torch.stack(self.update_embedding)
                self.embedding = up_emb.mean(dim=0)
                self.updated = True
                self.update_embedding = []
        #  如果状态未激活则标记为待激活状态
        if self.state == FaceState.Inactivate:
            self.state = FaceState.Tentative
            self.hit += 1
        #  如果状态待激活且满足大于一定帧数 则标记为确认状态
        if self.state == FaceState.Tentative:
            self.hit += 1
            if self.hit >= self.max_match_step:
                self.state = FaceState.Confirmed
                self.hit = 0
        #  记录当前更新时间和框位置
        self.last_update_time = time.time()
        self.bbox = bbox[:4]
        self.landmark = bbox[4:]

    def unhit(self):
        self.hit = 0
        self.state = FaceState.Inactivate

    def is_face_confirmed(self):
        return self.state == FaceState.Confirmed

    def get_face_embedding(self):
        return self.embedding
