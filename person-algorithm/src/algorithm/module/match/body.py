import time
import torch

class BodyState:
    Inactivate = 0
    Tentative = 1
    Confirmed = 2


class _Body():
    def __init__(self, device, min_hit=5, ratio_thre=1.8, face_qualified=0.7, body_qualified=0.6, update_gap_time=2, max_count=5, iou_max=0.5):
        self.min_hit = min_hit
        self.ratio_thre = ratio_thre
        self.face_qualified = face_qualified
        self.body_qualified = body_qualified
        self.update_gap_time = update_gap_time
        self.max_count = max_count
        self.iou_max = iou_max
        self.init_time = time.time()
        self.last_update_time = self.init_time
        self.bbox = None
        self.device = device
        self.embedding = torch.ones((4096,), dtype=torch.float32, requires_grad=False).to(self.device) * -1
        # self.embedding = torch.ones((2048,), dtype=torch.float32, requires_grad=False).cuda() * -1
        self.unfit_count = 0

    def update(self, bbox, emb=None, face_dis=None, body_dis=None, iou=None):
        if face_dis is not None and body_dis is not None and (time.time() - self.last_update_time <= self.update_gap_time):
            if face_dis < self.face_qualified and body_dis > self.body_qualified:
                self.unfit_count = self.unfit_count + 1
            else:
                self.unfit_count = 0
            if self.unfit_count >= self.max_count and float(bbox[3]-bbox[1])/float(bbox[2]-bbox[0]) >= self.ratio_thre and iou <= self.iou_max: # assume bbox is in form of ['bxc','byc','bw','bh']
                self.embedding = emb
                self.unfit_count = 0
        self.bbox = bbox
        self.last_update_time = time.time()

    def get_body_embedding(self):
        return self.embedding
