from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
#evaluate
collect_dir = "/raid/hzj/data/keypoint_project/Collection/20210615"
coco = COCO(os.path.join(collect_dir, "dataset.json"))
res_file = os.path.join(collect_dir, "my_result.json")
coco_dt = coco.loadRes(res_file)
coco_eval = COCOeval(coco, coco_dt, 'bbox')
coco_eval.params.useSegm = None
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
coco_eval = COCOeval(coco, coco_dt, 'keypoints')
coco_eval.params.useSegm = None
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()