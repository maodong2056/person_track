import numpy as np

def dict2detection(dicts):
    dets = np.ones((len(dicts), 47)) * -1
    for idx, dic in enumerate(dicts):
        dets[idx, 0:4] = dic["body_box"]
        dets[idx, 4] = dic["body_conf"]
        dets[idx, 5:9] = dic["face_box"] if dic["face_box"] else np.ones((4,))*-1
        dets[idx, 9] = dic["face_conf"]
        dets[idx, 10:20] = dic["face_lm"]
        dets[idx, 20:24] = dic["lh_box"]
        dets[idx, 24] = dic["lh_conf"]
        dets[idx, 25:29] = dic["rh_box"]
        dets[idx, 29] = dic["rh_conf"]