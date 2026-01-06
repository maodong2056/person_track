import torch
import copy
import numpy as np
import logging
logger = logging.getLogger(__name__)


def loadModel(model, model_path):
    model_state_dict = model.state_dict()
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'

    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.info('Skip loading parameter {}, required shape{}, ' \
                            'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            logger.info('Drop parameter {}.'.format(k) + msg)

    for k in model_state_dict:
        if not (k in state_dict):
            logger.info('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return model


def KP_pose_norm(bbox, kp):
    kp_n = copy.deepcopy(kp)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    kp_n[:, 0] = (kp_n[:, 0] - bbox[0]) / w
    kp_n[:, 1] = (kp_n[:, 1] - bbox[1]) / h
    return kp_n


def KP_transform(keypoints):
    # PoseTrack(15 points), but COCO(18 points) is used in our keypoint model;
    neck_point = np.mean(np.stack([keypoints[5], keypoints[6]], 1), 1)
    center_point = np.mean(np.stack([keypoints[5], keypoints[6], keypoints[11], keypoints[12]], 1), 1)
    trans = [keypoints[0], neck_point, keypoints[6], keypoints[8], keypoints[10], keypoints[5], keypoints[7],
             keypoints[9], keypoints[12], keypoints[14], keypoints[16], keypoints[13], keypoints[15], keypoints[16],
             center_point]
    new_KP = np.concatenate(trans, axis=0)
    return new_KP, center_point


def keypoints2graph2data(keypoints, bbox):
    num_elements = len(keypoints)
    num_keypoints = num_elements / 3
    assert(num_keypoints == 15)

    x0, y0, w, h = bbox
    graph = 15 * [(0, 0)]
    for id in range(15):
        # normalize the corrdinates: mean 0, standard deviation 1
        x = keypoints[3 * id] - x0
        y = keypoints[3 * id + 1] - y0
        score = keypoints[3 * id + 2]
        graph[id] = (int(x), int(y))

    data_numpy = np.zeros((2, 1, 15, 1))
    data_numpy[0, 0, :, 0] = [x[0] for x in graph]
    data_numpy[1, 0, :, 0] = [x[1] for x in graph]

    return data_numpy


def enlarge_bbox(bbox: object, scale: object) -> object:
    assert (scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0: margin_x = 2
    if margin_y < 0: margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x = 0
        max_x = 2
        min_y = 0
        max_y = 2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def get_bbox_from_keypoints(body_keypoints, enlarge_scale):
    bmin = body_keypoints.min(0)
    bmax = body_keypoints.max(0)

    bbox = enlarge_bbox([bmin[0], bmin[1], bmax[0], bmax[1]], enlarge_scale)
    # bbox_in_xywh = x1y1x2y2_to_xywh(bbox)
    return np.array(bbox)

