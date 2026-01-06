import numpy as np
from src.algorithm.baseAlgorithm import xyxy2xywh
from src.utils import PersonItem, PartName

def get_3d_depth_bybox(depth_image, bboxes, depth_scale, percent=20):
    depth_person_map = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        depth_person = depth_image[int(ymin):int(ymax),
                       int(xmin):int(xmax)].reshape(-1)
        z = depth_person * depth_scale
        z = z[np.where(z > 0.2)]
        percent_num = np.percentile(z, percent)
        zs = np.delete(z, np.where((z < (percent_num - 1)) | (z > (percent_num + 1)))).mean()
        depth_person_map.append(zs * 1000)
    return np.array(depth_person_map)

def get_cam_depth_bybox(depth_image, bboxes, f, c, depth_scale, percent=20):
    x_map = []
    y_map = []
    z_map = []
    rotate_map = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        center_x, center_y = (xmin + (xmax - xmin)/2., ymin + (ymax - ymin)/2.)
        depth_person = depth_image[int(ymin):int(ymax),
                       int(xmin):int(xmax)].reshape(-1)
        z = depth_person * depth_scale
        z = z[np.where(z > 0.2)]
        percent_num = np.percentile(z, percent)
        zs = np.delete(z, np.where((z < (percent_num - 1)) | (z > (percent_num + 1)))).mean()
        xs = (center_x - c[0]) / f[0] * (zs * 1000)
        x_map.append(xs)
        ys = (center_y - c[1]) / f[1] * (zs * 1000)
        y_map.append(ys)
        z_map.append(zs * 1000)
        rotate = np.arctan(xs/(zs * 1000))
        rotate_map.append(rotate/np.pi * 180)
    return np.array([x_map, y_map, z_map, rotate_map])

def get_3d_depth_by_single_box(depth_image, bbox, depth_scale, percent=20):
    xmin, ymin, xmax, ymax = bbox
    depth_person = depth_image[int(ymin):int(ymax),
                   int(xmin):int(xmax)].reshape(-1)
    z = depth_person * depth_scale
    z = z[np.where(z > 0.2)]
    percent_num = np.percentile(z, percent)
    zs = np.delete(z, np.where((z < (percent_num - 1)) | (z > (percent_num + 1)))).mean()
    return zs * 1000

def process_bbox(bbox, width, height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = 1
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox



bbox_real = [2000, 2000]
def eval_rgb_depth_by_boxes(image, bboxes, f):
    bb_cp = bboxes.copy()
    depth = []
    if len(bb_cp) == 0:
        return depth
    bb_cp = xyxy2xywh(bb_cp)
    original_img_height, original_img_width, _ = image.shape
    for box in bb_cp:
        new_bbox = process_bbox(box, original_img_width, original_img_height)
        k_value = np.array(
            [np.sqrt(bbox_real[0] * bbox_real[1] * f[0] * f[1] / (
                     new_bbox[2] * new_bbox[3]))]).astype(np.float32)
        depth.append(k_value * 1.09)
    return depth


def eval_rgb_depth_by_kps(kps, kps_name, vis_thres, f):
    kp_cp = kps.copy()
    head_idx = kps_name.index("Head")
    neck_idx = kps_name.index("Neck")
    pelvis_idx = kps_name.index("Pelvis")
    lhip_idx = kps_name.index("L_Hip")
    lknee_idx = kps_name.index("L_Knee")
    lwrist_idx = kps_name.index("L_Ankle")
    rhip_idx = kps_name.index("R_Hip")
    rknee_idx = kps_name.index("R_Knee")
    rwrist_idx = kps_name.index("R_Ankle")
    kp_idx = np.array([head_idx, neck_idx, pelvis_idx, lhip_idx, lknee_idx, lwrist_idx,
                       rhip_idx, rknee_idx, rwrist_idx])
    cal_kp = kp_cp[:, kp_idx, :]
    top_exist = (cal_kp[:, [0, 1, 2], 2] > vis_thres).sum(axis=1) == 3
    l_leg_exist = (cal_kp[:, [3, 4, 5], 2] > vis_thres).sum(axis=1) == 3
    r_leg_exist = (cal_kp[:, [6, 7, 8], 2] > vis_thres).sum(axis=1) == 3
    vis_mask = (top_exist & (l_leg_exist | r_leg_exist))
    head_dis = np.linalg.norm(cal_kp[:, 0, :2] - cal_kp[:, 1, :2])
    body_dis = np.linalg.norm(cal_kp[:, 1, :2] - cal_kp[:, 2, :2])
    l_leg_dis_1 = np.linalg.norm(cal_kp[:, 3, :2] - cal_kp[:, 4, :2], axis=1)
    l_leg_dis_2 = np.linalg.norm(cal_kp[:, 4, :2] - cal_kp[:, 5, :2], axis=1)
    r_leg_dis_1 = np.linalg.norm(cal_kp[:, 6, :2] - cal_kp[:, 7, :2], axis=1)
    r_leg_dis_2 = np.linalg.norm(cal_kp[:, 7, :2] - cal_kp[:, 8, :2], axis=1)
    l_leg_dis = (l_leg_dis_1 + l_leg_dis_2)
    r_leg_dis = (r_leg_dis_1 + r_leg_dis_2)
    leg_dis_mean = (l_leg_dis + r_leg_dis) / 2.
    leg_dis = np.where(l_leg_exist & r_leg_exist, leg_dis_mean, np.where(l_leg_exist, l_leg_dis, r_leg_dis))
    whole_dis = head_dis + body_dis + leg_dis
    depth = []
    for vis, dis in zip(vis_mask, whole_dis):
        dis = dis * 1.3
        k_value = np.array(
                [np.sqrt(bbox_real[0] * bbox_real[1] * f[0] * f[1] / (
                         dis * dis))]).astype(np.float32)
        depth.append(k_value * 1.09 if vis else -1)
    return depth

def eval_rgb_depth_by_personItem(image, person_items, kps_name, vis_thres, focal, type):
    if type == "skeleton":
        boxes = np.array([item.get_box(PartName.body_part) for item in person_items])
        kps = np.array([item.get_keypoint(PartName.body_part) for item in person_items])
        depth_kps = eval_rgb_depth_by_kps(kps,
                                          kps_name,
                                          vis_thres,
                                          focal)
        depth_boxes = eval_rgb_depth_by_boxes(image,
                                             boxes,
                                             focal)
        root_depth = [kp if kp != -1 else box for (kp, box) in zip(depth_kps, depth_boxes)]


    else:
        boxes = np.array([item.get_box(PartName.body_part) for item in person_items])
        root_depth = eval_rgb_depth_by_boxes(image,
                                             boxes,
                                             focal)
    return root_depth