import numpy as np
import random
import cv2
import math
# helper functions
def get_aug_config():
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2

    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

    do_occlusion = random.random() <= 0.5
    # do_occlusion = False
    return scale, rot, do_flip, color_scale, do_occlusion


def generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion, input_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    # synthetic occlusion
    if bbox is None:
        bbox = np.array([0., 0., 256., 256.])

    if do_occlusion:
        while True:
            area_min = 0.0
            area_max = 0.7
            synth_area = (random.random() * (area_max - area_min) + area_min) * bbox[2] * bbox[3]

            ratio_min = 0.3
            ratio_max = 1 / 0.3
            synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

            synth_h = math.sqrt(synth_area * synth_ratio)
            synth_w = math.sqrt(synth_area / synth_ratio)
            synth_xmin = random.random() * (bbox[2] - synth_w - 1) + bbox[0]
            synth_ymin = random.random() * (bbox[3] - synth_h - 1) + bbox[1]

            if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < img_width and synth_ymin + synth_h < img_height:
                xmin = int(synth_xmin)
                ymin = int(synth_ymin)
                w = int(synth_w)
                h = int(synth_h)
                img[ymin:ymin + h, xmin:xmin + w, :] = np.random.rand(h, w, 3) * 255
                break
    if bbox is not None:
        bb_c_x = float(bbox[0] + 0.5 * bbox[2])
        bb_c_y = float(bbox[1] + 0.5 * bbox[3])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])
    else:
        bb_c_x, bb_c_y = 128., 128.
        bb_width, bb_height = 256., 256.

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale,
                                    rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_patch)
    # plt.show()
    # bgrtorgb

    img_patch = img_patch[:, :, ::-1].copy()
    # import matplotlib.pyplot as plt
    # plt.imshow(img_patch)
    # plt.show()
    # img_patch = img_patch.astype(np.float32)

    return img_patch, trans

def generate_patch_kps(kps, scale, rot, input_shape):
    # synthetic occlusion
    top_left_x = kps[:, 0].min()
    top_left_y = kps[:, 1].min()
    bottom_right_x = kps[:, 0].max()
    bottom_right_y = kps[:, 1].max()
    center_x = (top_left_x + bottom_right_x) / 2.
    center_y = (top_left_y + bottom_right_y) / 2.
    w = bottom_right_x - top_left_x
    h = bottom_right_y - top_left_y
    aspect_ratio = input_shape[1] / input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    w = w * 1.25
    h = h * 1.25

    trans = gen_trans_from_patch_cv(center_x, center_y, w, h, input_shape[1], input_shape[0], scale,
                                    rot, inv=False)
    # img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_patch)
    # plt.show()
    # bgrtorgb

    # img_patch = img_patch[:, :, ::-1].copy()
    # import matplotlib.pyplot as plt
    # plt.imshow(img_patch)
    # plt.show()
    # img_patch = img_patch.astype(np.float32)

    return trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def trans_kps(kps, input_shape):
    keypoints = kps.copy()
    trans = generate_patch_kps(kps, 1.2, 0, input_shape)
    for idx, kp in enumerate(kps):
        keypoints[idx][:2] = trans_point2d(kp[:2], trans)
    return keypoints
