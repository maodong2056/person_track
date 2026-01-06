# @Time : 2021/3/17 11:08 
# @Author : Altair.Huazj
# @File : transpose.py 
# @Software: PyCharm
import cv2
import numpy as np


def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.

    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform

    Returns:
        np.ndarray: Transformed points.
    """
    assert len(pt) == 2
    new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])

    return new_pt


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt

def get_box_cs(box, input_size, scale_rate):
    """

    :param box: (n, 4) input the xywh bbox shape
    :param input_size:(2, ) the size of image to input the network
    :return: return the center(n, 2), scale (n, 2)
    """
    x, y, w, h = box[:, 0:1], box[:, 1:2], box[:, 2:3], box[:, 3:4]
    aspect_ratio = input_size[0] / input_size[1]
    center = np.concatenate([x + w * 0.5, y + h * 0.5], axis=1)

    h = np.where(w > aspect_ratio * h, w * 1.0 / aspect_ratio, h)
    w = np.where(w < aspect_ratio * h, h * aspect_ratio, w)

    scale = np.concatenate([w / 200.0, h / 200.0], axis=1)
    scale = scale * scale_rate

    return center, scale

def get_kps_cs(kps, input_size, scale_rate):
    """

    :param box: (n, 4) input the xywh bbox shape
    :param input_size:(2, ) the size of image to input the network
    :return: return the center(n, 2), scale (n, 2)
    """
    kps_x_min, kps_x_max = kps[:, :, 0:1].min(axis=1), kps[:, :, 0:1].max(axis=1)
    kps_y_min, kps_y_max = kps[:, :, 1:2].min(axis=1), kps[:, :, 1:2].max(axis=1)
    w, h = kps_x_max - kps_x_min, kps_y_max - kps_y_min

    aspect_ratio = input_size[0] / input_size[1]
    center = np.concatenate([kps_x_min + w * 0.5, kps_y_min + h * 0.5], axis=1)

    h = np.where(w > aspect_ratio * h, w * 1.0 / aspect_ratio, h)
    w = np.where(w < aspect_ratio * h, h * aspect_ratio, w)

    scale = np.concatenate([w / 200.0, h / 200.0], axis=1)
    scale = scale * scale_rate

    return center, scale


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    First calculate the transformation matrix from `get_affine_transform()`,
    then affine transform the predicted keypoint coordinates back
    to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, tags, scores)
            * If ndims=5, corrds are composed of (x, y, tags,
              flipped_tags, scores)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    if use_udp:
        # The input scale is normalized by deviding a factor of 200.
        # Here is a recover.
        scale = scale * 200.0
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
        target_coords = np.zeros(coords.shape)
        target_coords[:,
                      0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:,
                      1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    else:
        target_coords = coords.copy()
        trans = get_affine_transform(center, scale, 0, output_size, inv=True)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def crop_image_by_bbox(image, bboxes, input_size, scale):
    c, s = get_box_cs(bboxes, input_size, scale)
    img = []
    for i in range(len(bboxes)):
        trans = get_affine_transform(c[i], s[i], 0, input_size)
        img.append(cv2.warpAffine(
                    image,
                    trans, (int(input_size[0]), int(input_size[1])),
                    flags=cv2.INTER_LINEAR))
    return img, c, s


def crop_by_keypoints(kps, input_size, scale):
    c, s = get_kps_cs(kps, input_size, scale)
    return c, s


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
    trans = generate_patch_kps(kps, 1.25, 0, input_shape)
    for idx, kp in enumerate(kps):
        keypoints[idx] = trans_point2d(kp, trans)
    return keypoints