import torch
import numpy as np

def wrap(func, unsqueeze, *args):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape) - 1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape) - 1)
    return v + 2 * (q[..., :1] * uv + uuv)

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(X, R, t):
    return wrap(qrot, False, np.tile(R, X.shape[:-1] + (1,)), X) + t

def pixcel_camera_transfer(pixel_coord, f1, c1, f2, c2):
    u = (f2[0] / f1[0]) * (pixel_coord[:, 0] - c1[0]) + c2[0]
    v = (f2[1] / f1[1]) * (pixel_coord[:, 1] - c1[1]) + c2[1]
    pixel_new = np.concatenate((u[:, None], v[:, None]), 1)
    return pixel_new


def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t

def rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2

def get_bbox(joint_img):
    # bbox extract from keypoint coordinates
    bbox = np.zeros((4))
    xmin = np.min(joint_img[:,0])
    ymin = np.min(joint_img[:,1])
    xmax = np.max(joint_img[:,0])
    ymax = np.max(joint_img[:,1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1
    
    bbox[0] = (xmin + xmax)/2. - width/2*1.2
    bbox[1] = (ymin + ymax)/2. - height/2*1.2
    bbox[2] = width*1.2
    bbox[3] = height*1.2

    return bbox


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]))

    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def fliplr_joints(_joints, width, matched_parts):
    """
    flip coords
    joints: numpy array, nJoints * dim, dim == 2 [x, y] or dim == 3  [x, y, z]
    width: image width
    matched_parts: list of pairs
    """
    joints = _joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped

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
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox