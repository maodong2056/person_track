import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

def min_cost_matching(distance_metric, max_distance, fboxes, bboxes, names):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """

    if len(fboxes) == 0 or len(bboxes) == 0:
        unmatched_fboxes = []
        unmatched_bboxes = []
        if len(fboxes) != 0:
            for i, fbox in enumerate(fboxes):
                unmatched_fboxes.append((fbox, names[i]))
        if len(bboxes) != 0:
            for i, bbox in enumerate(bboxes):
                unmatched_bboxes.append(bbox)
        return [], unmatched_fboxes, unmatched_bboxes  # Nothing to match.

    cost_matrix = distance_metric(fboxes, bboxes)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_fboxes, unmatched_bboxes = [], [], []
    for col, bbox in enumerate(bboxes):
        if col not in col_indices:
            unmatched_bboxes.append(bbox)
    for row, fbox in enumerate(fboxes):
        if row not in row_indices:
            unmatched_fboxes.append((fbox, names[row]))
    for row, col in zip(row_indices, col_indices):
        fbox = fboxes[row]
        bbox = bboxes[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_fboxes.append((fbox, names[row]))
            unmatched_bboxes.append(bbox)
        else:
            matches.append(((fbox, names[row]), bbox))
    return matches, unmatched_fboxes, unmatched_bboxes

def inside_iou_cost(fbox, bbox):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """

    cost_matrix = np.zeros((len(fbox), len(bbox)))
    foxes = fbox.copy()
    boxes = bbox.copy()
    candidates = np.asarray([to_tlwh(boxes[i][:4]) for i in range(len(boxes))])
    for row in range(fbox.shape[0]):
        fox = to_tlwh(foxes[row][:4])
        cost_matrix[row, :] = 1. - inside_iou(fox, candidates)
    return cost_matrix

def min_cost_matching_tag(distance_metric, max_distance, ftags, btags, fboxes=None, bboxes=None, iou_dis=0.5, use_conf=False):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """

    if len(ftags) == 0 or len(btags) == 0:
        unmatched_fboxes = []
        unmatched_bboxes = []
        if len(ftags) != 0:
            for i, ftag in enumerate(ftags):
                unmatched_fboxes.append(i)
        if len(btags) != 0:
            for i, btag in enumerate(btags):
                unmatched_bboxes.append(i)
        return [], unmatched_fboxes, unmatched_bboxes  # Nothing to match.

    if not use_conf:
        cost_matrix = distance_metric(ftags, btags)
    else:
        fconf = fboxes[:, 4]
        cost_matrix = distance_metric(ftags, btags, fconf)
    if fboxes is not None and bboxes is not None:
        iou_matrix = inside_iou_cost(fboxes, bboxes)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    if fboxes is not None and bboxes is not None:
        cost_matrix[iou_matrix > iou_dis] = 10000

    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_ftags, unmatched_btags = [], [], []
    for col, bbox in enumerate(btags):
        if col not in col_indices:
            unmatched_btags.append(col)
    for row, fbox in enumerate(ftags):
        if row not in row_indices:
            unmatched_ftags.append(row)
    for row, col in zip(row_indices, col_indices):
        # fbox = ftags[row]
        # bbox = btags[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_ftags.append(row)
            unmatched_btags.append(col)
        else:
            matches.append((row, col))
    return matches, unmatched_ftags, unmatched_btags

def min_cost_matching_openpose(body_ct, part_ct, tag_map, fboxes=None, bboxes=None,
                               depart_num=10, max_distance=7, iou_dis=0.5):
    if len(part_ct) == 0 or len(body_ct) == 0:
        unmatched_pboxes = []
        unmatched_bboxes = []
        if len(part_ct) != 0:
            for i, ftag in enumerate(part_ct):
                unmatched_pboxes.append(i)
        if len(body_ct) != 0:
            for i, btag in enumerate(body_ct):
                unmatched_bboxes.append(i)
        return [], unmatched_pboxes, unmatched_bboxes  # Nothing to match.
    part_ct_1 = part_ct[:, None, ...]
    body_ct_1 = body_ct[None, ...]
    lck = np.sqrt(np.sum((part_ct_1 - body_ct_1) ** 2, axis=2))  # l2 dis (head, body) axis = 2
    lck = np.expand_dims(lck, 2) + 1e-5
    tag_map_1 = tag_map.reshape(2, -1)
    limb = part_ct_1 - body_ct_1  # (head,body,2)
    v = limb / lck  # v (head,body,2)/(head,body)
    v = v.reshape((part_ct.shape[0] * body_ct.shape[0], 2))  # (head*body, 2)
    v4 = np.tile(v, (depart_num, 1))
    point = np.linspace(part_ct_1, body_ct_1, num=depart_num)  # (num, head, body, 2)
    point = point.reshape((depart_num * part_ct.shape[0] * body_ct.shape[0], 2))  # (num*head*body, 2)
    dot = np.sum(tag_map_1[:, point[:, 1].round().astype(int) * 128 + point[:, 0].round().astype(int)] * v4.T, axis=0).reshape(
        (depart_num, part_ct.shape[0] * body_ct.shape[0]))
    E = np.sum(dot, axis=0)
    E = E.reshape((part_ct.shape[0], body_ct.shape[0]))
    cost_matrix = 10 - E
    if fboxes is not None and bboxes is not None:
        iou_matrix = inside_iou_cost(fboxes, bboxes)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    if fboxes is not None and bboxes is not None:
        cost_matrix[iou_matrix > iou_dis] = 10000
    row_indices, col_indices = linear_assignment(cost_matrix)
    matches, unmatched_ptags, unmatched_btags = [], [], []
    for col, bbox in enumerate(body_ct):
        if col not in col_indices:
            unmatched_btags.append(col)
    for row, fbox in enumerate(part_ct):
        if row not in row_indices:
            unmatched_ptags.append(row)
    for row, col in zip(row_indices, col_indices):
        # fbox = ftags[row]
        # bbox = btags[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_ptags.append(row)
            unmatched_btags.append(col)
        else:
            matches.append((row, col))
    return matches, unmatched_ptags, unmatched_btags



def tag_cost(ftag, btag, fconf=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """

    cost_m = np.zeros((len(ftag), len(btag)))
    ftags = ftag.copy()
    btags = btag.copy()
    if fconf is None:
        cost_matrix = (ftags - btags.T) ** 2
    else:
        cost_matrix = (ftags - btags.T) ** 2 / np.power(fconf, 2)[:, None]
    # candidates = np.asarray([btags[i] for i in range(len(btags))])
    # candidates = np.asarray([btags[i] for i in range(len(btags))])
    # for row in range(fbox.shape[0]):
    #     fox = to_tlwh(foxes[row][:4])
    #     cost_matrix[row, :] = 1. - inside_iou(fox, candidates)
    return cost_matrix

def to_tlwh(ret):
    """Get current position in bounding box format `(top left x, top left y,
    width, height)`.

    Returns
    -------
    ndarray
        The bounding box.

    """
    ret[2] = ret[2] - ret[0]
    ret[3] = ret[3] - ret[1]
    return ret

def inside_iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    # area_bbox = bbox[2:].prod()
    area_candidates = bbox[2:].prod(axis=0)
    return area_intersection / area_candidates