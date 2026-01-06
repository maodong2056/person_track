import sys
import json
import cv2
import numpy as np
import os
import math
from munkres import Munkres, make_cost_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
import matplotlib.pyplot as plt
# from pycocotools.coco import COCO

skeleton = [[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
                     [12, 13], [8, 14], [14, 15], [15, 16], [0, 1],
                     [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]]
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
pose_limb_color = palette[[
    0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 0, 4, 8, 12, 16
]]
pose_kpt_color = palette[[
    0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16,
    0
]]
# colorNotChange = [[13, 12], [12, 13]]  # In pose match with Angle measure, color of some point not change, leia 20210802
# collect_dir = "/raid/hzj/data/keypoint_project/Collection/20210615"
# images_dir = os.path.join(collect_dir, "images")
# vis_dir = os.path.join(collect_dir, "vis_res")
# dataset_file = os.path.join(collect_dir, "dataset.json")
# res_file = os.path.join(collect_dir, "my_result.json")


class MyKpsEval():
    def __init__(self, image_dir, vis_dir, dataset_dir, result_dir, iou_thre_kpt=0.5, iou_thre_bbox=0.5, write_image=True):
        self.dataset_dir = dataset_dir
        self.vis_dir = vis_dir
        self.result_dir = result_dir
        self.image_dir = image_dir
        self.file_names = {}
        self.gt_bboxes = {}
        self.gt_kpts = {}
        self.gt_areas = {}
        self.pr_bboxes = {}
        self.pr_kpts = {}
        self.write_image = write_image
        self.kpt_oks_sigmas = np.array(
        [.89, .89, .87, 1.07, 1.07, .87, .89, 1.07, .35, .26, .25, .79, .72, .62, .79, .72, .62]) / 10.0
        self.load_dataset(self.dataset_dir)
        self.load_result(self.result_dir)
        self.iou_thre_kpt = iou_thre_kpt
        self.iou_thre_bbox = iou_thre_bbox

    def load_dataset(self, dataset_dir):
        # ======== groud truth ========#
        with open(dataset_dir) as f1:
            dataset = json.load(f1)
        for annotation in dataset["annotations"]:
            image_id = annotation['image_id']
            file_name = dataset["images"][image_id]["file_name"]
            if image_id not in self.file_names.keys():
                self.file_names[image_id] = file_name
            bbox = annotation['bbox']  # xmin, ymin, w, h
            if image_id in self.gt_bboxes.keys():
                self.gt_bboxes[image_id].append(bbox)
            else:
                self.gt_bboxes[image_id] = []
                self.gt_bboxes[image_id].append(bbox)
            keypoints = annotation["keypoints"]
            tmp_kpt = []
            tmp_kpts = []
            for i in range(len(keypoints)):
                tmp_kpt.append(keypoints[i])
                if i % 3 == 2:
                    tmp_kpts.append(tmp_kpt)
                    tmp_kpt = []
            if image_id in self.gt_kpts.keys():
                self.gt_kpts[image_id].append(tmp_kpts)
            else:
                self.gt_kpts[image_id] = []
                self.gt_kpts[image_id].append(tmp_kpts)
            areas = annotation["area"]
            if image_id in self.gt_areas.keys():
                self.gt_areas[image_id].append(areas)
            else:
                self.gt_areas[image_id] = []
                self.gt_areas[image_id].append(areas)

    def load_result(self, res_file):
        # ======== pred ========#
        with open(res_file) as result_file:
            results = json.load(result_file)
        for result in results:
            image_id = result["image_id"]
            bbox = result['bbox']  # xmin, ymin, w, h
            if image_id in self.pr_bboxes.keys():
                self.pr_bboxes[image_id].append(bbox)
            else:
                self.pr_bboxes[image_id] = []
                self.pr_bboxes[image_id].append(bbox)
            keypoints = result["keypoints"]
            tmp_kpt = []
            tmp_kpts = []
            for i in range(len(keypoints)):
                tmp_kpt.append(keypoints[i])
                if i % 3 == 2:
                    tmp_kpts.append(tmp_kpt)
                    tmp_kpt = []
            if image_id in self.pr_kpts.keys():
                self.pr_kpts[image_id].append(tmp_kpts)
            else:
                self.pr_kpts[image_id] = []
                self.pr_kpts[image_id].append(tmp_kpts)

    def draw_kps(self, img, kps, kp_mask, colors, kp_thresh=0.4, alpha=1):
        # Draw the keypoints.
        for l in range(len(skeleton)):
            i1 = skeleton[l][0]
            i2 = skeleton[l][1]
            p1 = kps[i1, 0].astype(np.int32), kps[i1, 1].astype(np.int32)
            p2 = kps[i2, 0].astype(np.int32), kps[i2, 1].astype(np.int32)
            if kps[i1, 2] > kp_thresh and kps[i2, 2] > kp_thresh:
                cv2.line(
                    kp_mask, p1, p2,
                    color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            if kps[i1, 2] > kp_thresh:
                cv2.circle(
                    kp_mask, p1,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
                # cv2.putText(kp_mask, "{}".format(i1), p1, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
            if kps[i2, 2] > kp_thresh:
                cv2.circle(
                    kp_mask, p2,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
                # cv2.putText(kp_mask, "{}".format(i2), p2, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

    def vis_keypoints(self, img, kpts, flag, kp_thresh=0.4, alpha=1):
        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        if flag == 0:
            colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
            colors = [(0, 255, 0) for c in colors]
        elif flag == 1:
            colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
            colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
        else:
            colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
            colors = [(0, 0, 255) for c in colors]
        # Perform the drawing on a copy of the image, to allow for blending.
        kp_mask = np.copy(img)

        # Draw the keypoints.
        for kid, kpt in enumerate(kpts):
            img = self.draw_kps(img, kpt, kp_mask, colors, kp_thresh, alpha)
        # Blend the keypoints.
        return img

    # def draw_hkp(self, img, kpts, flag, kpt_score_thr=0.02):
    #     if flag == 0:  # gt
    #         transp = 0.3
    #     elif flag == 1:  # correct predition
    #         transp = 1
    #     else:  # false predicion
    #         transp = 0.8
    #     img_h, img_w, _ = img.shape
    #
    #     for kid, kpt in enumerate(kpts):
    #         for i in range(len(kpt)):
    #             x_coord, y_coord, kpt_score = int(kpt[i][0]), int(
    #                 kpt[i][1]), kpt[i][2]
    #             # if x_coord < 0 or y_coord < 0:
    #             # print("out of area!")
    #             if kpt_score > kpt_score_thr:
    #                 img_copy = img.copy()
    #                 if flag == 0:
    #                     r, g, b = 0, 255, 0
    #                 elif flag == 1:
    #                     r, g, b = pose_kpt_color[i]
    #                 else:
    #                     r, g, b = 0, 0, 255
    #                 cv2.circle(img_copy, (int(x_coord), int(y_coord)),
    #                            2, (int(r), int(g), int(b)), 1)
    #                 transparency = transp
    #                 cv2.addWeighted(
    #                     img_copy,
    #                     transparency,
    #                     img,
    #                     1 - transparency,
    #                     0,
    #                     dst=img)
    #     for kid, kpt in enumerate(kpts):
    #         for sk_id, sk in enumerate(skeleton):
    #             pos1 = (int(kpt[sk[0] - 1, 0]), int(kpt[sk[0] - 1,
    #                                                     1]))
    #             pos2 = (int(kpt[sk[1] - 1, 0]), int(kpt[sk[1] - 1,
    #                                                     1]))
    #             if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
    #                     and pos1[1] < img_h and pos2[0] > 0
    #                     and pos2[0] < img_w and pos2[1] > 0
    #                     and pos2[1] < img_h
    #                     and kpt[sk[0] - 1, 2] > kpt_score_thr
    #                     and kpt[sk[1] - 1, 2] > kpt_score_thr):
    #                 img_copy = img.copy()
    #                 X = (pos1[0], pos2[0])
    #                 Y = (pos1[1], pos2[1])
    #                 mX = np.mean(X)
    #                 mY = np.mean(Y)
    #                 length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
    #                 angle = math.degrees(
    #                     math.atan2(Y[0] - Y[1], X[0] - X[1]))
    #                 stickwidth = 1
    #                 polygon = cv2.ellipse2Poly(
    #                     (int(mX), int(mY)),
    #                     (int(length / 2), int(stickwidth)), int(angle),
    #                     0, 360, 1)
    #
    #                 if flag == 0:
    #                     r, g, b = 0, 255, 0
    #                 elif flag == 1:
    #                     r, g, b = pose_limb_color[sk_id]
    #                 else:
    #                     r, g, b = 0, 0, 255
    #                 cv2.fillConvexPoly(img_copy, polygon,
    #                                    (int(r), int(g), int(b)))
    #                 transparency = transp
    #                 cv2.addWeighted(
    #                     img_copy,
    #                     transparency,
    #                     img,
    #                     1 - transparency,
    #                     0,
    #                     dst=img)
    #     return img


    def draw_box(self, img, bboxes, flag):
        # print(flag)
        if flag == 0:
            r, g, b = 0, 0, 255  # red
        elif flag == 1:
            r, g, b = 255, 0, 0  # blue
        else:
            r, g, b = 0, 255, 255  # yellow
        for bbox in bboxes:
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(img, start_point, end_point, (r, g, b), 1)
        return img


    def cal_iou(self, boxA, boxB):
        xmaxA = boxA[0] + boxA[2]
        ymaxA = boxA[1] + boxA[3]
        xmaxB = boxB[0] + boxB[2]
        ymaxB = boxB[1] + boxB[3]
        left = max(boxA[0], boxB[0])
        top = max(boxA[1], boxB[1])
        right = min(xmaxA, xmaxB)
        bottom = min(ymaxA, ymaxB)
        inter = max(0, right - left) * max(0, bottom - top)
        Sa = boxA[2] * boxA[3]
        Sb = boxB[2] * boxB[3]
        union = Sa + Sb - inter
        iou = inter / union
        return iou


    def computeIou(self, image_id):
        # dimention here should be Nxm
        gt_bbox = self.gt_bboxes[image_id]
        pr_bbox = self.pr_bboxes[image_id]
        if len(gt_bbox) == 0 or len(pr_bbox) == 0:
            return []
        ious = np.zeros((len(pr_bbox), len(gt_bbox)))
        # compute oks between each detection and ground truth object
        for j, gbox in enumerate(gt_bbox):
            # create bounds for ignore regions(double the gt bbox)
            gbox = np.array(gbox)
            for i, dbox in enumerate(pr_bbox):
                dbox = np.array(dbox)
                ious[i, j] = self.cal_iou(gbox, dbox)
        return ious

    def computeOks(self, image_id):
        # dimention here should be Nxm
        gt_bbox = self.gt_bboxes[image_id]
        gt_kpt = self.gt_kpts[image_id]
        pr_bbox = self.pr_bboxes[image_id]
        pr_kpt = self.pr_kpts[image_id]
        area = self.gt_areas[image_id]
        if len(gt_kpt) == 0 or len(pr_kpt) == 0:
            return []
        ious = np.zeros((len(pr_kpt), len(gt_kpt)))
        sigmas = self.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gt_kpt):
            # create bounds for ignore regions(double the gt bbox)
            gt = np.array(gt)
            xg = gt[:, 0]
            yg = gt[:, 1]
            vg = gt[:, 2]
            k1 = np.count_nonzero(vg > 0)
            bb = gt_bbox[j]
            x0 = max(0, bb[0] - bb[2])
            x1 = min(1280, bb[0] + bb[2] * 2)
            y0 = max(0, bb[1] - bb[3])
            y1 = min(720, bb[1] + bb[3] * 2)
            for i, dt in enumerate(pr_kpt):
                dt = np.array(dt)
                xd = dt[:, 0]
                yd = dt[:, 1]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx ** 2 + dy ** 2) / vars / (area[j] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious


    def min_cost_matching(self, distance_metric, img_id, max_distance, flag):
        if flag == 'bbox':
            gt_boxes = self.gt_bboxes[img_id]
            pr_boxes = self.pr_bboxes[img_id]
        else:
            gt_boxes = self.gt_kpts[img_id]
            pr_boxes = self.pr_kpts[img_id]
        matches, undetected_boxes, false_boxes = [], [], []
        if len(gt_boxes) == 0 or len(pr_boxes) == 0:
            if len(gt_boxes) != 0:
                for i, fbox in enumerate(gt_boxes):
                    undetected_boxes.append(fbox)
            if len(pr_boxes) != 0:
                for i, bbox in enumerate(pr_boxes):
                    false_boxes.append(bbox)
            return matches, undetected_boxes, false_boxes  # Nothing to match.

        cost_matrix = distance_metric(img_id)
        cost_matrix = np.array(make_cost_matrix(cost_matrix))
        cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

        row_indices, col_indices = linear_assignment(cost_matrix)

        for col, bbox in enumerate(gt_boxes):
            if col not in col_indices:
                undetected_boxes.append(bbox)  # gt undetected
        for row, fbox in enumerate(pr_boxes):
            if row not in row_indices:
                false_boxes.append(fbox)  # pr false
        for row, col in zip(row_indices, col_indices):
            fbox = pr_boxes[row]
            if cost_matrix[row, col] > max_distance:
                false_boxes.append(fbox)  # false
            else:
                matches.append(fbox)
        return matches, undetected_boxes, false_boxes

    def eval(self):
        TP_boxes = 0
        false_box = 0
        undetected_box = 0
        TP_kpt = 0
        false_kpt = 0
        undetected_kpt = 0
        len_gt_bboxes = np.array([len(self.gt_bboxes[bboxes_id]) for bboxes_id in self.gt_bboxes.keys()]).sum()
        len_pr_bboxes = np.array([len(self.pr_bboxes[bboxes_id]) for bboxes_id in self.pr_bboxes.keys()]).sum()
        for img_id in range(len(self.gt_bboxes)):
            if img_id % 100 == 0:
                print(img_id)
            file_name = self.file_names[img_id]
            if file_name == '06-30_10_13_46_765.jpg':
                print(1)
            image_path = os.path.join(self.image_dir, file_name)
            img = cv2.imread(image_path)

            gt_bbox = self.gt_bboxes[img_id]
            if img_id not in self.pr_bboxes.keys():
                img = self.vis_keypoints(img, np.array(self.gt_kpts[img_id]), 0)  # draw ground truth
                img = self.draw_box(img, gt_bbox, 2)
                cv2.imwrite(os.path.join(self.vis_dir, file_name), img)
                continue

            img = self.vis_keypoints(img, np.array(self.gt_kpts[img_id]), 0)  # draw ground truth
            matches_boxes, undetected_boxes, false_boxes = self.min_cost_matching(self.computeIou, img_id, self.iou_thre_bbox, 'bbox')
            TP_boxes += len(matches_boxes)
            false_box += len(false_boxes)
            undetected_box += len(undetected_boxes)
            if self.write_image:
                img = self.draw_box(img, matches_boxes, 1)
                img = self.draw_box(img, undetected_boxes, 2)
                img = self.draw_box(img, false_boxes, 0)
            matches_kps, undetected_kps, false_kps = self.min_cost_matching(self.computeOks, img_id, self.iou_thre_kpt, 'kpt')
            TP_kpt += len(matches_kps)
            false_kpt += len(false_kps)
            undetected_kpt += len(undetected_kps)
            if self.write_image:
                img = self.vis_keypoints(img, np.array(matches_kps), 1)
                img = self.vis_keypoints(img, np.array(false_kps), 2)
                cv2.imwrite(os.path.join(self.vis_dir, file_name), img)
        AP_boxes = TP_boxes / len_pr_bboxes
        AR_boxes = TP_boxes / len_gt_bboxes
        AP_kps = TP_kpt / len_pr_bboxes
        AR_kps = TP_kpt / len_gt_bboxes
        return AP_boxes, AR_boxes, AP_kps, AR_kps