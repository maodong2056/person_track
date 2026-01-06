# @Time : 2021/3/18 15:16 
# @Author : Altair.Huazj
# @File : draw_kps.py 
# @Software: PyCharm
import cv2
import numpy as np
import math
# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
skeleton = [[1, 2], [2, 3], [3, 4], [9, 15], [15, 16], [16, 17],
            [1, 5], [5, 6], [6, 7], [9, 12], [12, 13], [13, 14],
            [1, 8], [8, 9], [9, 10], [10, 11]]
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
# pose_limb_color = palette[[
#     0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 0, 4, 8, 12, 16
# ]]
pose_limb_color = palette[[0, 0, 0, 0, 0, 0,
                           2, 2, 2, 2, 2, 2,
                           4, 4, 4, 4]]
# pose_kpt_color = palette[[
#     0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16,
#     0
# ]]
pose_kpt_color = palette[[4, 1, 1, 1,
                         3, 3, 3, 4, 4, 4, 4,
                         3, 3, 3, 1, 1, 1]]
colorNotChange = [[13, 12], [12, 13]]  # In pose match with Angle measure, color of some point not change, leia 20210802

def draw_hkp(img, kpts, kpt_score_thr=0.3):
    img_h, img_w, _ = img.shape
    for kid, kpt in enumerate(kpts):
        x_coord, y_coord, kpt_score = int(kpt[0]), int(
            kpt[1]), kpt[2]
        if kpt_score > kpt_score_thr:
            img_copy = img.copy()
            r, g, b = pose_kpt_color[kid]
            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                       4, (int(r), int(g), int(b)), -1)
            transparency = max(0, min(1, kpt_score))
            cv2.addWeighted(
                img_copy,
                transparency,
                img,
                1 - transparency,
                0,
                dst=img)
    for sk_id, sk in enumerate(skeleton):
        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                  1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                  1]))
        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                and pos1[1] < img_h and pos2[0] > 0
                and pos2[0] < img_w and pos2[1] > 0
                and pos2[1] < img_h
                and kpts[sk[0] - 1, 2] > kpt_score_thr
                and kpts[sk[1] - 1, 2] > kpt_score_thr):
            img_copy = img.copy()
            X = (pos1[0], pos2[0])
            Y = (pos1[1], pos2[1])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(
                math.atan2(Y[0] - Y[1], X[0] - X[1]))
            stickwidth = 2
            polygon = cv2.ellipse2Poly(
                (int(mX), int(mY)),
                (int(length / 2), int(stickwidth)), int(angle),
                0, 360, 1)

            r, g, b = pose_limb_color[sk_id]
            cv2.fillConvexPoly(img_copy, polygon,
                               (int(r), int(g), int(b)))
            transparency = max(
                0,
                min(
                    1, 0.5 *
                       (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
            cv2.addWeighted(
                img_copy,
                transparency,
                img,
                1 - transparency,
                0,
                dst=img)
    return img

def draw_hkp_simple(img, kpts, kpt_score_thr=0.05):
    img_h, img_w, _ = img.shape
    for kid, kpt in enumerate(kpts):
        x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
        if kpt_score > kpt_score_thr:
            # img_copy = img.copy()
            r, g, b = pose_kpt_color[kid]
            cv2.circle(img, (int(x_coord), int(y_coord)),
                       4, (int(r), int(g), int(b)), -1)
            # transparency = max(0, min(1, kpt_score))
            # cv2.addWeighted(
            #     img_copy,
            #     transparency,
            #     img,
            #     1 - transparency,
            #     0,
            #     dst=img)
    for sk_id, sk in enumerate(skeleton):
        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                  1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                  1]))
        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                and pos1[1] < img_h and pos2[0] > 0
                and pos2[0] < img_w and pos2[1] > 0
                and pos2[1] < img_h
                and kpts[sk[0] - 1, 2] > kpt_score_thr
                and kpts[sk[1] - 1, 2] > kpt_score_thr):
            # img_copy = img.copy()
            X = (pos1[0], pos2[0])
            Y = (pos1[1], pos2[1])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(
                math.atan2(Y[0] - Y[1], X[0] - X[1]))
            stickwidth = 2
            polygon = cv2.ellipse2Poly(
                (int(mX), int(mY)),
                (int(length / 2), int(stickwidth)), int(angle),
                0, 360, 1)

            r, g, b = pose_limb_color[sk_id]
            cv2.fillConvexPoly(img, polygon,
                               (int(r), int(g), int(b)))
            # transparency = max(
            #     0,
            #     min(
            #         1, 0.5 *
            #            (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
            # cv2.addWeighted(
            #     img_copy,
            #     transparency,
            #     img,
            #     1 - transparency,
            #     0,
            #     dst=img)
    return img


def draw_hkp_with_angle_wrong(img, kpts, angle_wrong, wrong_skeleton, kpt_score_thr=0.05):
    img_h, img_w, _ = img.shape
    idx_count = 1
    for kid, kpt in enumerate(kpts):
        x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
        if kpt_score > kpt_score_thr:
            if idx_count in angle_wrong or idx_count - 2 in angle_wrong:
                r, g, b = 0, 0, 255
                cv2.circle(img, (int(x_coord), int(y_coord)),
                           10, (int(r), int(g), int(b)), -1)
            else:
                # r, g, b = pose_kpt_color[kid]
                r, g, b = 255, 0, 0
                cv2.circle(img, (int(x_coord), int(y_coord)),
                           4, (int(r), int(g), int(b)), -1)
        idx_count = idx_count + 1

    for sk_id, sk in enumerate(skeleton):
        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                and pos1[1] < img_h and pos2[0] > 0
                and pos2[0] < img_w and pos2[1] > 0
                and pos2[1] < img_h
                and kpts[sk[0] - 1, 2] > kpt_score_thr
                and kpts[sk[1] - 1, 2] > kpt_score_thr):
            # img_copy = img.copy()
            X = (pos1[0], pos2[0])
            Y = (pos1[1], pos2[1])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(
                math.atan2(Y[0] - Y[1], X[0] - X[1]))
            stickwidth = 2
            polygon = cv2.ellipse2Poly(
                (int(mX), int(mY)),
                (int(length / 2), int(stickwidth)), int(angle),
                0, 360, 1)

            # if (sk[1] in angle_wrong) and (sk[1] in colorChange):
            if ([sk[0], sk[1]] in wrong_skeleton) or ([sk[1], sk[0]] in wrong_skeleton):
                if [sk[0], sk[1]] not in colorNotChange:
                    r, g, b = 0, 0, 255
            else:
                # r, g, b = pose_limb_color[sk_id]
                r, g, b = 255, 0, 0
            cv2.fillConvexPoly(img, polygon,
                               (int(r), int(g), int(b)))
    return img