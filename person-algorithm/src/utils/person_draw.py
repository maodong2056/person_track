import cv2
import math
import enum
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import PersonItem, PartName, KeyPointType
from src.utils.pose_utils import cam2pixel
from src.utils.bbox import xcycwh2xyxy



base_color = np.array([154, 250, 0]).astype(np.uint8)
face_base_color = np.array([133, 21, 199]).astype(np.uint8)
lh_base_color = np.array([154, 250, 0]).astype(np.uint8)
rh_base_color = np.array([225, 105, 65]).astype(np.uint8)
color = (np.random.random((18, 3)) * 255).astype(np.uint8)
lm_color = np.array([[0, 0, 255], [0, 255, 255],
                     [255, 0, 255], [0, 255, 0],
                     [255, 0, 0]]).astype(np.uint8)

def fig2data(fig):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGB", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def draw_person_bbox(image,
                     person_items: list,
                     draw_face=True,
                     draw_hand=True,
                     draw_lm=True,
                     draw_id=True,
                     draw_conf=False,
                     draw_tracking=False):
    def do_draw(return_image, person_item):
        # get color
        if draw_id and person_item.get_track_id() is not None:
            body_color = color[int(person_item.get_track_id() % len(color))]
        else:
            body_color = color[int(idx % len(color))]
        # face_color = body_color
        face_color = (body_color * 0.7 + face_base_color * 0.3).astype(np.uint8)
        lh_color = (body_color * 0.7 + lh_base_color * 0.3).astype(np.uint8)
        rh_color = (body_color * 0.7 + rh_base_color * 0.3).astype(np.uint8)
        # draw body
        x1, y1, x2, y2 = person_item.get_box(PartName.body_part)
        cv2.rectangle(return_image, (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      body_color.tolist(), 2, 1)
        if draw_tracking:
            track_state = person_item.get_track_state()
            if track_state:
                w = x2 - x1
                h = y2 - y1
                new_x1 = x1 + 0.2 * w
                new_y1 = y1 + 0.2 * h
                new_x2 = x2 - 0.2 * w
                new_y2 = y2 - 0.2 * h
                cv2.rectangle(return_image, (int(new_x1), int(new_y1)),
                              (int(new_x2), int(new_y2)),
                              [0, 255, 0], 2, 1)
        if draw_conf:
            conf_str = "{:.3f}".format(person_item.get_conf(PartName.body_part))
            x_s, y_s = cv2.getTextSize(conf_str,
                                     cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(return_image, (int(x1), int(y1 - y_s - 4)),
                          (int(x1) + x_s + 3, int(y1)),
                          body_color.tolist(), -1)
            cv2.putText(return_image, conf_str,
                        (int(x1), int(y1) - 2),
                        cv2.FONT_HERSHEY_PLAIN, 1,
                        [255, 255, 255], 1)
        if draw_id and person_item.get_track_id() is not None:
            track_id_str = str(person_item.get_track_id())
            x_s, y_s = cv2.getTextSize(track_id_str,
                                     cv2.FONT_HERSHEY_PLAIN, 1.2, 2)[0]
            cv2.rectangle(return_image, (int(x1), int(y1)),
                          (int(x1) + x_s + 3, int(y1 + y_s + 4)),
                          body_color.tolist(), -1)
            cv2.putText(return_image, track_id_str,
                        (int(x1), int(y1 + 14)),
                        cv2.FONT_HERSHEY_PLAIN, 1.2,
                        [0, 0, 255], 2)
        # draw face
        if draw_face and person_item.get_part_state(PartName.face_part):
            x1, y1, x2, y2 = person_item.get_box(PartName.face_part)
            cv2.rectangle(return_image, (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          face_color.tolist(), 2, 1)
            if draw_conf:
                conf_str = "{:.3f}".format(person_item.get_conf(PartName.face_part))
                x_s, y_s = cv2.getTextSize(conf_str,
                                           cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(return_image, (int(x1), int(y1 - y_s - 4)),
                              (int(x1) + x_s + 3, int(y1)),
                              body_color.tolist(), -1)
                cv2.putText(return_image, conf_str,
                            (int(x1), int(y1) - 2),
                            cv2.FONT_HERSHEY_PLAIN, 1,
                            [255, 255, 255], 1)
            if draw_lm:
                lm = person_item.get_lm(PartName.face_part)
                for p, c in zip(lm, lm_color[:len(lm)]):
                    cv2.circle(return_image, (int(p[0]), int(p[1])), 1, c.tolist(), 4)
        # draw hand
        if draw_hand:
            # left hand
            if person_item.get_part_state(PartName.lefthand_part):
                x1, y1, x2, y2 = person_item.get_box(PartName.lefthand_part)
                cv2.rectangle(return_image, (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              lh_color.tolist(), 2, 1)
                if draw_conf:
                    conf_str = "{:.3f}".format(person_item.get_conf(PartName.lefthand_part))
                    x_s, y_s = cv2.getTextSize(conf_str,
                                               cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.rectangle(return_image, (int(x1), int(y1 - y_s - 4)),
                                  (int(x1) + x_s + 3, int(y1)),
                                  body_color.tolist(), -1)
                    cv2.putText(return_image, conf_str,
                                (int(x1), int(y1) - 2),
                                cv2.FONT_HERSHEY_PLAIN, 1,
                                [255, 255, 255], 1)
                if draw_lm:
                    lm = person_item.get_lm(PartName.lefthand_part)
                    for p, c in zip(lm, lm_color[:len(lm)]):
                        cv2.circle(return_image, (int(p[0]), int(p[1])), 1, c.tolist(), 4)
            # right hand
            if person_item.get_part_state(PartName.righthand_part):
                x1, y1, x2, y2 = person_item.get_box(PartName.righthand_part)
                cv2.rectangle(return_image, (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              rh_color.tolist(), 2, 1)
                if draw_conf:
                    conf_str = "{:.3f}".format(person_item.get_conf(PartName.righthand_part))
                    x_s, y_s = cv2.getTextSize(conf_str,
                                               cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.rectangle(return_image, (int(x1), int(y1 - y_s - 4)),
                                  (int(x1) + x_s + 3, int(y1)),
                                  body_color.tolist(), -1)
                    cv2.putText(return_image, conf_str,
                                (int(x1), int(y1) - 2),
                                cv2.FONT_HERSHEY_PLAIN, 1,
                                [255, 255, 255], 1)
                if draw_lm:
                    lm = person_item.get_lm(PartName.righthand_part)
                    for p, c in zip(lm, lm_color[:len(lm)]):
                        cv2.circle(return_image, (int(p[0]), int(p[1])), 1, c.tolist(), 4)
    return_image = image.copy()
    for idx, person_item in enumerate(person_items):
        if draw_tracking:
            track_state = person_item.get_track_state()
            if track_state:
                do_draw(return_image, person_item)
            else:
                continue
        else:
            do_draw(return_image, person_item)
    return return_image



def draw_keypoints(image,
                   person_items: list,
                   person_part,
                   keypoint_limb,
                   limb_color,
                   keypoint_color,
                   vis_thres: float):
    def draw_kps(img, kpt, color):
        x_coord, y_coord, kpt_score = kpt
        if kpt_score > vis_thres:
            # img_copy = img.copy()
            b, g, r = color
            cv2.circle(img, (int(x_coord), int(y_coord)),
                       4, (int(b), int(g), int(r)), -1)
            # transparency = max(0, min(1, kpt_score))
            # cv2.addWeighted(
            #     img_copy,
            #     transparency,
            #     img,
            #     1 - transparency,
            #     0,
            #     dst=img)
        return img
    def draw_skeleton(img, pos1, pos2, color, img_w, img_h):
        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                and pos1[1] < img_h and pos2[0] > 0
                and pos2[0] < img_w and pos2[1] > 0
                and pos2[1] < img_h
                and pos1[2] > vis_thres
                and pos2[2] > vis_thres):
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

            b, g, r = color
            cv2.fillConvexPoly(img, polygon,
                               (int(b), int(g), int(r)))
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

    img_h, img_w, _ = image.shape
    return_image = image.copy()
    for idx, person_item in enumerate(person_items):
        kpts = person_item.get_keypoint(person_part, keypoint_type=KeyPointType.Kps2D)
        for sk_id, sk in enumerate(keypoint_limb):
            pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]), kpts[sk[0], 2])
            pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]), kpts[sk[1], 2])
            return_image = draw_kps(return_image, pos1, keypoint_color[sk[0]])
            return_image = draw_kps(return_image, pos2, keypoint_color[sk[1]])
            return_image = draw_skeleton(return_image, pos1, pos2, limb_color[sk_id], img_w, img_h)
    return return_image


def draw_keypoints_3d(person_items: list,
                      person_part,
                      keypoint_limb,
                      vis_thres: float,
                      filename=None,
                      z_depth=0,
                      elev=30,
                      azim=-60):

    def draw_3d_kpts(ax, kpt_3d):
        for l in range(len(keypoint_limb)):
            i1 = keypoint_limb[l][0]
            i2 = keypoint_limb[l][1]
            x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
            y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
            z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

            if kpt_3d[i1, 3] > vis_thres and kpt_3d[i2, 3] > vis_thres:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d[i1, 3] > vis_thres:
                ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker='o')
            if kpt_3d[i2, 3] > vis_thres:
                ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker='o')
    # draw 3d kps
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(keypoint_limb) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    for person_item in person_items:
        kpts_3d = person_item.get_keypoint(person_part, keypoint_type=KeyPointType.Kps3D).copy()
        kpts_3d[:, 1] = kpts_3d[:, 1] - kpts_3d[:, 1].max()
        draw_3d_kpts(ax, kpts_3d)


    # for l in range(len(keypoint_limb)):
    #     i1 = keypoint_limb[l][0]
    #     i2 = keypoint_limb[l][1]
    #
    #     person_num = kpt_3d.shape[0]
    #     for n in range(person_num):
    #         x = np.array([kpt_3d[n, i1, 0], kpt_3d[n, i2, 0]])
    #         y = np.array([kpt_3d[n, i1, 1], kpt_3d[n, i2, 1]])
    #         z = np.array([kpt_3d[n, i1, 2], kpt_3d[n, i2, 2]])
    #
    #         if kpt_3d[n, i1, 3] > conf_thres and kpt_3d[n, i2, 3] > conf_thres:
    #             ax.plot(x, z, -y, c=colors[l], linewidth=2)
    #         if kpt_3d[n, i1, 3] > conf_thres:
    #             ax.scatter(kpt_3d[n, i1, 0], kpt_3d[n, i1, 2], -kpt_3d[n, i1, 1], c=colors[l], marker='o')
    #         if kpt_3d[n, i2, 3] > conf_thres:
    #             ax.scatter(kpt_3d[n, i2, 0], kpt_3d[n, i2, 2], -kpt_3d[n, i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(z_depth - 1000, z_depth + 1000)
    ax.set_zlim(0, 2000)
    # ax.legend()

    image = fig2data(fig)
    plt.close(fig)
    return image




class HandLandmark(enum.IntEnum):
    """The 21 hand landmarks."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


HAND_CONNECTIONS = frozenset([
    (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
    (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
    (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    (HandLandmark.WRIST, HandLandmark.INDEX_FINGER_MCP),
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_PIP),
    (HandLandmark.INDEX_FINGER_PIP, HandLandmark.INDEX_FINGER_DIP),
    (HandLandmark.INDEX_FINGER_DIP, HandLandmark.INDEX_FINGER_TIP),
    (HandLandmark.INDEX_FINGER_MCP, HandLandmark.MIDDLE_FINGER_MCP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_PIP),
    (HandLandmark.MIDDLE_FINGER_PIP, HandLandmark.MIDDLE_FINGER_DIP),
    (HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.MIDDLE_FINGER_TIP),
    (HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.RING_FINGER_MCP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_PIP),
    (HandLandmark.RING_FINGER_PIP, HandLandmark.RING_FINGER_DIP),
    (HandLandmark.RING_FINGER_DIP, HandLandmark.RING_FINGER_TIP),
    (HandLandmark.RING_FINGER_MCP, HandLandmark.PINKY_MCP),
    (HandLandmark.WRIST, HandLandmark.PINKY_MCP),
    (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP)
])

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

skeleton_color = (255, 0, 102)
keypoint_color = (0, 102, 255)
bbox_color = (255, 0, 255)

def draw_hand(image,
             hand_items: list,
             draw_hand_box=True,
             draw_hand_kps=True,
             draw_hand_roi=True,
              is_norm = True,
              tracking = False
              ):
    track_state = []
    for hand_item in hand_items:
        hand_box = hand_item._hand_box
        hand_conf = hand_item._hand_conf
        hand_lm = hand_item._hand_lm
        hand_keypoint = hand_item._hand_keypoint
        hand_roi = hand_item._hand_roi
        hand_tracking = hand_item._hand_tracking_state#是０的话就没有启动跟踪

        track_state.append(hand_tracking)
        if draw_hand_box and hand_box !=[] and not tracking :
            image = draw_hand_bbox(image, hand_box, is_norm)
        if draw_hand_kps and hand_keypoint is not None:
            image = draw_hand_landmark(image, hand_keypoint, is_norm)
        if draw_hand_roi and hand_roi is not None:
            image = draw_hand_bbox(image, xcycwh2xyxy(hand_roi[None, :4])[0], is_norm)

    tracking = (np.array(track_state)==1).all()
    if track_state == []:
        tracking = False # 进行修正
    # print(f'3hand_tracking2:{track_state}')
    return image, tracking


def draw_hand_landmark(img, landmarks, is_norm=False):
    img_h, img_w, _ = img.shape
    if is_norm:
        kpts = landmarks * np.array([img_w, img_h, img_w])
    else:
        kpts = landmarks

    for connect in HAND_CONNECTIONS:
        pos1 = (int(kpts[connect[0], 0]), int(kpts[connect[0], 1]))
        pos2 = (int(kpts[connect[1], 0]), int(kpts[connect[1], 1]))
        cv2.line(img, pos1, pos2, skeleton_color, thickness=2, lineType=cv2.LINE_AA)
    for kid, kpt in enumerate(kpts):
        x_coord, y_coord, _ = int(kpt[0]), int(kpt[1]), kpt[2]
        cv2.circle(img, (int(x_coord), int(y_coord)),
                   4, keypoint_color, -1)

    return img


def draw_hand_bbox(img_orial, bbox, is_norm=False):
    img = img_orial.copy()
    img_h, img_w, _ = img.shape
    if is_norm:
        this_bboxes = bbox * np.array([img_w, img_h, img_w, img_h])
    else:
        this_bboxes = bbox
    cv2.rectangle(img, (int(this_bboxes[0]), int(this_bboxes[1])),(int(this_bboxes[2]), int(this_bboxes[3])),bbox_color, 2, 1)

    return img


def draw_hand_rotate_rect(img, rotate_rect, is_norm=False):
    img_h, img_w, _ = img.shape
    if is_norm:
        this_rect = rotate_rect * np.array([img_w, img_h, img_w, img_h, 180 / math.pi])
    else:
        this_rect = rotate_rect * np.array([1.0, 1.0, 1.0, 1.0, 180 / math.pi])
    rect_tuple = ((this_rect[0], this_rect[1]), (this_rect[2], this_rect[3]), this_rect[4])
    points = cv2.boxPoints(rect_tuple).astype(np.int)
    # print(points)
    point1 = tuple(points[0])
    point2 = tuple(points[1])
    point3 = tuple(points[2])
    point4 = tuple(points[3])
    cv2.line(img, point1, point2, bbox_color, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(img, point2, point3, bbox_color, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(img, point3, point4, bbox_color, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(img, point4, point1, bbox_color, thickness=2, lineType=cv2.LINE_AA)
    return img


def draw_hand_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (int(point[0]), int(point[1])), 1+int(index/2), (152, 251, 152), 2)
    return image