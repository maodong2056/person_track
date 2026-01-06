import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from matplotlib.axes._axes import _log as matplotlib_axes_logger

matplotlib_axes_logger.setLevel('ERROR')
joint_num = 18 # original:17, but manually added 'Thorax'
joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )

def vis_keypoints(img, kps, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

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
            cv2.putText(kp_mask, "{}".format(i1), p1, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
        if kps[i2, 2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(kp_mask, "{}".format(i2), p2, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)


def vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n, i1, 0], kpt_3d[n, i2, 0]])
            y = np.array([kpt_3d[n, i1, 1], kpt_3d[n, i2, 1]])
            z = np.array([kpt_3d[n, i1, 2], kpt_3d[n, i2, 2]])

            if kpt_3d_vis[n, i1, 0] > 0 and kpt_3d_vis[n, i2, 0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n, i1, 0] > 0:
                ax.scatter(kpt_3d[n, i1, 0], kpt_3d[n, i1, 2], -kpt_3d[n, i1, 1], c=colors[l], marker='o')
            if kpt_3d_vis[n, i2, 0] > 0:
                ax.scatter(kpt_3d[n, i2, 0], kpt_3d[n, i2, 2], -kpt_3d[n, i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    # ax.set_xlim(-750, 750)
    # ax.set_ylim(1000, 5000)
    # ax.set_zlim(-750, 750)
    ax.legend()

    plt.show()
    cv2.waitKey(0)


def vis_3d_multiple_skeleton_toimage(kpt_3d, kpt_3d_vis, kps_lines, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n, i1, 0], kpt_3d[n, i2, 0]])
            y = np.array([kpt_3d[n, i1, 1], kpt_3d[n, i2, 1]])
            z = np.array([kpt_3d[n, i1, 2], kpt_3d[n, i2, 2]])

            if kpt_3d_vis[n, i1, 0] > 0 and kpt_3d_vis[n, i2, 0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n, i1, 0] > 0:
                ax.scatter(kpt_3d[n, i1, 0], kpt_3d[n, i1, 2], -kpt_3d[n, i1, 1], c=colors[l], marker='o')
            if kpt_3d_vis[n, i2, 0] > 0:
                ax.scatter(kpt_3d[n, i2, 0], kpt_3d[n, i2, 2], -kpt_3d[n, i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.set_xlim(-750, 750)
    ax.set_ylim(1000, 5000)
    ax.set_zlim(-750, 750)
    # ax.legend()

    image = fig2data(fig)
    plt.close(fig)
    return image


def vis_3d_multiple_skeleton_conf_toimage(kpt_3d, conf_thres=0.2, filename=None):
    fig = plt.figure(figsize=(8, 8), dpi=90)
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(skeleton)):
        i1 = skeleton[l][0]
        i2 = skeleton[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n, i1, 0], kpt_3d[n, i2, 0]])
            y = np.array([kpt_3d[n, i1, 1], kpt_3d[n, i2, 1]])
            z = np.array([kpt_3d[n, i1, 2], kpt_3d[n, i2, 2]])

            if kpt_3d[n, i1, 3] > conf_thres and kpt_3d[n, i2, 3] > conf_thres:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d[n, i1, 3] > conf_thres:
                ax.scatter(kpt_3d[n, i1, 0], kpt_3d[n, i1, 2], -kpt_3d[n, i1, 1], c=colors[l], marker='o')
            if kpt_3d[n, i2, 3] > conf_thres:
                ax.scatter(kpt_3d[n, i2, 0], kpt_3d[n, i2, 2], -kpt_3d[n, i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.set_xlim(0, 3000)
    ax.set_ylim(1000, 5000)
    ax.set_zlim(-750, 1200)
    # ax.legend()

    image = fig2data(fig)
    plt.close(fig)
    return image


def vis_3d_multiple_skeleton_conf_addpoint_toimage(kpt_3d, point_p, conf_thres=0.2, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    ax.scatter(point_p[0, 0], point_p[0, 2], -point_p[0, 1], c=np.array([0., 0., 1.]), marker='o')
    for l in range(len(skeleton)):
        i1 = skeleton[l][0]
        i2 = skeleton[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n, i1, 0], kpt_3d[n, i2, 0]])
            y = np.array([kpt_3d[n, i1, 1], kpt_3d[n, i2, 1]])
            z = np.array([kpt_3d[n, i1, 2], kpt_3d[n, i2, 2]])

            if kpt_3d[n, i1, 3] > conf_thres and kpt_3d[n, i2, 3] > conf_thres:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d[n, i1, 3] > conf_thres:
                ax.scatter(kpt_3d[n, i1, 0], kpt_3d[n, i1, 2], -kpt_3d[n, i1, 1], c=colors[l], marker='o')
            if kpt_3d[n, i2, 3] > conf_thres:
                ax.scatter(kpt_3d[n, i2, 0], kpt_3d[n, i2, 2], -kpt_3d[n, i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.set_xlim(-500, 2000)
    ax.set_ylim(1000, 5000)
    ax.set_zlim(-750, 1200)
    # ax.legend()

    image = fig2data(fig)
    plt.close(fig)
    return image

def fig2data(fig):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGB", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def vis_root_3d_multiple_skeleton_conf_toimage(kpt_3d, conf_thres=0.2, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(skeleton) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(skeleton)):
        i1 = skeleton[l][0]
        i2 = skeleton[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n, i1, 0], kpt_3d[n, i2, 0]])
            y = np.array([kpt_3d[n, i1, 1], kpt_3d[n, i2, 1]])
            z = np.array([kpt_3d[n, i1, 2], kpt_3d[n, i2, 2]])

            if kpt_3d[n, i1, 3] > conf_thres and kpt_3d[n, i2, 3] > conf_thres:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d[n, i1, 3] > conf_thres:
                ax.scatter(kpt_3d[n, i1, 0], kpt_3d[n, i1, 2], -kpt_3d[n, i1, 1], c=colors[l], marker='o')
            if kpt_3d[n, i2, 3] > conf_thres:
                ax.scatter(kpt_3d[n, i2, 0], kpt_3d[n, i2, 2], -kpt_3d[n, i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.set_xlim(-1800, 800)
    ax.set_ylim(-1000, 4000)
    ax.set_zlim(-10, 1600)
    # ax.set_xlim(-800, 800)
    # ax.set_ylim(-2000, 7000)
    # ax.set_zlim(-800, 800)
   # ax.legend()

    image = fig2data(fig)
    plt.close(fig)
    return image
