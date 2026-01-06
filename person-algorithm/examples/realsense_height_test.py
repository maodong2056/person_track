import cv2
import pyrealsense2 as rs
import numpy as np
from src.algorithm.api import PersonDetection
from src.utils import PartName, KeyPointType
from src.utils import draw_person_bbox, draw_keypoints, draw_keypoints_3d
pipeline = rs.pipeline()
config = rs.config()

aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
profile = pipeline.start(config)
# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)
person_det = PersonDetection("user/settings/model/detection/body_detection/centernet.json")

while 1:
    frames = pipeline.wait_for_frames()
    frames = aligned_stream.process(frames)
    depth_frame = frames.get_depth_frame()
    points = point_cloud.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 1280, 3)    # xyz
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(frames.get_data())
    input_image = np.asanyarray(color_frame.get_data())
    output = person_det.get_output(input_image)
    if len(output) != 0:
        bbox = output[0].get_box(PartName.body_part)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        obj_points = verts[int(bbox[1]):int(bbox[3]),
                        int(bbox[0]):int(bbox[2])].reshape(-1, 3)
        obj_points_z = verts[int(bbox[1] + h/4):int(bbox[3] - h/2),
                        int(bbox[0] + w/3):int(bbox[2] - w/3)].reshape(-1, 3)
        crop_image = input_image[int(bbox[1] + h/4):int(bbox[3] - h/2),
                            int(bbox[0] + w/3):int(bbox[2] - w/3)]
        zs = obj_points[:, 2]
        ys = obj_points[:, 1]
        z = np.median(obj_points_z[:, 2])
        ys = np.delete(ys, np.where((zs < z - 1) | (zs > z + 1)))
        my = np.amin(ys, initial=1)
        My = np.amax(ys, initial=-1)
        height = (My - my)
        print(height)
        cv2.imshow("crop_im", crop_image)
    image = draw_person_bbox(input_image, output, draw_conf=True)
    cv2.imshow("im", image)
    cv2.waitKey(1)
    # print(1)