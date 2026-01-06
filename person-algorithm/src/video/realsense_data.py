import cv2
import pyrealsense2 as rs
import numpy as np

class RealSenceVideo():
    def __init__(self, video_width=None, video_height=None, video_fps=None, *args, **kwargs):
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.depth, video_width, video_height, rs.format.z16, video_fps)
        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, video_width, video_height, rs.format.bgr8, video_fps)
        else:
            config.enable_stream(rs.stream.color, video_width, video_height, rs.format.bgr8, video_fps)
        profile = self.pipeline.start(config)
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)
        #  clipping_distance_in_meters meters away
        # clipping_distance_in_meters = 1  # 1 meter
        # clipping_distance = clipping_distance_in_meters / self.depth_scale
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        # point_cloud = rs.pointcloud()

    def capOneFrame(self):
        try:
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            return depth_image, color_image
        except:
            return None, None