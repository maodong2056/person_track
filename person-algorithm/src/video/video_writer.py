import os
import cv2
import datetime
import logging
from src.utils import caller_class
logger = logging.getLogger(__name__)

@caller_class
class VideoWriter(object):
    def __init__(self, exp_name, tag, size=(640, 480), fps=15, fourcc="mp4v", ex_name="avi", filename=None, *args, **kwargs):
        time = datetime.datetime.now()
        self.time = "{}_{}_{}_{}_{}".format(time.year, time.month, time.day,
                                                time.hour, time.minute)
        self.exp_name = exp_name
        output_base_dir = os.path.join("user/output", filename)
        if not os.path.exists(output_base_dir):
            os.mkdir(output_base_dir)
        output_exp_dir = os.path.join(output_base_dir, self.exp_name)
        if not os.path.exists(output_exp_dir):
            os.mkdir(output_exp_dir)
        self.tag = tag
        self.size = size
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.video_name = "{}_{}.{}".format(self.time, tag, ex_name)
        self.output_video = os.path.join(output_exp_dir, self.video_name)
        # self.writer = cv2.VideoWriter(self.output_video, self.fourcc, self.fps, self.size)
        self.writer = None
        logger.info("Write video to {}".format(self.output_video))
        self.count = 0

    def write_frame(self, frame):
        if self.writer is None:
            size = (frame.shape[1], frame.shape[0])
            self.size = size
            self.writer = cv2.VideoWriter(self.output_video, self.fourcc, self.fps, self.size)
        assert frame.shape[0] == self.size[1] and frame.shape[1] == self.size[0], \
            "video size: {}, frame size: {}".format(self.size, frame.shape)
        self.writer.write(frame)
        self.count += 1

    def end_write(self):
        self.writer.release()
        logger.info("Write {} frames to {}".format(self.count, self.output_video))
