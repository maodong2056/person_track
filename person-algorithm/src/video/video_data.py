import cv2


class Video():
    def __init__(self, mode, name, video_width=None, video_height=None, video_fps=None, *args, **kwargs):
        # mode True: camera; mode False: video
        self.mode = mode
        self.name = name  # 视频名称
        self.video_width = video_width
        self.video_height = video_height  # 采集分辨率
        self.video_fps = video_fps  # 采集帧率
        self.cap = self.initCamera() if mode else self.initVideo()
        self.wholeFrameNum = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) if not mode else 0
        self.nowFrame = 0
        self.frameRate = self.cap.get(cv2.CAP_PROP_FPS) if not mode else 15

    def initCamera(self):
        cap = cv2.VideoCapture(int(self.name))
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        if self.video_width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        if self.video_height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        # if self.video_width is not None:
        #     cap.set(cv2.CAP_PROP_FPS, self.video_fps)
        return cap

    def initVideo(self):
        return cv2.VideoCapture(self.name)

    def capOneFrame(self):
        ret, frame = self.cap.read()
        if ret:
            self.nowFrame += 1
        return ret, frame

    def retrieveOneFrame(self):
        self.cap.grab()
        ret, frame = self.cap.retrieve()
        return ret, frame

    def setFramePos(self, percent):
        assert not self.mode
        frame_num = int(self.wholeFrameNum * percent)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.nowFrame = frame_num

    def getFrameID(self):
        return self.nowFrame

    def getFrameRate(self):
        return self.frameRate

    def isCameraConnected(self):
        return self.cap.isOpened()

    def disconnectCamera(self):
        if self.cap.isOpened():
            self.cap.release()
            return True
        else:
            return False


if __name__ == '__main__':
    cap = Video(1, r"C:\Users\Altair\Desktop\0380_2020-10-25_16-18-22.avi")
    cap.setFramePos(0.2)
    print(cap.nowFrame)
    while True:
        ret, frame = cap.capOneFrame()
        if not ret:
            break
        cv2.imshow("win", frame)
        cv2.waitKey(int(1000/cap.frameRate))
    print(cap.nowFrame)



