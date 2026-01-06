import cv2
import os
import glob
import random
import numpy as np

class RealsensePicture():
    def __init__(self, mode, dir, type="jpg", shuffle=False, *args, **kwargs):
        # mode True: pic dictionary; mode False: pic dictionary list
        self.mode = mode
        self.dir = dir  # 目录名称
        self.type = type
        self.pics, self.depth_pics = self.initPics()
        self.index = 0
        self.wholeFrameNum = len(self.pics)
        if shuffle:
            index = list(range(len(self.pics)))
            np.random.shuffle(index)
            self.pics = np.array(self.pics)[index].tolist()
            self.depth_pics = np.array(self.depth_pics)[index].tolist()

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, index):
        if index > (len(self) - 1):
            raise StopIteration
        img_name, depth_img_name, frame, depth_frame = self.capOneFrame()
        return img_name, depth_img_name, frame, depth_frame

    def initPics(self):
        if self.mode:
            assert isinstance(self.dir, str)
            pic = glob.glob(self.dir + "/*.{}".format(self.type))
            depth_pic = [p.replace(self.type, "npy") for p in pic if os.path.exists(p.replace(self.type, "npy"))]
            pics = pic
            depth_pics = depth_pic
        else:
            assert isinstance(self.dir, list)
            pics = []
            depth_pics = []
            for dir in self.dir:
                pic = glob.glob(dir + "/*.{}".format(self.type))
                pic.sort()
                depth_pic = [p.replace(self.type, ".npy") for p in pic if os.path.exists(p.replace(self.type, ".npy"))]
                pics += pic
                depth_pics += depth_pic
        return pics, depth_pics


    def capOneFrame(self):
        frame = cv2.imread(self.pics[self.index])
        depth_frame = np.load(self.depth_pics[self.index])
        if frame is not None:
            self.set_next_frame()
        return self.pics[self.index], self.depth_pics[self.index], frame, depth_frame

    def retrieveOneFrame(self):
        frame = cv2.imread(self.pics[self.index])
        depth_frame = np.load(self.depth_pics[self.index])
        return frame, depth_frame

    def set_next_frame(self):
        self.index += 1
        if self.index > (self.wholeFrameNum - 1):
            self.index = self.wholeFrameNum - 1

    def set_last_frame(self):
        self.index -= 1
        if self.index < 0:
            self.index = 0

    def setFramePos(self, percent):
        assert not self.mode
        frame_num = int(self.wholeFrameNum * percent)
        self.index = frame_num

    def getFrameID(self):
        return self.index



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



