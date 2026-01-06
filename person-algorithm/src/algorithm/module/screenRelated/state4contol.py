import numpy as np
import time

class stateNote:
    def __init__(self):
        self.last_order = time.time()
        self.start_time = time.time()
        self.end_time = time.time()
        self.flagAction1 = None
        self.flagAction2 = None
        self.screenDirection = None
        self.observeBuffer = []
        self.angleBuffer = []
        self.active = None
        self.lastAngle = 180
        self.overtime = False
        self.instructon = ''


class actionState():
    def __init__(self, stateNote):
        self.order_ratio_thre = 0.35  # 有效动作评估阈值；
        self.measurePointIdx = [7, 6, 11, 9]
        self.state = stateNote

    def __ifOvertime(self):
        if time.time() - self.state.start_time > 20:
            self.state.overtime = True

    def __ifOrder(self, keypoint):
        ActionFlag = (keypoint[0][10][1] - keypoint[0][8][1]) / (keypoint[0][11][1] - keypoint[0][7][1])
        if ActionFlag < self.order_ratio_thre:
            self.state.active = True
            self.state.last_order = time.time()
        else:
            self.state.active = False

    def __actionKeeping(self):
        action = None
        stable = np.mean(self.state.observeBuffer[-15:])
        if abs(stable) < 10:
            if abs(np.mean(self.state.angleBuffer[-6:])) < 45:
                action = 'Horizontal'
            else:
                action = 'Vertical'
        return action

    def __calculateAngle(self, keypoints):
        p = self.measurePointIdx
        angle = twoLineAngle(keypoints[0][p[0] - 1], keypoints[0][p[1] - 1], keypoints[0][p[2] - 1], keypoints[0][p[3] - 1])
        return angle

    def __clearHistory(self):
        self.state.observeBuffer = []  # clear history!
        self.state.flagAction1 = None
        self.state.flagAction2 = None
        self.state.screenDirection = None
        self.state.newStart = False
        self.state.overtime = False

    def updateState(self, keypoints):
        self.__ifOrder(keypoints)
        self.__ifOvertime()
        actionAngle = None
        angleChange = None

        if self.state.flagAction1 is not None and self.state.overtime:
            self.__clearHistory()
            self.state.instructon = 'Over Time!!! Invalid Action!'

        elif self.state.active:
            actionAngle = self.__calculateAngle(keypoints)
            angleChange = actionAngle - self.state.lastAngle
            self.state.angleBuffer.append(actionAngle)
            self.state.observeBuffer.append(angleChange)
            self.state.lastAngle = actionAngle

            if abs(actionAngle) < 20 or (70 < actionAngle < 110):
                action = self.__actionKeeping()
                if action is not None:
                    # Try to fix First action!! Fail....made performance wrong..Leia, 20210806
                    # if self.state.flagAction1 is None or time.time()-self.state.start_time < 0.05:
                    if self.state.flagAction1 is None:  # First action can be fixed!!
                        self.state.flagAction1 = self.__actionKeeping()
                        self.state.start_time = time.time()
                    # Try to fix Second action!! Fail....made performance wrong..Leia, 20210806
                    # elif time.time()-self.state.start_time > 2 or time.time() - self.state.end_time < 0.05:
                    elif time.time()-self.state.start_time > 1: # Second action can be fixed!!
                        self.state.flagAction2 = self.__actionKeeping()
                        self.state.end_time = time.time()
                        if self.state.flagAction1 != self.state.flagAction2:
                            self.state.screenDirection = self.state.flagAction2
                        else:
                            self.state.instructon = 'Something Wrong!'
                else:
                    self.state.instructon = 'No start/end!'
                print('------------------------------')
                print(action)
                print(self.state.flagAction1)
                print(self.state.flagAction1)

            else:
                self.state.instructon = 'During Acting!'

        elif time.time() - self.state.last_order > 0.5:
            self.__clearHistory()
            self.state.instructon = 'Order finished!!'
        return actionAngle, angleChange


def twoLineAngle(point1, point2, point3, point4):
    arr_0 = np.array([(point2[0] - point1[0]), (point2[1] - point1[1])])
    arr_1 = np.array([(point4[0] - point3[0]), (point4[1] - point3[1])])
    angle = clockwise_angle(arr_0, arr_1)
    return angle


def clockwise_angle(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    theta = np.arctan2(det, dot)
    theta = theta if theta > 0 else 2 * np.pi + theta
    angle = 180 - theta * 180 / np.pi
    angle = round(angle, 1)
    return angle


def subMean(score_list):
    ave_score_list = []
    if len(score_list) < 3:
        ave_score_list = score_list
    else:
        count = len(score_list)//3
        for i in range(0, 3*count, 3):
            ave_score_list.append(round(abs(np.mean(score_list[i:i+3])), 2))
    return ave_score_list







# def __ifOrder(self, keypoint):
#     ActionFlag = (keypoint[0][10][1] - keypoint[0][8][1]) / (keypoint[0][11][1] - keypoint[0][7][1])
#     if ActionFlag < self.order_ratio_thre:
#         self.state.active = True
#         if self.state.last_order is None or time.time() - self.state.last_order > 0.5:
#             self.state.newStart = True
#             self.state.start_time = time.time()
#         else:
#             self.state.newStart = False
#         self.state.last_order = time.time()
#     else:
#         self.state.active = False
#         if time.time() - self.state.last_order > 2:
#             self.state.observeBuffer = []  # clear history!
#             self.state.newStart = False
#             self.state.startAngle = None
#             self.state.newEnd = False
#             self.state.endAngle = None

#     if self.state.newStart:
#         self.state.startAngle = actionAngle
#         self.state.endAngle = actionAngle
#         if abs(actionAngle) < 50:
#             self.state.flagAction1 = 'Horizontal'
#             self.state.lastAngle = 0
#         else:
#             self.state.flagAction1 = 'Vertical'
#             self.state.lastAngle = 90
#     else:
#         angleChange = actionAngle - self.state.lastAngle
#         self.state.observeBuffer.append(angleChange)
#
#         condition_1 = self.state.newEnd is False
#         condition_2 = abs(actionAngle-self.state.startAngle) > 45
#         condition_3 = np.sum(self.state.observeBuffer[-10:]) < 10
#         condition_4 = actionAngle - self.state.endAngle > 10
#
#         if condition_1 and condition_2 and condition_3 and condition_4:
#             self.state.newEnd = True
#             self.state.endAngle = actionAngle
#             if abs(actionAngle) < 45:
#                 self.state.flagAction2 = 'Horizontal'
#             else:
#                 self.state.flagAction2 = 'Vertical'
#         else:
#             self.state.newEnd = False
#
#         self.state.lastAngle = actionAngle
# return actionAngle, angleChange