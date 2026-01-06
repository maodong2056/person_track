import numpy as np
from src.algorithm.module.traceFilter.LK import EmaFilter, OneEuroFilter
from src.algorithm.deepModel.deepKeypoint import DeepKPModel

class LandmarkFilter():
    def __init__(self, thres=1.5, smooth_landmark=0.75, mode='euro'):
        self.previous_landmarks_set = None

        self.with_landmark = True
        self.alpha = smooth_landmark
        self.landmark_dict = {}
        self.thres = thres

        if mode == 'ema':
            self.filter = EmaFilter(self.alpha)
        else:
            self.filter = OneEuroFilter()

    def get_smooth_landmark(self, lm_model: DeepKPModel, output: list, image: np.ndarray):
        # out_dict = {output[i, -1]: output[i] for i in range(len(output))}
        now_landmark = {}
        # bbox = output[:, :4]
        bbox = np.array([out["body_box"] for out in output])
        landmarks = lm_model.get_output(image, bbox)
        for i in range(len(landmarks)):
            # id = output[i][-1]
            id = output[i]["id"]
            now_landmark[id] = landmarks[i]
        out_landmark = self.smooth(now_landmark, self.landmark_dict)
        for i in range(len(out_landmark)):
            # id = output[i][-1]
            id = output[i]["id"]
            now_landmark[id] = out_landmark[i]
        self.landmark_dict = now_landmark
        # for i in range(len(output)):
        #     id = output[i][-1]
        #     bbox = output[i][:4]
        #     lm_model.get_output(image, )
        return out_landmark


    def smooth(self, now_landmarks, previous_landmarks):

        result = []
        for i in now_landmarks.keys():
            if i in previous_landmarks.keys():
                dis = np.sqrt(np.square(now_landmarks[i][:, 0] - previous_landmarks[i][:, 0]) + np.square(
                    now_landmarks[i][:, 1] - previous_landmarks[i][:, 1]))
                print(dis[now_landmarks[i][:, 2] > 0.05].mean())
                if dis[now_landmarks[i][:, 2] > 0.05].mean() < self.thres:
                    result.append(previous_landmarks[i])
                else:
                    result.append(self.filter(now_landmarks[i], previous_landmarks[i]))
            else:
                result.append(now_landmarks[i])

        return np.array(result)

    def do_moving_average(self, p_now, p_previous):
        p = self.alpha * p_now + (1 - self.alpha) * p_previous
        return p
