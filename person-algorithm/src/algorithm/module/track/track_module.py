from src.algorithm.deepModel.deepDetection import DeepDetModel
from src.algorithm.deepModel.deepDetection import DeepDetModelWhole
from src.algorithm.deepModel.deepSort import DeepSort


class TrackModule(DeepDetModel, DeepSort):
    def __init__(self, *args, **kwargs):
        # DeepDetModel.__init__(*args, **kwargs)
        # DeepSort.__init__(*args, **kwargs)
        DeepDetModel.__init__(self, square_input=False,*args, **kwargs)
        DeepSort.__init__(self, *args, **kwargs)
        # super(trackModule, self).__init__(*args, **kwargs)
        # super(DeepSort, self).__init__()

    def get_result(self, image, detection_face_thres=0.5,
                   detection_body_thres=0.5, Knum=10, preprocess_pad=True,
                   tag_thres=0.5, *args, **kwargs):
        # img0, det_result, facedet_result, features, btag, ftag = \
        #     self.get_output(image, detection_face_thres, detection_body_thres, Knum)
        img0Shape, result = self.get_output(image, detection_face_thres, detection_body_thres, Knum, preprocess_pad)
        # self.update(det_result[:, :4], det_result[:, 4], features, image, btag=btag)
        # self.update_face(facedet_result[:, :4], facedet_result[:, 4], facedet_result[:, 5:],
        #                  ftag=ftag, tag_match=True, tag_thres=tag_thres)
        # output = self.update_notag(det_result[:, :4], det_result[:, 4], features,
        #             facedet_result[:, :4], facedet_result[:, 4], facedet_result[:, 5:],
        #             btag, ftag, tag_match=True, tag_thres=tag_thres, img_size=img0.shape)
        output = self.update_notag(result, img_size=img0Shape)
        return output

class TrackWholeModule(DeepDetModelWhole, DeepSort):
    def __init__(self, *args, **kwargs):
        # DeepDetModel.__init__(*args, **kwargs)
        # DeepSort.__init__(*args, **kwargs)
        DeepDetModelWhole.__init__(self, square_input=False,*args, **kwargs)
        DeepSort.__init__(self, *args, **kwargs)
        # super(trackModule, self).__init__(*args, **kwargs)
        # super(DeepSort, self).__init__()

    def get_result(self, image, detection_face_thres=0.5,
                   detection_body_thres=0.5, Knum=10, preprocess_pad=True,
                   tag_thres=0.5, *args, **kwargs):
        # img0, det_result, facedet_result, features, btag, ftag = \
        #     self.get_output(image, detection_face_thres, detection_body_thres, Knum)
        img0Shape, result = self.get_output(image, detection_face_thres, detection_body_thres, Knum, preprocess_pad)
        # self.update(det_result[:, :4], det_result[:, 4], features, image, btag=btag)
        # self.update_face(facedet_result[:, :4], facedet_result[:, 4], facedet_result[:, 5:],
        #                  ftag=ftag, tag_match=True, tag_thres=tag_thres)
        # output = self.update_notag(det_result[:, :4], det_result[:, 4], features,
        #             facedet_result[:, :4], facedet_result[:, 4], facedet_result[:, 5:],
        #             btag, ftag, tag_match=True, tag_thres=tag_thres, img_size=img0.shape)
        output = self.update_notag(result, img_size=img0Shape)
        return output

