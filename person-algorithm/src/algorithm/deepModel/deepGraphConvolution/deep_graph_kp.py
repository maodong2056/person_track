import os

from src.algorithm.deepModel.basic_model import BasicDeepModel
from src.algorithm.deepModel.deepGraphConvolution import SiameseNet
from src.algorithm.deepModel.deepGraphConvolution import Graph
from .utils.graphUtils import *


logger = logging.getLogger(__name__)


class DeepGraphKPModel(BasicDeepModel):
    # def __init__(self, is_useGpu, gpu_num):
    def __init__(self, *args, **kwargs):
        super(DeepGraphKPModel, self).__init__(*args, **kwargs)
        A = Graph().A
        self.model = SiameseNet(A).eval()

        # is_useGpu = False
        # if is_useGpu:
        #     self.device = torch.device('cuda:{}'.format(gpu_num))
        # else:
        #     self.device = torch.device('cpu')

    def load_model(self, model_path, *args, **kwargs):
        if os.path.exists(model_path):
            try:
                self.model = loadModel(self.model, model_path)
            except Exception as e:
                logger.error(e)
                return False
            self.model = self.model.to(self.device).eval()
            return True
        else:
            return False

    def __preprocess(self, keypoints, bbox, pose_norm=False, PoseTrack2COCO=True, *args, **kwargs):
        if pose_norm:
            keypoints = KP_pose_norm(bbox, keypoints)

        if PoseTrack2COCO:
            # PoseTrack(15 points), but COCO(18 points) is used in our keypoint model;
            keypoints, center_point = KP_transform(keypoints)

        KPdata = keypoints2graph2data(keypoints, bbox)
        KPdata = torch.from_numpy(KPdata).unsqueeze(0).float().to(self.device)
        return KPdata

    def get_feature(self, keypoints, bbox, *args, **kwargs):
        KPdata = self.__preprocess(keypoints, bbox)
        with torch.no_grad():
            feature = self.model.forward(KPdata)
        return feature
