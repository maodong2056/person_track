from .graformer import Graformer
from .posenet import PoseNet, PoseNetSiamDR
from .rootnet import RootNet
from .action_classify import DeepActionClassify

def get_pose_3d_model(name, model_dir, *args, **kwargs):
    if name == "posenet":
        model = PoseNet(**kwargs)
    elif name == "posenet_siamdr":
        model = PoseNetSiamDR(**kwargs)
    elif name == "graformer":
        model = Graformer(**kwargs)
    elif name == "rootnet":
        model = RootNet(**kwargs)
    elif name == "action_classify":
        model = DeepActionClassify(**kwargs)
    else:
        raise NotImplementedError
    if model_dir is not None and model_dir!="":
        ret = model.load_model(model_dir)
        if not ret:
            raise FileNotFoundError
    return model
