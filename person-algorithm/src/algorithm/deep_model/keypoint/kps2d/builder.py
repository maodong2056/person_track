from .simba import DeepSimbaKPModel
from .siamdr import DeepSiamDRKPModel


def get_pose_2d_model(name, model_dir, *args, **kwargs):
    if name == "simba":
        model = DeepSimbaKPModel(**kwargs)
    elif name == "siamdr":
        model = DeepSiamDRKPModel(**kwargs)
    else:
        raise NotImplementedError
    if model_dir is not None and model_dir!="":
        ret = model.load_model(model_dir)
        if not ret:
            raise FileNotFoundError
    return model
