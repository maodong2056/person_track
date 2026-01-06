from .centernet import CenterNetBodyDetModel, CenterNetBodyDetWholeModel
from .yolox import YoloXBodyDetModel


def get_body_det_model(name, model_dir=None, *args, **kwargs):
    if name == "centernet":
        model = CenterNetBodyDetModel(**kwargs)
    elif name == "centernet_whole":
        model = CenterNetBodyDetWholeModel(**kwargs)
    elif name == "yolox":
        model = YoloXBodyDetModel(**kwargs)
    else:
        raise NotImplementedError
    if model_dir is not None and model_dir!="":
        ret = model.load_model(model_dir)
        if not ret:
            raise FileNotFoundError
    return model
