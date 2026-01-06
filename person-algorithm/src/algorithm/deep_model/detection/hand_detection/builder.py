#added by zhxh,2022.03.30
from .blazeplam import BlazePlamHandDetModel


def get_hand_det_model(name, model_dir=None, *args, **kwargs):
    if name == 'blazepalm_det':
        model = BlazePlamHandDetModel(**kwargs)
    else:
        raise NotImplementedError
    if model_dir is not None and model_dir!="":
        ret = model.load_model(model_dir)
        if not ret:
            raise FileNotFoundError
    return model
