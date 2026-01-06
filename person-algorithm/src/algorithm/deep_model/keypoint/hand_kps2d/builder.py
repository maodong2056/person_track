
# added by zh,2022.03.30
from .blazepalm_kps import BlazePlamHandKpsModel


def get_hand_2d_model(name, model_dir, *args, **kwargs):
    if name == "blazepalm_kps":
        model = BlazePlamHandKpsModel(**kwargs)
    else:
        raise NotImplementedError
    if model_dir is not None and model_dir!="":
        ret = model.load_model(model_dir)
        if not ret:
            raise FileNotFoundError
    return model
