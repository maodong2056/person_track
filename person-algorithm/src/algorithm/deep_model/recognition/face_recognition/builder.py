from .arcface import ArcfaceResnet
from .arcface import ArcfaceMobilefacenet, ArcfaceResnet50

def get_face_rec_model(name, model_dir, *args, **kwargs):
    if name == "arcface_res":
        model = ArcfaceResnet(**kwargs)
    elif name == 'arcface_mobilefacenet':
        model = ArcfaceMobilefacenet(**kwargs)
    elif name == "arcface_resnet50":
        model = ArcfaceResnet50(**kwargs)
    else:
        raise NotImplementedError
    if model_dir is not None and model_dir!="":
        ret = model.load_model(model_dir)
        if not ret:
            raise FileNotFoundError
    return model
