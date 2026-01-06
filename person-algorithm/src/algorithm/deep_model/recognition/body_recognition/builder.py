from .pcb import PCBReidDense, PCBReidMobilev2


def get_body_rec_model(name, model_dir, *args, **kwargs):
    if name == "pcb_reid_dense":
        model = PCBReidDense(**kwargs)
    elif name == "pcb_reid_mobilev2":
        model = PCBReidMobilev2(**kwargs)
    else:
        raise NotImplementedError
    if model_dir is not None and model_dir!="":
        ret = model.load_model(model_dir)
        if not ret:
            raise FileNotFoundError
    return model
