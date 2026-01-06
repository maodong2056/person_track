from .deepSort import DeepSort
from .byteTrack import BYTETracker

def get_track_model(name, *args, **kwargs):
    if name == "deep_sort":
        model = DeepSort(**kwargs)
    elif name == "byte_track":
        model = BYTETracker(**kwargs)
    else:
        raise NotImplementedError
    return model
