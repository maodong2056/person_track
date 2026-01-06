import os
import inspect

def caller_class(class_func):
    def inner(*args, **kwargs):
        frame_stack = inspect.stack()
        caller_frame = frame_stack[1]
        caller_file_path = caller_frame.filename
        p, file_name = os.path.split(caller_file_path)
        f, ext = os.path.splitext(file_name)
        get_class = class_func(*args, **kwargs, **{"filename": f})
        return get_class
    return inner

