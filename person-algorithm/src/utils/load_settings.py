import json

def load_model_setting(setting_file):
    with open(setting_file, 'r') as f:
        setting = json.load(f)
    return setting["name"], setting["model_dir"], setting["structure"], setting["parameters"], setting["gpu_setting"]


if __name__ == '__main__':
    para = load_model_setting("user/settings/model/detection/body_detection/centernet.json")