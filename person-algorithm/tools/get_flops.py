import argparse
from mmcv.cnn import get_model_complexity_info

from src import DeepKPModel as DeepKPModel
from src import DeepDetModel, DeepDetModelWhole, Deep3DKPModel, DeepReidRecMobileModel, DeepFaceRecModel, Deep3DKPLiteModel
import json

with open("../user/lib/setting_trail.json") as f:
    config = json.load(f)
my_kps = Deep3DKPLiteModel(**config["deepModel"]["deep3DKeypoint"]["structure"],
                     is_useGpu=False, gpu_num=0)

input_shape = (3, 256, 256)
model = my_kps.model
flops, params = get_model_complexity_info(model.cpu(), input_shape)
split_line = '=' * 30
print(f'{split_line}\nInput shape: {input_shape}\n'
      f'Flops: {flops}\nParams: {params}\n{split_line}')
print('!!!Please be cautious if you use the results in papers. '
      'You may need to check if all ops are supported and verify that the '
      'flops computation is correct.')
# print(model)