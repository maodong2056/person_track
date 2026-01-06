from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import torch
import torch.nn as nn
# from src.algorithm.deepDetection.model import create_model, load_model
from src import DeepSiamDRKPModel as DeepKPModel
# from src import DeepKPModel
from src import DeepDetModel, DeepDetModelWhole
from src import Deep3DKPModel, Deep3DKPLiteModel

def fuse_deconv_and_bn(deconv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.ConvTranspose2d(deconv.in_channels,
                                             deconv.out_channels,
                                             kernel_size=deconv.kernel_size,
                                             stride=deconv.stride,
                                             padding=deconv.padding,
                                             groups= deconv.out_channels,
                                             bias=True
                                             ).requires_grad_(False).to(deconv.weight.device)

    # prepare filters
    w_conv = deconv.weight.clone().view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(deconv.weight.size(0), device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        for i in s:
            unicodedata.numeric(i)  # 把一个表示数字的字符串转换为浮点数返回的函数
            #return True
        return True
    except (TypeError, ValueError):
        pass
    return False

def name_to_strline(name):
    split_text = name.split(".")
    str_line = ""
    for st in split_text:
        if not is_number(st):
            str_line += ".{}".format(st) if len(str_line) != 0 else st
        else:
            str_line += "[{}]".format(st)
    return str_line
#my_trackModule.load_model(detection_model)  input_shape=(384,512)

# 获取所有deconv bn的name
def get_deconv_layers(model):
    module = model.model.named_modules()
    for m in module:
        if isinstance(m[1], nn.ConvTranspose2d):
            bn = next(module)
            if isinstance(bn[1], nn.BatchNorm2d):
                # conv = fuse_deconv_and_bn(m[1], bn[1])
                conv_name = name_to_strline(m[0])
                bn_name = name_to_strline(bn[0])
                print("deconv:", "model.model." + conv_name, "bn:", "model.model." + bn_name)

# 通过获取的name修改该函数
def convert_onnx(model, input_shape=(256,192), exp_id='mobilex_det_new'):
    model.model.neck.deconv1[1][0] = fuse_deconv_and_bn(model.model.neck.deconv1[1][0], model.model.neck.deconv1[1][1])
    del model.model.neck.deconv1[1][1]
    # # if type(m) is deConv and hasattr(m, 'bn'):
    model.model.neck.deconv2[1][0] = fuse_deconv_and_bn(model.model.neck.deconv2[1][0], model.model.neck.deconv2[1][1])
    del model.model.neck.deconv2[1][1]
    model.model.neck.deconv3[1][0] = fuse_deconv_and_bn(model.model.neck.deconv3[1][0], model.model.neck.deconv3[1][1])
    del model.model.neck.deconv3[1][1]
    # model = model.to(device)
    model.model.eval()
    dummy_input1 = torch.randn(1, 3, input_shape[0], input_shape[1]).to(model.device)
    torch.onnx.export(
          model.model, (dummy_input1, ), "../user/onnx/{}.onnx".format(exp_id), opset_version=9)

    print('convert onnx finish')


if __name__ == '__main__':
    # -----------------------Load config-------------------------------------------
    with open("../user/lib/setting_whole_trail.json") as f:
        config = json.load(f)
    gpu_setting = config["gpu_setting"]

    # ---------------------------Keypoint-------------------------------------------
    # my_kp = DeepKPModel(**config["deepModel"]["deepKeypoint"]["structure"],
    #                     **gpu_setting, onnx_cat_output=True)
    # kp_load_ret = my_kp.load_model(config["deepModel"]["deepKeypoint"]["model"])
    # ---------------------------Detection------------------------------------------
    # my_detModule = DeepDetModel(**config["deepModel"]["deepDetection"]["structure"],
    #                             **gpu_setting)
    # det_settinyangg = config["deepModel"]["deepDetection"]["parameters"]
    # det_load_ret = my_detModule.load_model(config["deepModel"]["deepDetection"]["model"])
    # ---------------------------Detection Whole------------------------------------
    my_detModule = DeepDetModelWhole(**config["deepModel"]["deepDetection"]["structure"],
                                **gpu_setting)
    det_setting = config["deepModel"]["deepDetection"]["parameters"]
    det_load_ret = my_detModule.load_model(config["deepModel"]["deepDetection"]["model"])
    # ---------------------------3d Keypoint-------------------------------------------
    # my_3dkp_model = Deep3DKPModel(**config["deepModel"]["deep3DKeypoint"]["structure"],
    #                                   **gpu_setting)
    # tdkp_load_ret = my_3dkp_model.load_model(config["deepModel"]["deep3DKeypoint"]["model"])
    # ---------------------------Detection------------------------------------------
    # ---------------------------Convert-------------------------------------------
    # get_deconv_layers(my_3dkp_model)
    # convert_onnx(my_3dkp_model, input_shape=(256,256), exp_id="3dkps_news_ep159")
    convert_onnx(my_detModule, input_shape=(384, 512), exp_id="mobilex_whole_model_tag")