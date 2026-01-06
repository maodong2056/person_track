import torch
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert models to Libtorch pt')
    parser.add_argument('--output-file', type=str, default='tmp.pt')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 256, 192],
        help='input size')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_path = "../user/libtorch"
    ####################################################################
    from src import DeepSiamDRKPModel as DeepKPModel
    import json
    with open("../user/lib/setting_trail.json") as f:
        config = json.load(f)
    my_kps = DeepKPModel(**config["deepModel"]["deepKeypoint"]["structure"],
                        is_useGpu=False, gpu_num=0)

    kp_load_ret = my_kps.load_model(config["deepModel"]["deepKeypoint"]["model"])
    input = torch.randn(args.shape).cpu()
    try:
        save_script_module = torch.jit.trace(my_kps.model, input)
        torch.jit.save(save_script_module, os.path.join(save_path, args.output_file))
        print('Save Libtorch file to {} sucess!'.format(os.path.join(save_path, args.output_file)))
    except Exception as e:
        print('Save Libtorch file fail! Error: {}'.format(e))