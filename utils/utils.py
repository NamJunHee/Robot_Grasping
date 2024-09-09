import logging
import torch
import argparse
import os
import yaml




logging.basicConfig(level=logging.INFO)
def get_device(force_cpu):
    # Check if CUDA can be used
    if torch.cuda.is_available() and not force_cpu:
        logging.info("CUDA detected. Running with GPU acceleration.")
        device = torch.device("cuda")
    elif force_cpu:
        logging.info("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
        device = torch.device("cpu")
    else:
        logging.info("CUDA is *NOT* detected. Running with only CPU.")
        device = torch.device("cpu")
    return device



def parse_args():
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)

    default_log_loc = "./logs/"
    parser.add_argument('--log_location', action='store',
                        dest='log_location',
                        help='log location',
                        default=default_log_loc, type=str, required=False)

    parser.add_argument('--configs_path', action='store',
                        dest='configs_path',
                        help='configs_path',
                        default='default.yml', type=str, required=False)
    
    parser.add_argument('--model_configs_path', action='store',
                        dest='model_configs_path',
                        help='model_configs_path',
                        default='model_info.yml', type=str, required=False)
    
    parser.add_argument('--seed', action='store',
                        dest='rand_seed',
                        help='random seed',
                        default='1', type=int, required=False)
    
    parser.add_argument('--test_name', action='store',
                        dest='test_name',
                        help='test_name',
                         type=str, required=True)
    
    args = parser.parse_args()
    return args





def import_default_configs(test_name, log_path):
    cfg_folder = "./configs/"
    default_cfg_path = os.path.join(cfg_folder,"default.yml")

    with open(default_cfg_path, 'r') as f:
        default_cfg = yaml.load(f, Loader=yaml.FullLoader)


    out_cfg = default_cfg["TestName"][test_name]
    out_cfg["log_path"] = log_path
    

    return out_cfg

def import_gripper_configs(gripper_type):
    cfg_folder = "./configs/"
    gripper_cfg_path = os.path.join(cfg_folder, "gripper_info.yml")
    
    with open(gripper_cfg_path, 'r') as f:
        gripper_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    out_cfg = gripper_cfg[gripper_type]
    
    return out_cfg

def import_model_configs(model):
    cfg_folder = "./configs/"
    gripper_cfg_path = os.path.join(cfg_folder, "model_info.yml")
    
    with open(gripper_cfg_path, 'r') as f:
        gripper_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    out_cfg = gripper_cfg[model]
    
    return out_cfg