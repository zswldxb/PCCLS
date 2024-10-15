import os
import yaml
from easydict import EasyDict
import random
import numpy as np
import torch


# set random seed 
def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_printoptions(10)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True 


# io
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def print(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def merge_new_config(old_cfg, new_cfg):
    if '_BASE_CONFIG_' in new_cfg:
        with open(new_cfg['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        old_cfg.update(EasyDict(yaml_config))

    for key, val in new_cfg.items():
        if not isinstance(val, dict):
            old_cfg[key] = val
            continue
        if key not in old_cfg:
            old_cfg[key] = EasyDict()
        merge_new_config(old_cfg[key], val)

    return old_cfg


def load_config(cfg_path):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    with open(cfg_path) as f:
        try:
            cfg = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            cfg = yaml.safe_load(f)
    return cfg