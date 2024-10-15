import torch.nn as nn
from functools import partial


act_dict = {
    'ReLU': partial(nn.ReLU),
    'LeakyReLU': partial(nn.LeakyReLU, negative_slope=0.2),
    'GELU': partial(nn.GELU),
    'GLU': partial(nn.GLU),
}


def act_fn(NAME, **kwargs):
    return act_dict[NAME](**kwargs)