import torch.nn as nn
from functools import partial
from pccls.models.module_utils.norm_utils import *
from pccls.models.module_utils.act_utils import *
from pccls.models.module_utils.pool_utils import *


def proj_fn(d_in, d_out, bias=True, *, ndim=None):
    return nn.Linear(d_in, d_out, bias=bias)


class MLPND(nn.Module):
    def __init__(self, d_in, d_out, norm, act, *, ndim):
        super().__init__()
        self.mlp = nn.Sequential(
            proj_fn(d_in, d_out, bias=False, ndim=ndim),
            norm_fn(norm, d_feat=d_out, ndim=ndim),
            act_fn(act)
        )

    def forward(self, x):
        return self.mlp(x)


class ResMLPND(nn.Module):
    def __init__(self, d_feat, ratio, norm, act, *, ndim):
        super().__init__()
        d_hid = int(round(d_feat * ratio))
        self.mlp = nn.Sequential(
            proj_fn(d_feat, d_hid, bias=False, ndim=ndim),
            norm_fn(norm, d_feat=d_hid, ndim=ndim),
            act_fn(act),
            proj_fn(d_hid, d_feat, bias=False, ndim=ndim),
            norm_fn(norm, d_feat=d_feat, ndim=ndim)
        )
        self.act = act_fn(act)

    def forward(self, x):
        return self.act(self.mlp(x) + x)