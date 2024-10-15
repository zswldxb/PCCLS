import torch.nn as nn
from functools import partial
from pccls.models.module_utils.norm_utils import *
from pccls.models.module_utils.act_utils import *
from pccls.models.module_utils.pool_utils import *


ndim_list = [0, 1, 2, 3]
proj_dict = {
    True: {
        0: nn.Linear,
        1: partial(nn.Conv1d, kernel_size=1),
        2: partial(nn.Conv2d, kernel_size=1),
        3: partial(nn.Conv3d, kernel_size=1),
    },
    False: {
        0: nn.Linear,
        1: nn.Linear,
        2: nn.Linear,
        3: nn.Linear,
    }
}
def proj_fn(d_in, d_out, bias=True, *, ndim, channels_first=False):
    return proj_dict[channels_first][ndim](d_in, d_out, bias=bias)


class MLPND(nn.Module):
    def __init__(self, d_in, d_out, norm, act, *, ndim, channels_first=False):
        super().__init__()
        self.mlp = nn.Sequential(
            proj_fn(d_in, d_out, bias=False, ndim=ndim, channels_first=channels_first),
            norm_fn(norm, d_feat=d_out, ndim=ndim, channels_first=channels_first),
            act_fn(act)
        )

    def forward(self, x):
        return self.mlp(x)


class ResMLPND(nn.Module):
    def __init__(self, d_feat, ratio, norm, act, *, ndim, channels_first=False):
        super().__init__()
        d_hid = int(round(d_feat * ratio))
        self.mlp = nn.Sequential(
            proj_fn(d_feat, d_hid, bias=False, ndim=ndim, channels_first=channels_first),
            norm_fn(norm, d_feat=d_hid, ndim=ndim, channels_first=channels_first),
            act_fn(act),
            proj_fn(d_hid, d_feat, bias=False, ndim=ndim, channels_first=channels_first),
            norm_fn(norm, d_feat=d_feat, ndim=ndim, channels_first=channels_first)
        )
        self.act = act_fn(act)

    def forward(self, x):
        return self.act(self.mlp(x) + x)