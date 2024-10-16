import torch.nn as nn
from functools import partial
from einops import rearrange


class BatchNormAdapter0D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, momentum=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_feat, eps=eps, momentum=momentum)

    def forward(self, feat):
        return self.norm(feat)


class BatchNormAdapter1D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, momentum=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_feat, eps=eps, momentum=momentum)

    def forward(self, feat):
        return rearrange(self.norm(rearrange(feat, "b x c -> b c x")), "b c x -> b x c")


class BatchNormAdapter2D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, momentum=0.1):
        super().__init__()
        self.norm = nn.BatchNorm2d(d_feat, eps=eps, momentum=momentum)

    def forward(self, feat):
        return rearrange(self.norm(rearrange(feat, "b x y c -> b c x y")), "b c x y -> b x y c")


class BatchNormAdapter3D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, momentum=0.1):
        super().__init__()
        self.norm = nn.BatchNorm3d(d_feat, eps=eps, momentum=momentum)

    def forward(self, feat):
        return rearrange(self.norm(rearrange(feat, "b x y z c -> b c x y z")), "b c x y z -> b x y z c")


norm_dict = {
    'BN': {
        0: partial(BatchNormAdapter0D, eps=1e-3, momentum=0.01),
        1: partial(BatchNormAdapter1D, eps=1e-3, momentum=0.01),
        2: partial(BatchNormAdapter2D, eps=1e-3, momentum=0.01),
        3: partial(BatchNormAdapter3D, eps=1e-3, momentum=0.01),
    },
    'LN': {
        0: partial(nn.LayerNorm, eps=1e-3),
        1: partial(nn.LayerNorm, eps=1e-3),
        2: partial(nn.LayerNorm, eps=1e-3),
        3: partial(nn.LayerNorm, eps=1e-3),
    }
}
def norm_fn(NAME, d_feat, *, ndim, **kwargs):
    return norm_dict[NAME][ndim](d_feat=d_feat, **kwargs)