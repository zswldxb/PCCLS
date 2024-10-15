import torch
import torch.nn as nn
from functools import partial
from einops import rearrange


class BatchNormAdapter0D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, momentum=0.1, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.norm = nn.BatchNorm1d(d_feat, eps=eps, momentum=momentum)

    def forward(self, feat):
        return self.norm(feat)


class BatchNormAdapter1D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, momentum=0.1, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.norm = nn.BatchNorm1d(d_feat, eps=eps, momentum=momentum)

    def forward(self, feat):
        return self.norm(feat) if self.channels_first else rearrange(self.norm(rearrange(feat, "b x c -> b c x")), "b c x -> b x c")


class BatchNormAdapter2D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, momentum=0.1, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.norm = nn.BatchNorm2d(d_feat, eps=eps, momentum=momentum)

    def forward(self, feat):
        return self.norm(feat) if self.channels_first else rearrange(self.norm(rearrange(feat, "b x y c -> b c x y")), "b c x y -> b x y c")


class BatchNormAdapter3D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, momentum=0.1, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.norm = nn.BatchNorm3d(d_feat, eps=eps, momentum=momentum)

    def forward(self, feat):
        return self.norm(feat) if self.channels_first else rearrange(self.norm(rearrange(feat, "b x y z c -> b c x y z")), "b c x y z -> b x y z c")


class LayerNormAdapter0D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.norm = nn.LayerNorm(d_feat, eps=eps)

    def forward(self, feat):
        return self.norm(feat)


class LayerNormAdapter1D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.norm = nn.LayerNorm(d_feat, eps=eps)

    def forward(self, feat):
        return rearrange(self.norm(rearrange(feat, "b c x -> b x c")), "b x c -> b c x") if self.channels_first else self.norm(feat)


class LayerNormAdapter2D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.norm = nn.LayerNorm(d_feat, eps=eps)

    def forward(self, feat):
        return rearrange(self.norm(rearrange(feat, "b c x y -> b x y c")), "b x y c -> b c x y") if self.channels_first else self.norm(feat)


class LayerNormAdapter3D(nn.Module):
    def __init__(self, d_feat, eps=1e-5, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.norm = nn.LayerNorm(d_feat, eps=eps)

    def forward(self, feat):
        return rearrange(self.norm(rearrange(feat, "b c x y z -> b x y z c")), "b x y z c -> b c x y z") if self.channels_first else self.norm(feat)


norm_dict = {
    'BN': {
        0: partial(BatchNormAdapter0D, eps=1e-3, momentum=0.01),
        1: partial(BatchNormAdapter1D, eps=1e-3, momentum=0.01),
        2: partial(BatchNormAdapter2D, eps=1e-3, momentum=0.01),
        3: partial(BatchNormAdapter3D, eps=1e-3, momentum=0.01),
    },
    'LN': {
        0: partial(LayerNormAdapter0D, eps=1e-3),
        1: partial(LayerNormAdapter1D, eps=1e-3),
        2: partial(LayerNormAdapter2D, eps=1e-3),
        3: partial(LayerNormAdapter3D, eps=1e-3),
    }
}
def norm_fn(NAME, d_feat, *, ndim, channels_first=False, **kwargs):
    return norm_dict[NAME][ndim](d_feat=d_feat, channels_first=channels_first, **kwargs)