import torch.nn as nn
from typing import List
from functools import partial
from pccls.models.module_utils.mlp_utils import proj_fn, norm_fn, act_fn
from pccls.models.PointClsHead import PointClsHead


class BasicLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, norm: str='BN', act: str='ReLU', *, ndim: int=1):
        super().__init__()
        part_proj_fn = partial(proj_fn, bias=False, ndim=ndim)
        part_bn_fn = partial(norm_fn, NAME=norm, ndim=ndim)
        part_act_fn = partial(act_fn, NAME=act)
        self.mlp1 = nn.Sequential(part_proj_fn(d_in, d_out), part_bn_fn(d_feat=d_out), part_act_fn(), part_proj_fn(d_out, d_out), part_bn_fn(d_feat=d_out))
        self.mlp2 = nn.Sequential(part_proj_fn(d_in, d_out), part_bn_fn(d_feat=d_out))
        self.relu = part_act_fn()

    def forward(self, x):
        x = self.relu(self.mlp1(x) + self.mlp2(x))
        return x


class PointNetV01Encoder(nn.Module):
    def __init__(self, d_in: int, d_feats: List[int], norm: str='BN', act: str='ReLU', *, ndim: int=1):
        super().__init__()
        self.n_stgs = len(d_feats)
        self.stgs = nn.ModuleList([BasicLayer(d_in if i == 0 else d_feats[i-1], d_feats[i], norm=norm, act=act, ndim=ndim) for i in range(self.n_stgs)])

    def forward(self, x):
        for stg in self.stgs:
            x = stg(x)
        return x


class PointNetV01(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PointNetV01Encoder(cfg.d_in, cfg.d_feats, cfg.norm, cfg.act, ndim=1)
        self.head = PointClsHead(cfg.d_feats[-1], cfg.n_cls, cfg.norm, cfg.act)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    import torch
    from easydict import EasyDict
    cfg = EasyDict(dict(n_cls=40, d_in=3, d_feats=[64, 128, 256, 512], norm='BN', act='ReLU'))
    batch_size, point_size, channel_size, class_size = (1 << 4), (1 << 10), 3, 40
    x, y = torch.rand((batch_size, point_size, channel_size)).cuda(), torch.randint(0, 40, (batch_size,)).long().cuda()
    model = PointNetV01(cfg).cuda()
    z = model(x)
    print(x.shape, y.shape, z.shape)