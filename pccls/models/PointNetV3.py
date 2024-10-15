import torch.nn as nn
from typing import List
from functools import partial
from pccls.models.module_utils.mlp_utils import MLPND
from pccls.models.PointClsStem import PointClsStem
from pccls.models.PointClsHead import PointClsHead


class PointNetV3Encoder(nn.Module):
    def __init__(self, d_in: int, d_feats: List[int], norm: str='BN', act: str='ReLU', *, ndim: int=1, channels_first: bool=False):
        super().__init__()
        self.n_stgs = len(d_feats)
        mlp_fn = partial(MLPND, norm=norm, act=act, ndim=ndim, channels_first=channels_first)
        self.stgs = nn.ModuleList([mlp_fn(d_in if i == 0 else d_feats[i-1], d_feats[i]) for i in range(self.n_stgs)])

    def forward(self, x):
        for stg in self.stgs:
            x = stg(x)
        return x


class PointNetV3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stem = PointClsStem(cfg.d_in, cfg.d_feats[0])
        self.feat = PointNetV3Encoder(cfg.d_feats[0], cfg.d_feats)
        self.head = PointClsHead(cfg.d_feats[-1], cfg.n_cls)

    def forward(self, x):
        x = self.stem(x)
        x = self.feat(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    import torch
    from easydict import EasyDict
    cfg = EasyDict(dict(n_cls=40, d_in=3, d_feats=[64, 128, 256, 512]))
    batch_size, point_size, channel_size, class_size = (1 << 4), (1 << 10), 3, 40
    x, y = torch.rand((batch_size, point_size, channel_size)).cuda(), torch.randint(0, 40, (batch_size,)).long().cuda()
    model = PointNetV3(cfg).cuda()
    z = model(x)
    print(x.shape, y.shape, z.shape)