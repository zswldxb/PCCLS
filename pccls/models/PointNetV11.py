import torch
import torch.nn as nn
from typing import List
from functools import partial
from pccls.models.module_utils.mlp_utils import proj_fn, norm_fn, act_fn
from pccls.models.module_utils.mlp_utils import MLPND
from pccls.models.module_utils.pool_utils import MaxPool
from pccls.models.PointClsStem import PointClsStem
from pccls.models.PointClsHead import PointClsHead
import pccls.ops.bpa_point_aggregation_ops.bpa_point_aggregation_op as bpa


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
    

class PointNetV11Encoder(nn.Module):
    def __init__(self, g_n, g_k, g_r, d_in, d_hid, d_out, norm: str='BN', act: str='ReLU'):
        super().__init__()
        self.g_n, self.g_k, self.g_r = g_n, g_k, g_r
        self.blk = nn.Sequential(MLPND(d_in, d_hid, norm=norm, act=act, ndim=2), BasicLayer(d_hid, d_out, norm=norm, act=act, ndim=2), MaxPool(2, keepdim=False))

    def forward(self, f, p):
        u_ce = bpa.bpa_furthest_point_sampling(p, self.g_n)
        p_ce = bpa.bpa_gather_query(p, u_ce)
        u_ne = bpa.bpa_ball_query(p, p_ce, self.g_k, self.g_r)
        p_ne = bpa.bpa_gather_group(p, u_ne) - p_ce.unsqueeze(2)
        f_ne = bpa.bpa_gather_group(f, u_ne)
        f_ce = self.blk(torch.cat([f_ne, p_ne], dim=-1))
        return f_ce, p_ce


class PointNetV11(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_pis, d_stem, d_fis, d_fos = cfg.d_in, cfg.d_stem, [cfg.d_stem, *cfg.d_feats[:-1]], cfg.d_feats
        g_ns, g_ks, g_rs = cfg.g_ns, cfg.g_ks, cfg.g_rs
        n_stg = len(d_fos)
        self.stem = PointClsStem(d_pis, d_stem, cfg.norm, cfg.act)
        self.encoder = nn.ModuleList([PointNetV11Encoder(g_n=g_ns[i], g_k=g_ks[i], g_r=g_rs[i], d_in=d_fis[i]+d_pis, d_hid=d_fis[i], d_out=d_fos[i]) for i in range(n_stg)])
        self.head = PointClsHead(d_fos[3], cfg.n_cls, cfg.norm, cfg.act)

    def forward(self, p):
        f = self.stem(p)
        for en in self.encoder:
            f, p = en(f, p)
        f = self.head(f)
        return f


if __name__ == '__main__':
    import torch
    from easydict import EasyDict
    cfg = EasyDict(dict(n_cls=40, d_in=3, d_stem=32, g_ns=[1024, 256, 64, 16], g_ks=[32, 32, 32, 16], g_rs=[0.1, 0.2, 0.4, 0.8], d_feats=[64, 128, 256, 512], norm='BN', act='ReLU'))
    batch_size, point_size, channel_size, class_size = (1 << 4), (1 << 10), 3, 40
    x, y = torch.rand((batch_size, point_size, channel_size)).cuda(), torch.randint(0, 40, (batch_size,)).long().cuda()
    model = PointNetV11(cfg).cuda()
    z = model(x)
    z.sum().backward()
    print(x.shape, y.shape, z.shape)
