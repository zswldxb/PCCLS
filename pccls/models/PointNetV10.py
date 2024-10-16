import torch.nn as nn
from typing import List
from functools import partial
from pccls.models.module_utils.mlp_utils import MLPND
from pccls.models.module_utils.pool_utils import MaxPool
from pccls.models.PointClsHead import PointClsHead
import torch
import torch.nn.functional as F
import pccls.ops.bpa_point_aggregation_ops.bpa_point_aggregation_op as bpa


def sample_and_group(fea, pnt, g_n, g_k, g_r):
    qry_idx = bpa.bpa_furthest_point_sampling(pnt, g_n)
    qry_pnt = bpa.bpa_gather_query(pnt, qry_idx)
    grp_idx = bpa.bpa_ball_query(pnt, qry_pnt, g_k, g_r)
    grp_pnt = bpa.bpa_gather_group(pnt, grp_idx) - qry_pnt.unsqueeze(2)

    if fea is None:
        grp_fea = grp_pnt
    else:
        grp_fea = bpa.bpa_gather_group(fea, grp_idx)
        grp_fea = torch.cat([grp_fea, grp_pnt], dim=-1)

    return grp_fea, qry_pnt


class PointNetSetAbstraction(nn.Module):
    def __init__(self, g_n, g_r, g_k, d_in, d_feats, group_all, norm: str='BN', act: str='ReLU'):
        super(PointNetSetAbstraction, self).__init__()
        self.g_n = g_n
        self.g_r = g_r
        self.g_k = g_k
        self.group_all = group_all
        mlp_fn = partial(MLPND, norm=norm, act=act, ndim=2)
        self.blk = nn.Sequential(
            *[mlp_fn(d_in if i == 0 else d_feats[i-1], d_feats[i]) for i in range(len(d_feats))],
            MaxPool(2, keepdim=False)
        )

    def forward(self, f, p):
        if self.group_all:
            f, p = torch.cat([f, p], dim=-1).unsqueeze(1), None
        else:
            f, p = sample_and_group(f, p, self.g_n, self.g_k, self.g_r)
       
        f = self.blk(f)
        return f, p


class PointNetV10(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(g_n=512, g_k=32, g_r=0.2, d_in=cfg.d_in, d_feats=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(g_n=128, g_k=64, g_r=0.4, d_in=128 + 3, d_feats=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(g_n=None, g_r=None, g_k=None, d_in=256 + 3, d_feats=[256, 512, 1024], group_all=True)
        self.head = PointClsHead(1024, cfg.n_cls, cfg.norm, cfg.act)

    def forward(self, p):
        l1_points, l1_xyz = self.sa1(None, p)
        l2_points, l2_xyz = self.sa2(l1_points, l1_xyz)
        l3_points, _ = self.sa3(l2_points, l2_xyz)
        x = self.head(l3_points)
        return x


if __name__ == '__main__':
    import torch
    from easydict import EasyDict
    cfg = EasyDict(dict(n_cls=40, d_in=3, d_feats=[64, 128, 256, 512], norm='BN', act='ReLU'))
    batch_size, point_size, channel_size, class_size = (1 << 4), (1 << 10), 3, 40
    x, y = torch.rand((batch_size, point_size, channel_size)).cuda(), torch.randint(0, 40, (batch_size,)).long().cuda()
    model = PointNetV10(cfg).cuda()
    z = model(x)
    z.sum().backward()
    print(x.shape, y.shape, z.shape)
