import torch
import torch.nn as nn
from pccls.models.module_utils.mlp_utils import MLPND
from pccls.models.module_utils.pool_utils import MaxPool
import pccls.ops.bpa_point_aggregation_ops.bpa_point_aggregation_op as bpa


class PointClsStem(nn.Module):
    def __init__(self, d_in: int, d_out: int, norm: str='BN', act: str='ReLU') -> None:
        super().__init__()
        self.g_k = 16
        self.mlp = MLPND(d_in*3, d_out, norm=norm, act=act, ndim=2)
        self.pool = MaxPool(2, keepdim=False)

    def forward(self, x):
        assert x.ndim == 3
        x_ne = bpa.bpa_gather_group(x, bpa.bpa_knn_query(x, x, self.g_k))
        x_ce = x.unsqueeze(2).expand([-1, -1, self.g_k, -1])
        x = self.pool(self.mlp(torch.cat([x_ce, x_ne - x_ce, x_ne], dim=3)))
        return x
