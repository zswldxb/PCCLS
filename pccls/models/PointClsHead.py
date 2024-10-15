import torch.nn as nn
from pccls.models.module_utils.mlp_utils import MLPND
from pccls.models.module_utils.pool_utils import MaxPool


class PointClsHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, norm: str='BN', act: str='ReLU', *, ndim: int=0, channels_first: bool=False) -> None:
        super().__init__()
        self.head = nn.Sequential(
            MaxPool(1, keepdim=False),
            MLPND(d_in, 512, norm=norm, act=act, ndim=ndim, channels_first=channels_first),
            nn.Dropout(0.5),
            MLPND(512, 256, norm=norm, act=act, ndim=ndim, channels_first=channels_first),
            nn.Dropout(0.5),
            nn.Linear(256, d_out, bias=True)
        )

    def forward(self, x):
        return self.head(x)     