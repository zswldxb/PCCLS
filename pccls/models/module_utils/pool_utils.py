import torch
import torch.nn as nn


class MaxPool(nn.Module):
    def __init__(self, dim, *, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, feat):
        return torch.max(feat, dim=self.dim, keepdim=self.keepdim)[0]


class AvgPool(nn.Module):
    def __init__(self, dim, *, keepdim):
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, feat):
        return torch.mean(feat, dim=self.dim, keepdim=self.keepdim)


class SumPool(nn.Module):
    def __init__(self, dim, *, keepdim):
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, feat):
        return torch.sum(feat, dim=self.dim, keepdim=self.keepdim)


pool_dict = {'Max': MaxPool, 'Avg': AvgPool, 'Sum': SumPool}
def pool_fn(NAME, **kwargs):
    return pool_dict[NAME](**kwargs)
