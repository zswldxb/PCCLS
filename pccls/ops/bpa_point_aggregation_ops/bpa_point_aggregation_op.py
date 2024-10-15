import torch
import warnings
from torch.autograd import Function
from typing import *
Tensor = torch.Tensor


from . import bpa_point_aggregation_cuda


class BPAFurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, pnt: Tensor, grp_tot: int) -> Tensor:
        """
        Input  : pnt: (b_s, p_n, c_s); grp_tot: g_n.
        Output : qry_idx: (b_s, g_n).
        """
        qry_idx = bpa_point_aggregation_cuda.bpa_furthest_point_sampling_gpu(pnt, grp_tot)
        ctx.mark_non_differentiable(qry_idx)
        return qry_idx

    @staticmethod
    def backward(ctx, grad_qry_idx=None):
        return ()


class BPAKNNQuery(Function):
    @staticmethod
    def forward(ctx, pnt, qry, grp_all):
        """
        Input  : pnt: (b_s, p_n, c_s); qry: (b_s, g_n, c_s); grp_all: g_k.
        Output : grp_idx: (b_s, g_n, g_k)
        """
        grp_idx = bpa_point_aggregation_cuda.bpa_knn_query_gpu(pnt, qry, grp_all)
        ctx.mark_non_differentiable(grp_idx)
        return grp_idx

    @staticmethod
    def backward(ctx, grad_grp_idx=None):
        return ()


class BPABallQuery(Function):
    @staticmethod
    def forward(ctx, pnt, qry, grp_all, grp_rad):
        """
        Input  : pnt: (b_s, p_n, c_s); qry: (b_s, g_n, c_s); grp_all: g_k: grp_rad: g_r.
        Output : grp_idx: (b_s, g_n, g_k)
        """
        grp_idx = bpa_point_aggregation_cuda.bpa_ball_query_gpu(pnt, qry, grp_all, grp_rad)
        ctx.mark_non_differentiable(grp_idx)
        return grp_idx

    @staticmethod
    def backward(ctx, grad_idx=None):
        return ()


class BPAGatherQuery(Function):
    @staticmethod
    def forward(ctx, pnt: Tensor, qry_idx: Tensor) -> Tensor:
        """
        Input  : pnt: (b_s, p_n, c_s), qry_idx : (b_s, g_n)
        Output : qry: (b_s, g_n, c_s)
        """
        assert qry_idx.ndim == 2
        qry = bpa_point_aggregation_cuda.bpa_gather_query_forward_gpu(pnt, qry_idx)
        ctx.for_backwards = (qry_idx, pnt.size(1))
        return qry

    @staticmethod
    def backward(ctx, grad_qry: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Input  : grad_qry: (b_s, g_n, c_s)
        Output : grad_pnt: (b_s, p_n, c_s), grad_qry_idx: None
        """
        qry_idx, p_n = ctx.for_backwards
        grad_pnt = bpa_point_aggregation_cuda.bpa_gather_query_backward_gpu(grad_qry, qry_idx, p_n)
        return grad_pnt, None


class BPAGatherGroup(Function):
    @staticmethod
    def forward(ctx, pnt: Tensor, grp_idx: Tensor) -> Tensor:
        """
        Input  : pnt: (b_s, p_n, c_s), idx : (b_s, g_n, g_k)
        Output : grp: (b_s, g_n, g_k, c_s)
        """
        assert grp_idx.ndim == 3
        grp = bpa_point_aggregation_cuda.bpa_gather_group_forward_gpu(pnt, grp_idx)
        ctx.for_backwards = (grp_idx, pnt.size(1))
        return grp

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Input  : grad_grp: (b_s, g_n, g_k, c_s)
        Output : grad_pnt: (b_s, p_n, c_s), grad_grp_idx: None
        """
        grp_idx, p_n = ctx.for_backwards
        grad_pnt = bpa_point_aggregation_cuda.bpa_gather_group_backward_gpu(grad_out, grp_idx, p_n)
        return grad_pnt, None


class BPAThreeNNQuery(Function):
    @staticmethod
    def forward(ctx, qry, key):
        """
        Input  : qry: (b_s, q_n, c_s), key : (b_s, k_n, c_s)
        Output : dis: (b_s, q_n, 3), idx: (b_s, q_n, 3)
        """
        dis, idx = bpa_point_aggregation_cuda.bpa_three_nn_query_gpu(qry, key)
        dis = torch.sqrt(dis)
        ctx.mark_non_differentiable(dis, idx)
        return dis, idx

    @staticmethod
    def backward(ctx, grad_dis, grad_idx):
        return ()


class BPAThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, src, idx, bse):
        """
        Input  : src: (b_s, s_n, c_s), idx: (b_s, d_n, 3), bse: (b_s, d_n, 3)
        Input  : dst: (b_s, d_n, c_s)
        """
        ctx.save_for_backward(idx, bse, src.size(1))
        dst = bpa_point_aggregation_cuda.bpa_three_interpolate_forward_gpu(src, idx, bse)
        return dst

    @staticmethod
    def backward(ctx, grad_dst):
        idx, bse, p_n = ctx.saved_tensors
        grad_pnt = bpa_point_aggregation_cuda.bpa_three_interpolate_backward_gpu(grad_dst, idx, bse, p_n)
        return grad_pnt, None, None


bpa_furthest_point_sampling = BPAFurthestPointSampling.apply
bpa_knn_query = BPAKNNQuery.apply
bpa_ball_query = BPABallQuery.apply
bpa_gather_query = BPAGatherQuery.apply
bpa_gather_group = BPAGatherGroup.apply
bpa_three_query = BPAThreeNNQuery.apply
bpa_three_interpolate = BPAThreeInterpolate.apply