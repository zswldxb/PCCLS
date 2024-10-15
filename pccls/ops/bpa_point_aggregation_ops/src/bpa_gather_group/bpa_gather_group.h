#pragma once
#include "../bpa_point_aggregation_utils.h"

void bpa_gather_group_forward_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, int g_k, const PointType * pnt, const IndexType * idx, PointType * grp);

at::Tensor bpa_gather_group_forward_gpu(at::Tensor pnt, at::Tensor idx);

void bpa_gather_group_backward_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, int g_k, const PointType * grad_grp, const IndexType * idx, PointType * grad_pnt);

at::Tensor bpa_gather_group_backward_gpu(at::Tensor grad_grp, at::Tensor idx, int p_n);

