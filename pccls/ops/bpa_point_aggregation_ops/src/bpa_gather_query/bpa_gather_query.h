#pragma once
#include "../bpa_point_aggregation_utils.h"

void bpa_gather_query_forward_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, const PointType * pnt, const IndexType * idx, PointType * qry);

at::Tensor bpa_gather_query_forward_gpu(at::Tensor pnt, at::Tensor idx);

void bpa_gather_query_backward_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, const PointType * grad_qry, const IndexType * idx, PointType * grad_pnt);

at::Tensor bpa_gather_query_backward_gpu(at::Tensor grad_qry, at::Tensor idx, int p_n);
