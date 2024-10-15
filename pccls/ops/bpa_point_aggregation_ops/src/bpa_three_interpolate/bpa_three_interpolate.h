#pragma once
#include "../bpa_point_aggregation_utils.h"

void bpa_three_nn_query_kernel_wrapper(int b_s, int c_s, int q_n, int k_n, const PointType * qry, const PointType * key, PointType * dis, IndexType * idx);

std::vector<at::Tensor> bpa_three_nn_query_gpu(at::Tensor qry, at::Tensor key);

void bpa_three_interpolate_forward_kernel_wrapper(int b_s, int c_s, int s_n, int d_n, const PointType * src, const IndexType * idx, const PointType * bse, PointType * dst);

at::Tensor bpa_three_interpolate_forward_gpu(at::Tensor src, at::Tensor idx, at::Tensor bse);

void bpa_three_interpolate_backward_kernel_wrapper(int b_s, int c_s, int s_n, int d_n, const PointType * grad_dst, const IndexType * idx, const PointType * bse, PointType * grad_src);

at::Tensor bpa_three_interpolate_backward_gpu(at::Tensor grad_dst, at::Tensor idx, at::Tensor bse, int p_n);