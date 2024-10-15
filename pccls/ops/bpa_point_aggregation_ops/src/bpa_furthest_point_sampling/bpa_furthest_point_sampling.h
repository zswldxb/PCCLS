#pragma once
#include "../bpa_point_aggregation_utils.h"

void bpa_furthest_point_sampling_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, const PointType * pnt, PointType * tmp, IndexType * idx);

at::Tensor bpa_furthest_point_sampling_gpu(at::Tensor pnt, int g_n);
