#pragma once
#include "../bpa_point_aggregation_utils.h"

void bpa_knn_query_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, int g_k, const PointType * pnt, const PointType * qry, PointType * dis, IndexType * idx);

at::Tensor bpa_knn_query_gpu(at::Tensor pnt, at::Tensor qry, int g_k);
