#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../bpa_point_aggregation_utils.h"
#include "bpa_knn_query.h"

at::Tensor bpa_knn_query_gpu(at::Tensor pnt, at::Tensor qry, int g_k) {
  CHECK_INPUT(pnt); CHECK_INPUT(qry); CHECK_GROUP_SIZE(g_k);
  at::Tensor dis = torch::full({qry.size(0), qry.size(1), g_k}, MAX_INF, at::device(qry.device()).dtype(at::ScalarType::Float));
  at::Tensor idx = torch::full({qry.size(0), qry.size(1), g_k}, pnt.size(1), at::device(qry.device()).dtype(at::ScalarType::Long));
  bpa_knn_query_kernel_wrapper(pnt.size(0), pnt.size(2), pnt.size(1), qry.size(1), g_k, pnt.data_ptr<PointType>(), qry.data_ptr<PointType>(), dis.data_ptr<PointType>(), idx.data_ptr<IndexType>());
  return idx;
}
