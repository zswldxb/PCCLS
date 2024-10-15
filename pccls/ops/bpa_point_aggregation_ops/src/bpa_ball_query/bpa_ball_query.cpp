#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../bpa_point_aggregation_utils.h"
#include "bpa_ball_query.h"

at::Tensor bpa_ball_query_gpu(at::Tensor pnt, at::Tensor qry, int g_k, float g_r) {
  CHECK_INPUT(pnt); CHECK_INPUT(qry); CHECK_GROUP_SIZE(g_k);
  at::Tensor dis = torch::full({qry.size(0), qry.size(1), g_k}, MAX_INF, at::device(qry.device()).dtype(at::ScalarType::Float));
  at::Tensor idx = torch::zeros({qry.size(0), qry.size(1), g_k}, at::device(qry.device()).dtype(at::ScalarType::Long));
  bpa_ball_query_kernel_wrapper(pnt.size(0), pnt.size(2), pnt.size(1), qry.size(1), idx.size(2), g_r, pnt.data_ptr<PointType>(), qry.data_ptr<PointType>(), dis.data_ptr<PointType>(), idx.data_ptr<IndexType>());
  return idx;
}
