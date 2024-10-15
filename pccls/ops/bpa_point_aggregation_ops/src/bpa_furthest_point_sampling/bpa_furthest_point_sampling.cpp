#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../bpa_point_aggregation_utils.h"
#include "bpa_furthest_point_sampling.h"

at::Tensor bpa_furthest_point_sampling_gpu(at::Tensor pnt, int g_n) {
  CHECK_INPUT(pnt);
  at::Tensor tmp = torch::full({pnt.size(0), pnt.size(1)}, MAX_INF, at::device(pnt.device()).dtype(at::ScalarType::Float));
  at::Tensor idx = torch::zeros({pnt.size(0), g_n}, at::device(pnt.device()).dtype(at::ScalarType::Long));
  bpa_furthest_point_sampling_kernel_wrapper(pnt.size(0), pnt.size(2), pnt.size(1), g_n, pnt.data_ptr<PointType>(), tmp.data_ptr<PointType>(), idx.data_ptr<IndexType>());
  return idx;
}