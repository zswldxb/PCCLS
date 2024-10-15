#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../bpa_point_aggregation_utils.h"
#include "bpa_three_interpolate.h"

std::vector<at::Tensor> bpa_three_nn_query_gpu(at::Tensor qry, at::Tensor key) {
  CHECK_INPUT(qry); CHECK_INPUT(key);
  at::Tensor dis = torch::full({qry.size(0), qry.size(1), 3}, MAX_INF, at::device(qry.device()).dtype(at::ScalarType::Float));
  at::Tensor idx = torch::zeros({qry.size(0), qry.size(1), 3}, at::device(qry.device()).dtype(at::ScalarType::Long));
  bpa_three_nn_query_kernel_wrapper(qry.size(0), qry.size(2), qry.size(1), key.size(1), qry.data_ptr<PointType>(), key.data_ptr<PointType>(), dis.data_ptr<PointType>(), idx.data_ptr<IndexType>());
  return {dis, idx};
}

at::Tensor bpa_three_interpolate_forward_gpu(at::Tensor src, at::Tensor idx, at::Tensor bse) {
  CHECK_INPUT(src); CHECK_INDEX(idx); CHECK_INPUT(bse);
  at::Tensor dst = torch::zeros({src.size(0), idx.size(1), src.size(2)}, at::device(src.device()).dtype(at::ScalarType::Float));
  bpa_three_interpolate_forward_kernel_wrapper(src.size(0), src.size(2), src.size(1), dst.size(1), src.data_ptr<PointType>(), idx.data_ptr<IndexType>(), bse.data_ptr<PointType>(), dst.data_ptr<PointType>());
  return dst;
}

at::Tensor bpa_three_interpolate_backward_gpu(at::Tensor grad_dst, at::Tensor idx, at::Tensor bse, int p_n) {
  CHECK_INPUT(grad_dst); CHECK_INDEX(idx); CHECK_INPUT(bse);
  at::Tensor grad_src = torch::zeros({grad_dst.size(0), p_n, grad_dst.size(2)}, at::device(grad_dst.device()).dtype(at::ScalarType::Float));
  bpa_three_interpolate_backward_kernel_wrapper(grad_dst.size(0), grad_dst.size(2), p_n, grad_dst.size(1), grad_dst.data_ptr<PointType>(), idx.data_ptr<IndexType>(), bse.data_ptr<PointType>(), grad_src.data_ptr<PointType>());
  return grad_src;
}

