#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../bpa_point_aggregation_utils.h"
#include "bpa_gather_query.h"

at::Tensor bpa_gather_query_forward_gpu(at::Tensor pnt, at::Tensor idx) {
  CHECK_INPUT(pnt); CHECK_INDEX(idx);
  at::Tensor qry = torch::zeros({pnt.size(0), idx.size(1), pnt.size(2)}, at::device(pnt.device()).dtype(at::ScalarType::Float));
  bpa_gather_query_forward_kernel_wrapper(pnt.size(0), pnt.size(2), pnt.size(1), idx.size(1), pnt.data_ptr<PointType>(), idx.data_ptr<IndexType>(), qry.data_ptr<PointType>());
  return qry;
}

at::Tensor bpa_gather_query_backward_gpu(at::Tensor grad_qry, at::Tensor idx, int p_n) {
  CHECK_INPUT(grad_qry); CHECK_INDEX(idx);
  at::Tensor grad_pnt = torch::zeros({grad_qry.size(0), p_n, grad_qry.size(2)}, at::device(grad_qry.device()).dtype(at::ScalarType::Float));
  bpa_gather_query_backward_kernel_wrapper(grad_qry.size(0), grad_qry.size(2), p_n, idx.size(1), grad_qry.data_ptr<PointType>(), idx.data_ptr<IndexType>(), grad_pnt.data_ptr<PointType>());
  return grad_pnt;
}
