#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../bpa_point_aggregation_utils.h"
#include "bpa_gather_group.h"

at::Tensor bpa_gather_group_forward_gpu(at::Tensor pnt, at::Tensor idx) {
  CHECK_INPUT(pnt); CHECK_INDEX(idx);
  at::Tensor grp = torch::zeros({pnt.size(0), idx.size(1), idx.size(2), pnt.size(2)}, at::device(pnt.device()).dtype(at::ScalarType::Float));
  bpa_gather_group_forward_kernel_wrapper(pnt.size(0), pnt.size(2), pnt.size(1), idx.size(1), idx.size(2), pnt.data_ptr<PointType>(), idx.data_ptr<IndexType>(), grp.data_ptr<PointType>());
  return grp;
}

at::Tensor bpa_gather_group_backward_gpu(at::Tensor grad_grp, at::Tensor idx, int p_n) {
  CHECK_INPUT(grad_grp); CHECK_INDEX(idx);
  at::Tensor grad_pnt = torch::zeros({grad_grp.size(0), p_n, grad_grp.size(3)}, at::device(grad_grp.device()).dtype(at::ScalarType::Float));
  bpa_gather_group_backward_kernel_wrapper(grad_grp.size(0), grad_grp.size(3), p_n, idx.size(1), idx.size(2), grad_grp.data_ptr<PointType>(), idx.data_ptr<IndexType>(), grad_pnt.data_ptr<PointType>());
  return grad_pnt;
}
