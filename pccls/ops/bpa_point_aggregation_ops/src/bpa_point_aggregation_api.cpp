#include <torch/extension.h>
#include "bpa_furthest_point_sampling/bpa_furthest_point_sampling.h"
#include "bpa_knn_query/bpa_knn_query.h"
#include "bpa_ball_query/bpa_ball_query.h"
#include "bpa_gather_query/bpa_gather_query.h"
#include "bpa_gather_group/bpa_gather_group.h"
#include "bpa_three_interpolate/bpa_three_interpolate.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bpa_furthest_point_sampling_gpu", &bpa_furthest_point_sampling_gpu);
  m.def("bpa_knn_query_gpu", &bpa_knn_query_gpu);
  m.def("bpa_ball_query_gpu", &bpa_ball_query_gpu);
  m.def("bpa_gather_query_forward_gpu", &bpa_gather_query_forward_gpu);
  m.def("bpa_gather_query_backward_gpu", &bpa_gather_query_backward_gpu);
  m.def("bpa_gather_group_forward_gpu", &bpa_gather_group_forward_gpu);
  m.def("bpa_gather_group_backward_gpu", &bpa_gather_group_backward_gpu);
  m.def("bpa_three_nn_query_gpu", &bpa_three_nn_query_gpu);
  m.def("bpa_three_interpolate_forward_gpu", &bpa_three_interpolate_forward_gpu);
  m.def("bpa_three_interpolate_backward_gpu", &bpa_three_interpolate_backward_gpu);
}
