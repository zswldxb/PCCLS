#include <stdio.h>
#include <stdlib.h>
#include "../bpa_point_aggregation_utils.h"
#include "bpa_gather_query.h"

// Input:
//    pnt: (b_s, p_n, c_s)
//    idx: (b_s, g_n)
// Output:
//    qry: (b_s, g_n, c_s)
__global__ void bpa_gather_query_forward_kernel(int b_s, int c_s, int p_n, int g_n, const PointType * __restrict__ pnt, const IndexType * __restrict__ idx, PointType * __restrict__ qry) {
  int b_i = blockIdx.z;
  int c_i = blockIdx.y;
  int g_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_i >= b_s || g_i >= g_n || c_i >= c_s) return;

  idx += b_i * g_n + g_i;
  pnt += b_i * p_n * c_s + idx[0] * c_s + c_i;
  qry += b_i * g_n * c_s + g_i * c_s + c_i;

  qry[0] = pnt[0];
}

// Input:
//    grad_qry: (b_s, g_n, c_s)
//    idx: (b_s, g_n)
// Output:
//    grad_pnt: (b_s, p_n, c_s)
__global__ void bpa_gather_query_backward_kernel(int b_s, int c_s, int p_n, int g_n, const PointType * __restrict__ grad_qry, const IndexType * __restrict__ idx, PointType * __restrict__ grad_pnt) {
  int b_i = blockIdx.z;
  int c_i = blockIdx.y * QUERY_SPLIT;
  int g_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_i >= b_s || g_i >= g_n || c_i >= c_s) return;

  idx += b_i * g_n + g_i;
  grad_qry += b_i * g_n * c_s + g_i * c_s + c_i;
  grad_pnt += b_i * p_n * c_s + idx[0] * c_s + c_i;

  atomicAdd(grad_pnt, grad_qry[0]);
}

void bpa_gather_query_forward_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, const PointType * pnt, const IndexType * idx, PointType * qry) {
  dim3 blocks(DIVUP(g_n, THREADS_PER_BLOCK), c_s, b_s);
  dim3 threads(THREADS_PER_BLOCK);
  bpa_gather_query_forward_kernel<<<blocks, threads>>>(b_s, c_s, p_n, g_n, pnt, idx, qry);
  CHECK_ERROR();
}

void bpa_gather_query_backward_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, const PointType * grad_qry, const IndexType * idx, PointType * grad_pnt) {
  dim3 blocks(DIVUP(g_n, THREADS_PER_BLOCK), c_s, b_s);
  dim3 threads(THREADS_PER_BLOCK);
  bpa_gather_query_backward_kernel<<<blocks, threads>>>(b_s, c_s, p_n, g_n, grad_qry, idx, grad_pnt);
  CHECK_ERROR();
}
