#include <stdio.h>
#include <stdlib.h>
#include "../bpa_point_aggregation_utils.h"
#include "bpa_gather_group.h"

// Input:
//    pnt: (b_s, p_n, c_s)
//    idx: (b_s, g_n, g_k)
// Output:
//    grp: (b_s, g_n, g_k, c_s)
__global__ void bpa_gather_group_forward_kernel(int b_s, int c_s, int p_n, int g_n, int g_k, const PointType * __restrict__ pnt, const IndexType * __restrict__ idx, PointType * __restrict__ grp) {
  int b_i = blockIdx.z;
  int c_i = blockIdx.y;
  int g_a = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_i >= b_s || g_a >= g_n * g_k || c_i >= c_s) return;
  int g_i = g_a / g_k, g_j = g_a % g_k;

  idx += b_i * g_n * g_k + g_i * g_k + g_j;
  pnt += b_i * p_n * c_s + idx[0] * c_s + c_i;
  grp += b_i * g_n * g_k * c_s + g_i * g_k * c_s + g_j * c_s + c_i;
  
  grp[0] = pnt[0];
}

// Input:
//    grad_grp: (b_s, g_n, g_k, c_s)
//    idx: (b_s, g_n, g_k)
// Output:
//    grad_pnt: (b_s, p_n, c_s)
__global__ void bpa_gather_group_backward_kernel(int b_s, int c_s, int p_n, int g_n, int g_k, const PointType * __restrict__ grad_grp, const IndexType * __restrict__ idx, PointType * __restrict__ grad_pnt) {
  int b_i = blockIdx.z;
  int c_i = blockIdx.y;
  int g_a = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_i >= b_s || g_a >= g_n * g_k || c_i >= c_s) return;
  int g_i = g_a / g_k, g_j = g_a % g_k;

  idx += b_i * g_n * g_k + g_i * g_k + g_j;
  grad_grp += b_i * g_n * g_k * c_s + g_i * g_k * c_s + g_j * c_s + c_i;
  grad_pnt += b_i * p_n * c_s + idx[0] * c_s + c_i;

  atomicAdd(grad_pnt, grad_grp[0]);
}

void bpa_gather_group_forward_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, int g_k, const PointType * pnt, const IndexType * idx, PointType * grp) {
  dim3 blocks(DIVUP(g_n * g_k, THREADS_PER_BLOCK), c_s, b_s);
  dim3 threads(THREADS_PER_BLOCK);
  bpa_gather_group_forward_kernel<<<blocks, threads>>>(b_s, c_s, p_n, g_n, g_k, pnt, idx, grp);
  CHECK_ERROR();
}

void bpa_gather_group_backward_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, int g_k, const PointType * grad_grp, const IndexType * idx, PointType * grad_pnt) {
  dim3 blocks(DIVUP(g_n * g_k, THREADS_PER_BLOCK), c_s, b_s);
  dim3 threads(THREADS_PER_BLOCK);
  bpa_gather_group_backward_kernel<<<blocks, threads>>>(b_s, c_s, p_n, g_n, g_k, grad_grp, idx, grad_pnt);
  CHECK_ERROR();
}
