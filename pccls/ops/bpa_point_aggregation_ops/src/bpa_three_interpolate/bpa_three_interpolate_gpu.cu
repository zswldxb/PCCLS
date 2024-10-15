#include <stdio.h>
#include <stdlib.h>
#include "../bpa_point_aggregation_utils.h"
#include "bpa_three_interpolate.h"

// Input:
//    qry: (b_s, q_n, c_s)
//    key: (b_s, k_n, c_s)
// Output:
//    dis: (b_s, q_n, 3)
//    idx: (b_s, q_n, 3)
__global__ void bpa_three_nn_query_kernel(int b_s, int c_s, int q_n, int k_n, const PointType * __restrict__ qry, const PointType * __restrict__ key, PointType * __restrict__ dis, IndexType * __restrict__ idx) {
  int b_i = blockIdx.y;
  int q_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_i >= b_s || q_i >= q_n) return;

  qry += b_i * q_n * c_s + q_i * c_s;
  key += b_i * k_n * c_s;
  dis += b_i * q_n * 3 + q_i * 3;
  idx += b_i * q_n * 3 + q_i * 3;

  PointType best_dis_1 = MAX_INF, best_dis_2 = MAX_INF, best_dis_3 = MAX_INF;
  IndexType best_idx_1 = 0, best_idx_2 = 0, best_idx_3 = 0;
  for (int k_i = 0; k_i < k_n; ++k_i) {
    const PointType * key_i = key + k_i * c_s;
    PointType d = 0.0;
    for (int c_i = 0; c_i < c_s; ++c_i) {
      d += POW_2(qry[c_i] - key_i[c_i]);
    }
    if (d < best_dis_1) {
      best_dis_3 = best_dis_2; best_idx_3 = best_idx_2;
      best_dis_2 = best_dis_1; best_idx_2 = best_idx_1;
      best_dis_1 = d; best_idx_1 = (IndexType) k_i;
    } 
    else if (d < best_dis_2) {
      best_dis_3 = best_dis_2; best_idx_3 = best_idx_2;
      best_dis_2 = d; best_idx_2 = (IndexType) k_i;
    } 
    else if (d < best_dis_3) {
      best_dis_3 = d; best_idx_3 = (IndexType) k_i;
    }
  }
  dis[0] = best_dis_1; dis[1] = best_dis_2; dis[2] = best_dis_3;
  idx[0] = best_idx_1; idx[1] = best_idx_2; idx[2] = best_idx_3;
}

// Input:
//    src: (b_s, s_n, c_s)
//    idx: (b_s, d_n, 3)
//    bse: (b_s, d_n, 3)
// output:
//    dst: (b_s, d_n, c_s)
__global__ void bpa_three_interpolate_forward_kernel(int b_s, int c_s, int s_n, int d_n, const PointType * __restrict__ src, const IndexType * __restrict__ idx, const PointType * __restrict__ bse, PointType * __restrict__ dst) {
  int b_i = blockIdx.z;
  int c_i = blockIdx.y;
  int d_i = blockIdx.x * blockDim.x + threadIdx.x;

  if (b_i >= b_s || d_i >= d_n || c_i >= c_s) return;

  src += b_i * s_n * c_s + c_i;
  idx += b_i * d_n * 3 + d_i * 3;
  bse += b_i * d_n * 3 + d_i * 3;
  dst += b_i * d_n * c_s + d_i * c_s + c_i;

  dst[0] = bse[0] * src[idx[0] * c_s] + bse[1] * src[idx[1] * c_s] + bse[2] * src[idx[2] * c_s];
}

// Input:
//    grad_dst: (b_s, d_n, c_s)
//    idx: (b_s, d_n, 3)
//    bse: (b_s, d_n, 3)
// Output:
//    grad_src: (b_s, s_n, c_s)
__global__ void bpa_three_interpolate_backward_kernel(int b_s, int c_s, int s_n, int d_n, const PointType * __restrict__ grad_dst, const IndexType * __restrict__ idx, const PointType * __restrict__ bse, PointType * __restrict__ grad_src) {
  int b_i = blockIdx.z;
  int c_i = blockIdx.y;
  int d_i = blockIdx.x * blockDim.x + threadIdx.x;

  if (b_i >= b_s || d_i >= d_n || c_i >= c_s) return;
  
  grad_dst += b_i * d_n * c_s + d_i * c_s + c_i;
  idx += b_i * d_n * 3 + d_i * 3;
  bse += b_i * d_n * 3 + d_i * 3;
  grad_src += b_i * s_n * c_s + c_i;

  atomicAdd(grad_src + idx[0] * c_s, grad_dst[0] * bse[0]);
  atomicAdd(grad_src + idx[1] * c_s, grad_dst[0] * bse[1]);
  atomicAdd(grad_src + idx[2] * c_s, grad_dst[0] * bse[2]);
}

void bpa_three_nn_query_kernel_wrapper(int b_s, int c_s, int q_n, int k_n, const PointType * qry, const PointType * key, PointType * dis, IndexType * idx) {
  dim3 blocks(DIVUP(q_n, THREADS_PER_BLOCK), b_s);
  dim3 threads(THREADS_PER_BLOCK);
  bpa_three_nn_query_kernel<<<blocks, threads>>>(b_s, c_s, q_n, k_n, qry, key, dis, idx);
  CHECK_ERROR();
}

void bpa_three_interpolate_forward_kernel_wrapper(int b_s, int c_s, int s_n, int d_n, const PointType * src, const IndexType * idx, const PointType * bse, PointType * dst) {
  dim3 blocks(DIVUP(d_n, THREADS_PER_BLOCK), c_s, b_s);
  dim3 threads(THREADS_PER_BLOCK);
  bpa_three_interpolate_forward_kernel<<<blocks, threads>>>(b_s, c_s, s_n, d_n, src, idx, bse, dst);
  CHECK_ERROR();
}

void bpa_three_interpolate_backward_kernel_wrapper(int b_s, int c_s, int s_n, int d_n, const PointType * grad_dst, const IndexType * idx, const PointType * bse, PointType * grad_src) {
  dim3 blocks(DIVUP(d_n, THREADS_PER_BLOCK), c_s, b_s);
  dim3 threads(THREADS_PER_BLOCK);
  bpa_three_interpolate_backward_kernel<<<blocks, threads>>>(b_s, c_s, s_n, d_n, grad_dst, idx, bse, grad_src);
  CHECK_ERROR();
}