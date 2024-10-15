#include <stdio.h>
#include <stdlib.h>
#include "../bpa_point_aggregation_utils.h"
#include "../bpa_big_heap_utils.h"
#include "bpa_knn_query.h"
                                                                                                                                                                                               
// Input:
//    pnt: (b_s, p_n, c_s)
//    qry: (b_s, g_n, c_s)
// Output:
//    dis: (b_s, g_n, g_k)
//    idx: (b_s, g_n, g_k)
__global__ void bpa_knn_query_kernel(int b_s, int c_s, int p_n, int g_n, int g_k, const PointType * __restrict__ pnt, const PointType * __restrict__ qry, PointType * __restrict__ dis, IndexType * __restrict__ idx) {
  int b_i = blockIdx.y;
  int g_i = blockIdx.x * blockDim.x + threadIdx.x;
  if (b_i >= b_s || g_i >= g_n) return;

  pnt += b_i * p_n * c_s;
  qry += b_i * g_n * c_s + g_i * c_s;
  dis += b_i * g_n * g_k + g_i * g_k;
  idx += b_i * g_n * g_k + g_i * g_k;

  PointType best_dis[HEAP_SIZE];
  IndexType best_idx[HEAP_SIZE];
  int val_hp_sz = 0;
  for (int p_i = 0; p_i < p_n; ++p_i) {
    const PointType * pnt_i = pnt + p_i * c_s;
    double tmp = 0.0;
    for (int c_i = 0; c_i < c_s; ++c_i) {
      tmp += POW_2((double)qry[c_i] - pnt_i[c_i]);
    }
    PointType d = tmp;
    if (val_hp_sz < g_k) {
      best_dis[val_hp_sz] = d;
      best_idx[val_hp_sz] = (IndexType) p_i;
      ++val_hp_sz;
      big_heap_push(best_dis, best_idx, val_hp_sz);
    }
    else if (d < best_dis[0]) {
      best_dis[0] = d;
      best_idx[0] = (IndexType) p_i;
      big_heap_pop(best_dis, best_idx, g_k);
    }
  }
  big_heap_sort(best_dis, best_idx, g_k, val_hp_sz);
  for (int k_i = 0; k_i < g_k; ++k_i) {
    dis[k_i] = best_dis[k_i];
    idx[k_i] = best_idx[k_i];
  }
}

void bpa_knn_query_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, int g_k, const PointType * pnt, const PointType * qry, PointType * dis, IndexType * idx) {
  dim3 blocks(DIVUP(g_n, THREADS_PER_BLOCK), b_s);
  dim3 threads(THREADS_PER_BLOCK);
  bpa_knn_query_kernel<<<blocks, threads>>>(b_s, c_s, p_n, g_n, g_k, pnt, qry, dis, idx);
  CHECK_ERROR();
}