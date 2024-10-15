#include <stdio.h>
#include <stdlib.h>
#include "../bpa_point_aggregation_utils.h"
#include "bpa_furthest_point_sampling.h"

__device__ inline void __update(PointType * __restrict__ dis, IndexType * __restrict__ idx, int t1, int t2) {
  PointType d1 = dis[t1], d2 = dis[t2];
  IndexType i1 = idx[t1], i2 = idx[t2];
  dis[t1] = d1 > d2 ? d1 : d2;
  idx[t1] = d1 > d2 ? i1 : i2;
}

// Input:
//    pnt: (b_s, p_n, c_s)
//    tmp: (b_s, p_n)
// Output:
//    gid: (b_s, g_n)
template <unsigned int SIZE>
__global__ void bpa_furthest_point_sampling_kernel(int b_s, int c_s, int p_n, int g_n, const PointType * __restrict__ pnt, PointType * __restrict__ tmp, IndexType * __restrict__ gid) {
  __shared__ PointType dis[SIZE];
  __shared__ IndexType idx[SIZE];

  int b_i = blockIdx.x;
  int t_i = threadIdx.x;
  pnt += b_i * p_n * c_s;
  tmp += b_i * p_n;
  gid += b_i * g_n;

  int old = 0;
  if (threadIdx.x == 0)
    gid[0] = (IndexType) old;

  __syncthreads();
  for (int g_i = 1; g_i < g_n; ++g_i) {
    const PointType * pnt_old = pnt + old * c_s;
    PointType best_dis = -1.0;
    IndexType best_idx = 0;
    for (int p_i = t_i; p_i < p_n; p_i += SIZE) {
      const PointType * pnt_now = pnt + p_i * c_s;
      PointType dis_now = 0.0;
      for (int c_i = 0; c_i < c_s; ++c_i) {
        dis_now += POW_2(pnt_now[c_i] - pnt_old[c_i]);
      }
      dis_now = min(dis_now, tmp[p_i]);
      tmp[p_i] = dis_now;
      best_dis = best_dis > dis_now ? best_dis : dis_now;
      best_idx = best_dis > dis_now ? best_idx : (IndexType) p_i;
    }
    dis[t_i] = best_dis;
    idx[t_i] = best_idx;
    __syncthreads();
    
    if (SIZE >= 1024) { if (t_i < 512) { __update(dis, idx, t_i, t_i + 512); } __syncthreads(); }
    if (SIZE >=  512) { if (t_i < 256) { __update(dis, idx, t_i, t_i + 256); } __syncthreads(); }
    if (SIZE >=  256) { if (t_i < 128) { __update(dis, idx, t_i, t_i + 128); } __syncthreads(); }
    if (SIZE >=  128) { if (t_i <  64) { __update(dis, idx, t_i, t_i +  64); } __syncthreads(); }
    if (SIZE >=   64) { if (t_i <  32) { __update(dis, idx, t_i, t_i +  32); } __syncthreads(); }
    if (SIZE >=   32) { if (t_i <  16) { __update(dis, idx, t_i, t_i +  16); } __syncthreads(); }
    if (SIZE >=   16) { if (t_i <   8) { __update(dis, idx, t_i, t_i +   8); } __syncthreads(); }
    if (SIZE >=    8) { if (t_i <   4) { __update(dis, idx, t_i, t_i +   4); } __syncthreads(); }
    if (SIZE >=    4) { if (t_i <   2) { __update(dis, idx, t_i, t_i +   2); } __syncthreads(); }
    if (SIZE >=    2) { if (t_i <   1) { __update(dis, idx, t_i, t_i +   1); } __syncthreads(); }

    old = idx[0];
    if (t_i == 0)
      gid[g_i] = (IndexType) old;
  }
}

void bpa_furthest_point_sampling_kernel_wrapper(int b_s, int c_s, int p_n, int g_n, const PointType * pnt, PointType * tmp, IndexType * gid) {
  unsigned int n_threads = opt_n_threads(p_n);

  switch (n_threads) {
    case 1024: bpa_furthest_point_sampling_kernel<1024><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    case  512: bpa_furthest_point_sampling_kernel< 512><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    case  256: bpa_furthest_point_sampling_kernel< 256><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    case  128: bpa_furthest_point_sampling_kernel< 128><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    case   64: bpa_furthest_point_sampling_kernel<  64><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    case   32: bpa_furthest_point_sampling_kernel<  32><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    case   16: bpa_furthest_point_sampling_kernel<  16><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    case    8: bpa_furthest_point_sampling_kernel<   8><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    case    4: bpa_furthest_point_sampling_kernel<   4><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    case    2: bpa_furthest_point_sampling_kernel<   2><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    case    1: bpa_furthest_point_sampling_kernel<   1><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid); break;
    default  : bpa_furthest_point_sampling_kernel<1024><<<b_s, n_threads>>>(b_s, c_s, p_n, g_n, pnt, tmp, gid);
  }

  CHECK_ERROR();
}
