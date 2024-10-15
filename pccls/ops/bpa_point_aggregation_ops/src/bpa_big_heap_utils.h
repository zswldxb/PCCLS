#pragma once
#include "bpa_point_aggregation_utils.h"

#define get_lch(pa) (((pa) << 1) + 1)
#define get_rch(pa) (((pa) << 1) + 2)
#define get_pa(ch) (((ch) - 1) >> 1)

template <typename T>
__device__ inline void swap_pointer(T * x, T * y) {
    T tmp = *x;
    *x = *y;
    *y = tmp;
}

__device__ inline void big_heap_push(PointType * dis, IndexType * idx, int val_hp_sz) {
    int ch = val_hp_sz - 1;
    int pa = get_pa(ch);
    while (ch > 0) {
        if (dis[ch] <= dis[pa])
            break;
        swap_pointer(&dis[pa], &dis[ch]);
        swap_pointer(&idx[pa], &idx[ch]);
        ch = pa;
        pa = get_pa(ch);
    }
}

__device__ inline void big_heap_pop(PointType * dis, IndexType * idx, int hp_sz) {
    int pa = 0;
    int ch = get_lch(pa);
    while (ch < hp_sz) {
        if (ch + 1 < hp_sz && dis[ch] <= dis[ch + 1])
            ch++;
        if (dis[pa] >= dis[ch])
            break;
        swap_pointer(&dis[pa], &dis[ch]);
        swap_pointer(&idx[pa], &idx[ch]);
        pa = ch;
        ch = get_lch(pa);
    }
}

__device__ inline void big_heap_sort(PointType * dis, IndexType * idx, int hp_sz, int val_hp_sz) {
    for (int k_i = val_hp_sz - 1; k_i > 0; --k_i) {
        swap_pointer(&dis[0], &dis[k_i]);
        swap_pointer(&idx[0], &idx[k_i]);
        big_heap_pop(dis, idx, k_i);
    }
    for (int k_i = val_hp_sz; k_i < hp_sz; ++k_i) {
        dis[k_i] = dis[0];
        idx[k_i] = idx[0];
    }
}