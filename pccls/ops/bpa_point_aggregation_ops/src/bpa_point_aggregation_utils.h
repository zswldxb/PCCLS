#ifndef _POINT_AGGREGATION_UTILS_H
#define _POINT_AGGREGATION_UTILS_H

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

typedef float PointType;
typedef int64_t IndexType;

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
#define POW_2(x) ((x) * (x))

#define MAX_INF 1e10
#define HEAP_SIZE 128

inline int opt_n_threads(int work_size) {
  const int p2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return std::max(std::min(1 << p2, THREADS_PER_BLOCK), 1);
}

inline dim3 opt_block_config(int x, int y) {
  const int x_threads = opt_n_threads(x);
  const int y_threads = std::max(std::min(opt_n_threads(y), THREADS_PER_BLOCK / x_threads), 1);
  dim3 block_config(x_threads, y_threads, 1);
  return block_config;
}

#define CHECK_ERROR() do {                                          \
  cudaError_t err = cudaGetLastError();                             \
  if (cudaSuccess != err) {                                         \
    fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
            cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
            __FILE__);                                              \
    exit(-1);                                                       \
  }                                                                 \
} while (0)

#define CHECK_CUDA(x) do { \
  AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor"); \
} while (0)

#define CHECK_CONTIGUOUS(x) do { \
  AT_ASSERT(x.is_contiguous(), #x " must be a contiguous tensor"); \
} while (0)

#define CHECK_INT(x) do { \
  AT_ASSERT(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor"); \
} while (0)

#define CHECK_LONG(x) do { \
  AT_ASSERT(x.scalar_type() == at::ScalarType::Long, #x " must be a long tensor"); \
} while (0)

#define CHECK_FLOAT(x) do { \
  AT_ASSERT(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor"); \
} while (0)

#define CHECK_INPUT(x) do { \
  CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x); \
} while (0)

#define CHECK_INDEX(x) do { \
  CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_LONG(x); \
} while (0)

#define CHECK_GROUP_SIZE(x) do { \
  if (x > HEAP_SIZE) { \
    fprintf(stderr, "%s must less than %d\n", #x, HEAP_SIZE); \
    exit(-1); \
  } \
} while (0)

#endif